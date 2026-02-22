# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
FlowGRPO image-observation buffer.

Key differences from PPOFlowImgBuffer:
  - No value_trajs / logprobs_trajs (no critic).
  - Stores taus, epsilons for CFM ratio computation.
  - Advantages are group-relative: clip((R_i - mean_R) / (std_R + eps), ±adv_clip)
    where R_i is the discounted cumulative reward for env i across n_steps.
  - Zero-signal filter: if std(returns) < 1e-6, sets self.degenerate = True
    and skips computing advantages (caller should skip the actor update).
"""

import numpy as np
import torch
import logging
log = logging.getLogger(__name__)

from agent.finetune.reinflow.buffer import PPOBuffer


class FlowGRPOImgBuffer(PPOBuffer):
    """
    On-policy buffer for FlowGRPO with image observations.

    Stored per step per env (numpy, CPU):
        obs_trajs         : {key: (T, E, To, *shape)}
        samples_trajs     : (T, E, Ta, Da)   — clean actions
        reward_trajs      : (T, E)
        terminated_trajs  : (T, E)
        firsts_trajs      : (T+1, E)
        taus_trajs        : (T, E, N_mc)
        epsilons_trajs    : (T, E, N_mc, Ta, Da)

    Filled in update_img:
        old_cfm_losses_trajs : (T, E, N_mc)
        advantages_trajs     : (T, E)         — constant per env i
    """

    def __init__(
        self,
        n_steps,
        n_envs,
        horizon_steps,
        act_steps,
        action_dim,
        n_cond_step,
        obs_dim,                    # dict: {key: shape_tuple}
        n_mc_samples,
        save_full_observation,
        furniture_sparse_reward,
        best_reward_threshold_for_success,
        reward_scale_running,
        gamma,
        adv_clip,
        reward_scale_const,
        aug,
        device,
    ):
        # PPOBuffer.__init__ expects scalar obs_dim; pass 0 as placeholder
        # (we override all methods that use self.obs_dim as a scalar)
        super().__init__(
            n_steps=n_steps,
            n_envs=n_envs,
            horizon_steps=horizon_steps,
            act_steps=act_steps,
            action_dim=action_dim,
            n_cond_step=n_cond_step,
            obs_dim=obs_dim,        # stored as dict
            save_full_observation=save_full_observation,
            furniture_sparse_reward=furniture_sparse_reward,
            best_reward_threshold_for_success=best_reward_threshold_for_success,
            reward_scale_running=reward_scale_running,
            gamma=gamma,
            gae_lambda=0.95,        # unused for GRPO, kept for compat
            reward_scale_const=reward_scale_const,
            device=device,
        )
        self.obs_dim = obs_dim      # overwrite with the dict version
        self.n_mc_samples = n_mc_samples
        self.adv_clip = adv_clip
        self.aug = aug
        self.degenerate = False

        # Statistics set in update_img (logged by training agent)
        self.mean_return = 0.0
        self.std_return = 0.0
        self.adv_mean = 0.0
        self.adv_std = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Allocate / zero-out all trajectory buffers."""
        self.obs_trajs = {
            k: np.zeros(
                (self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dim[k]),
                dtype=np.float32,
            )
            for k in self.obs_dim
        }
        self.samples_trajs = np.zeros(
            (self.n_steps, self.n_envs, self.horizon_steps, self.action_dim),
            dtype=np.float32,
        )
        self.reward_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.terminated_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs), dtype=np.float32)

        self.taus_trajs = np.zeros(
            (self.n_steps, self.n_envs, self.n_mc_samples), dtype=np.float32
        )
        self.epsilons_trajs = np.zeros(
            (self.n_steps, self.n_envs, self.n_mc_samples, self.horizon_steps, self.action_dim),
            dtype=np.float32,
        )

        # Filled in update_img
        self.old_cfm_losses_trajs = np.zeros(
            (self.n_steps, self.n_envs, self.n_mc_samples), dtype=np.float32
        )
        self.advantages_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.degenerate = False

    def add(self, step, prev_obs_venv, action_samples, taus, epsilons,
            reward_venv, terminated_venv, truncated_venv):
        """Store one environment step.

        Args:
            step          : int
            prev_obs_venv : dict {key: ndarray (E, To, *shape)}
            action_samples: ndarray (E, Ta, Da)
            taus          : ndarray (E, N_mc)
            epsilons      : ndarray (E, N_mc, Ta, Da)
            reward_venv   : ndarray (E,)
            terminated_venv: ndarray (E,)
            truncated_venv : ndarray (E,)
        """
        for k in self.obs_trajs:
            self.obs_trajs[k][step] = prev_obs_venv[k]
        self.samples_trajs[step] = action_samples
        self.taus_trajs[step] = taus
        self.epsilons_trajs[step] = epsilons
        self.reward_trajs[step] = reward_venv
        self.terminated_trajs[step] = terminated_venv
        self.firsts_trajs[step + 1] = terminated_venv | truncated_venv

    # ------------------------------------------------------------------
    # Phase 2: update (called once per iteration, before agent_update)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_img(self, obs_venv: dict, model):
        """
        Compute old CFM losses and group-relative advantages.

        Steps:
          1. normalize_reward()
          2. Convert obs to GPU tensors
          3. Augment RGB in-place (applied once; result used for both old CFM
             loss computation here and gradient updates in loss())
          4. Compute old CFM losses step by step using model.actor_ft
          5. Compute per-env discounted returns
          6. Group-normalize → advantages; set self.degenerate if zero-signal

        Args:
            obs_venv : dict — final observation after rollout (not stored)
            model    : FlowGRPOFlow
        """
        # 1. Running reward normalization
        self.normalize_reward()

        # 2. Convert all obs to GPU tensors
        obs_trajs_ts = {
            key: torch.from_numpy(self.obs_trajs[key]).float().to(self.device)
            for key in self.obs_dim
        }

        # 3. Augment RGB in-place across all (T, E, To) frames at once
        if self.aug and "rgb" in obs_trajs_ts:
            rgb = obs_trajs_ts["rgb"].flatten(0, 2)  # (T*E*To, C, H, W)
            rgb = self.aug(rgb)
            obs_trajs_ts["rgb"] = rgb.reshape(
                self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dim["rgb"]
            )
            # Write augmented RGB back so make_dataset() uses the same crops
            self.obs_trajs["rgb"] = obs_trajs_ts["rgb"].cpu().numpy()

        # 4. Compute old CFM losses per step
        for step in range(self.n_steps):
            cond = {
                key: obs_trajs_ts[key][step]  # (E, To, ...)
                for key in self.obs_dim
            }
            actions_t = torch.from_numpy(self.samples_trajs[step]).float().to(self.device)
            taus_t = torch.from_numpy(self.taus_trajs[step]).float().to(self.device)
            epsilons_t = torch.from_numpy(self.epsilons_trajs[step]).float().to(self.device)

            # Uses actor_ft (= start-of-iteration policy, before gradient updates)
            old_losses = model.compute_cfm_losses(
                model.actor_ft, cond, actions_t, taus_t, epsilons_t
            )  # (E, N_mc)
            self.old_cfm_losses_trajs[step] = old_losses.cpu().numpy()

        # 5. Per-env discounted cumulative returns over the n_steps window
        returns_env = self._compute_discounted_returns()  # (E,)

        # 6. Group-normalize and check for zero-signal
        mean_R = returns_env.mean()
        std_R = returns_env.std()

        self.mean_return = float(mean_R)
        self.std_return = float(std_R)

        if std_R < 1e-6:
            self.degenerate = True
            log.warning(
                "Zero-signal iteration: std(returns) = {:.2e} < 1e-6. "
                "All envs got identical rewards — skipping actor update.".format(std_R)
            )
            return

        self.degenerate = False
        adv_env = np.clip(
            (returns_env - mean_R) / (std_R + 1e-8),
            -self.adv_clip, self.adv_clip,
        )  # (E,)

        self.adv_mean = float(adv_env.mean())
        self.adv_std = float(adv_env.std())

        # 7. Broadcast: same advantage for every timestep of env i
        self.advantages_trajs = np.broadcast_to(
            adv_env[np.newaxis, :], (self.n_steps, self.n_envs)
        ).copy()

    def _compute_discounted_returns(self) -> np.ndarray:
        """
        Per-env discounted sum of rewards over the full n_steps window.
        Episode boundaries within the window are NOT reset — we treat
        the whole n_steps window as one trajectory for the GRPO comparison.

        Returns: ndarray (E,)
        """
        returns = np.zeros(self.n_envs, dtype=np.float64)
        gamma_t = 1.0
        for t in range(self.n_steps):
            returns += gamma_t * self.reward_trajs[t] * self.reward_scale_const
            gamma_t *= self.gamma
        return returns.astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset for gradient updates
    # ------------------------------------------------------------------

    def make_dataset(self):
        """
        Flatten (T, E) → (T*E,) and move to GPU.

        Returns:
            obs            : {'state': (B, To, Do), 'rgb': (B, To, C, H, W)}
            actions        : (B, Ta, Da)
            advantages     : (B,)
            taus           : (B, N_mc)
            epsilons       : (B, N_mc, Ta, Da)
            old_cfm_losses : (B, N_mc)
        """
        obs = {
            k: torch.tensor(self.obs_trajs[k], device=self.device)
                .float().flatten(0, 1)
            for k in self.obs_dim
        }
        actions = torch.tensor(self.samples_trajs, device=self.device).float().flatten(0, 1)
        advantages = torch.tensor(self.advantages_trajs, device=self.device).float().flatten(0, 1)
        taus = torch.tensor(self.taus_trajs, device=self.device).float().flatten(0, 1)
        epsilons = torch.tensor(self.epsilons_trajs, device=self.device).float().flatten(0, 1)
        old_cfm_losses = torch.tensor(
            self.old_cfm_losses_trajs, device=self.device
        ).float().flatten(0, 1)
        return obs, actions, advantages, taus, epsilons, old_cfm_losses
