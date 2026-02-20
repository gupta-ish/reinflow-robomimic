"""
FPO++ replay buffers.

Key difference from ReinFlow buffers:
  - No denoising chains stored (FPO++ doesn't need them)
  - No log-probabilities stored (FPO++ bypasses likelihoods)
  - Stores MC samples (taus, epsilons) and old CFM losses per step
  - Stores clean actions instead of chains
"""

import numpy as np
import torch
import logging

from model.common.critic import ViTCritic
from agent.finetune.reinflow.buffer import PPOBuffer

log = logging.getLogger(__name__)


class FPOPPBuffer(PPOBuffer):
    """On-policy buffer for FPO++ with state inputs and CPU storage."""

    def __init__(
        self,
        n_steps,
        n_envs,
        n_mc_samples,
        horizon_steps,
        act_steps,
        action_dim,
        n_cond_step,
        obs_dim,
        save_full_observation,
        furniture_sparse_reward,
        best_reward_threshold_for_success,
        reward_scale_running,
        gamma,
        gae_lambda,
        reward_scale_const,
        device,
    ):
        super().__init__(
            n_steps=n_steps,
            n_envs=n_envs,
            horizon_steps=horizon_steps,
            act_steps=act_steps,
            action_dim=action_dim,
            n_cond_step=n_cond_step,
            obs_dim=obs_dim,
            save_full_observation=save_full_observation,
            furniture_sparse_reward=furniture_sparse_reward,
            best_reward_threshold_for_success=best_reward_threshold_for_success,
            reward_scale_running=reward_scale_running,
            gamma=gamma,
            gae_lambda=gae_lambda,
            reward_scale_const=reward_scale_const,
            device=device,
        )
        self.n_mc_samples = n_mc_samples

    def reset(self):
        self.obs_trajs = {
            "state": np.zeros((self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim))
        }
        self.samples_trajs = np.zeros((self.n_steps, self.n_envs, self.horizon_steps, self.action_dim))
        self.reward_trajs = np.zeros((self.n_steps, self.n_envs))
        self.terminated_trajs = np.zeros((self.n_steps, self.n_envs))
        self.firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        self.value_trajs = np.empty((self.n_steps, self.n_envs))

        # FPO++ specific: MC samples and old CFM losses
        self.taus_trajs = np.zeros((self.n_steps, self.n_envs, self.n_mc_samples))
        self.epsilons_trajs = np.zeros(
            (self.n_steps, self.n_envs, self.n_mc_samples, self.horizon_steps, self.action_dim)
        )
        self.old_cfm_losses_trajs = np.zeros((self.n_steps, self.n_envs, self.n_mc_samples))

    def add(
        self,
        step,
        state_venv,
        actions_venv,
        reward_venv,
        terminated_venv,
        truncated_venv,
        value_venv,
        taus_venv,
        epsilons_venv,
        old_cfm_losses_venv,
    ):
        done_venv = terminated_venv | truncated_venv

        self.obs_trajs["state"][step] = state_venv
        self.samples_trajs[step] = actions_venv
        self.reward_trajs[step] = reward_venv
        self.terminated_trajs[step] = terminated_venv
        self.firsts_trajs[step + 1] = done_venv
        self.value_trajs[step] = value_venv
        self.taus_trajs[step] = taus_venv
        self.epsilons_trajs[step] = epsilons_venv
        self.old_cfm_losses_trajs[step] = old_cfm_losses_venv

    def make_dataset(self):
        obs = torch.tensor(self.obs_trajs["state"], device=self.device).float().flatten(0, 1)
        actions = torch.tensor(self.samples_trajs, device=self.device).float().flatten(0, 1)
        returns = torch.tensor(self.returns_trajs, device=self.device).float().flatten(0, 1)
        values = torch.tensor(self.value_trajs, device=self.device).float().flatten(0, 1)
        advantages = torch.tensor(self.advantages_trajs, device=self.device).float().flatten(0, 1)
        taus = torch.tensor(self.taus_trajs, device=self.device).float().flatten(0, 1)
        epsilons = torch.tensor(self.epsilons_trajs, device=self.device).float().flatten(0, 1)
        old_cfm_losses = torch.tensor(self.old_cfm_losses_trajs, device=self.device).float().flatten(0, 1)
        return obs, actions, returns, values, advantages, taus, epsilons, old_cfm_losses


class FPOPPImgBuffer(PPOBuffer):
    """On-policy buffer for FPO++ with image inputs and CPU storage.

    Like PPOFlowImgBuffer: values and old_cfm_losses are computed during update
    (after augmentation), not during rollout.
    """

    def __init__(
        self,
        n_steps,
        n_envs,
        n_mc_samples,
        horizon_steps,
        act_steps,
        action_dim,
        n_cond_step,
        obs_dim,
        save_full_observation,
        furniture_sparse_reward,
        best_reward_threshold_for_success,
        reward_scale_running,
        gamma,
        gae_lambda,
        reward_scale_const,
        aug,
        fix_nextvalue_augment_bug,
        device,
    ):
        super().__init__(
            n_steps=n_steps,
            n_envs=n_envs,
            horizon_steps=horizon_steps,
            act_steps=act_steps,
            action_dim=action_dim,
            n_cond_step=n_cond_step,
            obs_dim=obs_dim,
            save_full_observation=save_full_observation,
            furniture_sparse_reward=furniture_sparse_reward,
            best_reward_threshold_for_success=best_reward_threshold_for_success,
            reward_scale_running=reward_scale_running,
            gamma=gamma,
            gae_lambda=gae_lambda,
            reward_scale_const=reward_scale_const,
            device=device,
        )
        self.n_mc_samples = n_mc_samples
        self.aug = aug
        self.fix_nextvalue_augment_bug = fix_nextvalue_augment_bug

    def reset(self):
        # Visual + state observations
        self.obs_trajs = {
            k: np.zeros((self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dim[k]))
            for k in self.obs_dim
        }
        self.samples_trajs = np.zeros(
            (self.n_steps, self.n_envs, self.horizon_steps, self.action_dim)
        )
        self.reward_trajs = np.zeros((self.n_steps, self.n_envs))
        self.terminated_trajs = np.zeros((self.n_steps, self.n_envs))
        self.firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        self.value_trajs = np.empty((self.n_steps, self.n_envs))

        # FPO++ specific
        self.taus_trajs = np.zeros((self.n_steps, self.n_envs, self.n_mc_samples))
        self.epsilons_trajs = np.zeros(
            (self.n_steps, self.n_envs, self.n_mc_samples, self.horizon_steps, self.action_dim)
        )
        self.old_cfm_losses_trajs = np.zeros((self.n_steps, self.n_envs, self.n_mc_samples))

    def add(self, step, prev_obs_venv, actions_venv, reward_venv, terminated_venv, truncated_venv,
            taus_venv, epsilons_venv):
        for k in self.obs_trajs:
            self.obs_trajs[k][step] = prev_obs_venv[k]
        self.samples_trajs[step] = actions_venv
        self.reward_trajs[step] = reward_venv
        self.terminated_trajs[step] = terminated_venv
        self.firsts_trajs[step + 1] = terminated_venv | truncated_venv
        self.taus_trajs[step] = taus_venv
        self.epsilons_trajs[step] = epsilons_venv

    @torch.no_grad()
    def update_img(self, obs_venv, model):
        """Compute values and old CFM losses on augmented images, then GAE."""
        self.normalize_reward()
        self._compute_values_and_cfm(model)
        self._compute_adv_returns(obs_venv, model.critic)

    @torch.no_grad()
    def _compute_values_and_cfm(self, model):
        """Compute values and old CFM losses for each step after augmentation."""
        obs_trajs_ts = {
            key: torch.from_numpy(self.obs_trajs[key]).float().to(self.device)
            for key in self.obs_dim
        }
        if self.aug:
            rgb = obs_trajs_ts["rgb"].flatten(0, 2)  # (s*e*t, C, H, W)
            rgb = self.aug(rgb)
            obs_trajs_ts["rgb"] = rgb.reshape(
                self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dim["rgb"]
            )

        for step in range(self.n_steps):
            cond = {key: obs_trajs_ts[key][step].to(self.device) for key in self.obs_dim}
            actions = torch.from_numpy(self.samples_trajs[step]).float().to(self.device)
            taus = torch.from_numpy(self.taus_trajs[step]).float().to(self.device)
            epsilons = torch.from_numpy(self.epsilons_trajs[step]).float().to(self.device)

            critic: ViTCritic = model.critic
            self.value_trajs[step] = critic.forward(cond, no_augment=True).cpu().numpy().flatten()

            cfm_losses = model.compute_cfm_losses(model.actor_ft, cond, actions, taus, epsilons)
            self.old_cfm_losses_trajs[step] = cfm_losses.cpu().numpy()

    @torch.no_grad()
    def _compute_adv_returns(self, obs_venv, critic):
        """Compute GAE advantages and returns."""
        obs_venv_ts = {
            key: torch.from_numpy(obs_venv[key]).float().to(self.device)
            for key in self.obs_dim
        }
        if self.fix_nextvalue_augment_bug and self.aug:
            rgb = obs_venv_ts["rgb"].flatten(0, 1)
            rgb = self.aug(rgb)
            obs_venv_ts["rgb"] = rgb.reshape(self.n_envs, self.n_cond_step, *self.obs_dim["rgb"])

        self.advantages_trajs = np.zeros((self.n_steps, self.n_envs))
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextvalues = critic.forward(obs_venv_ts, no_augment=True).reshape(1, -1)
                nextvalues = nextvalues.cpu().numpy()
            else:
                nextvalues = self.value_trajs[t + 1]
            non_terminal = 1.0 - self.terminated_trajs[t]
            delta = (
                self.reward_trajs[t] * self.reward_scale_const
                + self.gamma * nextvalues * non_terminal
                - self.value_trajs[t]
            )
            self.advantages_trajs[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * non_terminal * lastgaelam
            )
        self.returns_trajs = self.advantages_trajs + self.value_trajs

    def make_dataset(self):
        obs = {
            "state": torch.tensor(self.obs_trajs["state"], device=self.device).float().flatten(0, 1),
            "rgb": torch.tensor(self.obs_trajs["rgb"], device=self.device).float().flatten(0, 1),
        }
        actions = torch.tensor(self.samples_trajs, device=self.device).float().flatten(0, 1)
        returns = torch.tensor(self.returns_trajs, device=self.device).float().flatten(0, 1)
        values = torch.tensor(self.value_trajs, device=self.device).float().flatten(0, 1)
        advantages = torch.tensor(self.advantages_trajs, device=self.device).float().flatten(0, 1)
        taus = torch.tensor(self.taus_trajs, device=self.device).float().flatten(0, 1)
        epsilons = torch.tensor(self.epsilons_trajs, device=self.device).float().flatten(0, 1)
        old_cfm_losses = torch.tensor(self.old_cfm_losses_trajs, device=self.device).float().flatten(0, 1)
        return obs, actions, returns, values, advantages, taus, epsilons, old_cfm_losses
