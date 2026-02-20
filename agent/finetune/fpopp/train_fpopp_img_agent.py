"""
FPO++ fine-tuning agent for image-based observations.

Extends TrainFPOPPAgent with:
  - Image augmentation (RandomShiftsAug)
  - Gradient accumulation for memory management
  - Values and old CFM losses computed after augmentation (in buffer.update_img)
"""

import logging

import numpy as np
import torch
from tqdm import tqdm

from agent.finetune.fpopp.train_fpopp_agent import TrainFPOPPAgent
from agent.finetune.fpopp.buffer import FPOPPImgBuffer
from model.common.modules import RandomShiftsAug
from model.flow.ft_fpopp.fpopp_flow import FPOPPFlow

log = logging.getLogger(__name__)


class TrainFPOPPImgAgent(TrainFPOPPAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Image augmentation
        self.augment = cfg.train.augment
        self.aug = RandomShiftsAug(pad=4) if self.augment else None

        # Obs dims for image inputs
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}

        # Gradient accumulation
        self.grad_accumulate = cfg.train.grad_accumulate

        self.buffer_device = self.device
        self.fix_nextvalue_augment_bug = True

    def init_buffer(self):
        self.buffer = FPOPPImgBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            n_mc_samples=self.n_mc_samples,
            horizon_steps=self.horizon_steps,
            act_steps=self.act_steps,
            action_dim=self.action_dim,
            n_cond_step=self.n_cond_step,
            obs_dim=self.obs_dims,
            save_full_observation=self.save_full_observations,
            furniture_sparse_reward=self.furniture_sparse_reward,
            best_reward_threshold_for_success=self.best_reward_threshold_for_success,
            reward_scale_running=self.reward_scale_running,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            reward_scale_const=self.reward_scale_const,
            aug=self.aug,
            fix_nextvalue_augment_bug=self.fix_nextvalue_augment_bug,
            device=self.device,
        )

    def run(self):
        self.init_buffer()
        self.prepare_run()
        self.buffer.reset()
        if self.resume:
            self.resume_training()

        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            self.reset_env(buffer_device=self.buffer_device)
            self.buffer.update_full_obs()

            # === PHASE 1: ROLLOUT ===
            for step in (tqdm(range(self.n_steps)) if self.verbose else range(self.n_steps)):
                if not self.verbose and step % 100 == 0:
                    print(f"Processed {step} of {self.n_steps}")
                with torch.no_grad():
                    cond = {
                        key: torch.from_numpy(self.prev_obs_venv[key]).float().to(self.device)
                        for key in self.obs_dims
                    }

                    # Sample actions via Euler integration
                    actions = self.model.get_actions(cond, eval_mode=self.eval_mode)
                    actions_np = actions.cpu().numpy()

                    # Generate MC samples (but don't compute CFM losses yet —
                    # those will be computed after augmentation in buffer.update_img)
                    taus = torch.rand(self.n_envs, self.n_mc_samples, device=self.device)
                    epsilons = torch.randn(
                        self.n_envs, self.n_mc_samples, self.horizon_steps, self.action_dim,
                        device=self.device,
                    )

                # Step environment
                action_venv = actions_np[:, : self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(
                    action_venv
                )

                self.buffer.add(
                    step,
                    self.prev_obs_venv,
                    actions_np,
                    reward_venv,
                    terminated_venv,
                    truncated_venv,
                    taus.cpu().numpy(),
                    epsilons.cpu().numpy(),
                )

                self.prev_obs_venv = obs_venv
                self.cnt_train_step += self.n_envs * self.act_steps if not self.eval_mode else 0

            self.buffer.summarize_episode_reward()

            # === PHASE 2: UPDATE ===
            if not self.eval_mode:
                # Compute values and old CFM losses on augmented images
                self.buffer.update_img(obs_venv, self.model)
                # Sync old policy before PPO updates
                self.model.sync_old_policy()
                self.agent_update(verbose=self.verbose)

            # === PHASE 3: LOGGING ===
            self.log()
            self.update_lr()
            self.save_model()
            self.itr += 1

            self.clear_cache()
            self.inspect_memory()

    def agent_update(self, verbose=True):
        clipfracs_list = []

        for update_epoch, batch_id, minibatch in self.minibatch_generator():
            obs, actions, returns, oldvalues, advantages, taus, epsilons, old_cfm_losses = minibatch

            (pg_loss, v_loss, clipfrac, approx_kl, ratio_mean, Q_values,
             cfm_new_mean, cfm_old_mean, ratio_std, ratio_max, ratio_min,
            ) = self.model.loss(
                obs,
                actions,
                returns,
                oldvalues,
                advantages,
                taus,
                epsilons,
                old_cfm_losses,
                verbose=verbose,
            )
            self.approx_kl = approx_kl

            if verbose:
                log.info(
                    f"epoch={update_epoch}/{self.update_epochs}, batch={batch_id}, "
                    f"ratio={ratio_mean:.3f}, clipfrac={clipfrac:.3f}, kl={approx_kl:.2e}"
                )

            loss = pg_loss + v_loss * self.vf_coef
            clipfracs_list.append(clipfrac)

            loss.backward()

            # Gradient accumulation
            if (batch_id + 1) % self.grad_accumulate == 0:
                actor_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.actor_ft.parameters(), max_norm=float("inf")
                )
                critic_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.critic.parameters(), max_norm=float("inf")
                )

                if self.itr >= self.n_critic_warmup_itr:
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.actor_ft.parameters(), self.max_grad_norm
                        )
                    self.actor_optimizer.step()

                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.critic.parameters(), self.max_grad_norm
                    )
                self.critic_optimizer.step()

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                if verbose:
                    log.info(f"grad update at batch {batch_id}, actor_norm={actor_norm:.2e}")

        self.train_ret_dict = {
            "loss": loss,
            "pg loss": pg_loss,
            "value loss": v_loss,
            "approx kl": self.approx_kl,
            "ratio": ratio_mean,
            "ratio_std": ratio_std,
            "ratio_max": ratio_max,
            "ratio_min": ratio_min,
            "clipfrac": np.mean(clipfracs_list),
            "explained variance": self.explained_var,
            "actor_norm": actor_norm if "actor_norm" in dir() else 0.0,
            "critic_norm": critic_norm if "critic_norm" in dir() else 0.0,
            "actor lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic lr": self.critic_optimizer.param_groups[0]["lr"],
            "Q_values": Q_values,
            "cfm_loss_new": cfm_new_mean,
            "cfm_loss_old": cfm_old_mean,
        }

    def minibatch_generator(self):
        self.approx_kl = 0.0

        obs, actions, returns, oldvalues, advantages, taus, epsilons, old_cfm_losses = (
            self.buffer.make_dataset()
        )
        self.explained_var = self.buffer.get_explained_var(oldvalues, returns)
        self.total_steps = self.n_steps * self.n_envs

        for update_epoch in range(self.update_epochs):
            kl_change_too_much = False
            indices = torch.randperm(self.total_steps, device=self.device)

            for batch_id, start in enumerate(range(0, self.total_steps, self.batch_size)):
                end = start + self.batch_size
                inds_b = indices[start:end]

                minibatch = (
                    {k: obs[k][inds_b] for k in obs},
                    actions[inds_b],
                    returns[inds_b],
                    oldvalues[inds_b],
                    advantages[inds_b],
                    taus[inds_b],
                    epsilons[inds_b],
                    old_cfm_losses[inds_b],
                )

                if (
                    self.target_kl
                    and self.approx_kl > self.target_kl
                    and self.itr >= self.n_critic_warmup_itr
                ):
                    kl_change_too_much = True
                    log.warning(f"KL too large: {self.approx_kl} > {self.target_kl}")
                    break

                yield update_epoch, batch_id, minibatch

            if kl_change_too_much:
                break
