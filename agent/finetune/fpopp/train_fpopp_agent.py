"""
FPO++ fine-tuning agent for state-based observations.

Mirrors TrainPPOFlowAgent but uses CFM-loss-based ratios instead of
log-probability-based ratios. No noise injection network is used.
"""

import os
import logging

import numpy as np
import torch

from agent.finetune.reinflow.train_ppo_agent import TrainPPOAgent
from agent.finetune.fpopp.buffer import FPOPPBuffer
from model.flow.ft_fpopp.fpopp_flow import FPOPPFlow

log = logging.getLogger(__name__)


class TrainFPOPPAgent(TrainPPOAgent):
    def __init__(self, cfg):
        # TrainPPOAgent.__init__ calls hydra.utils.instantiate(cfg.model)
        # which creates FPOPPFlow and sets self.model.
        # It also expects self.model.actor_ft for the optimizer.
        super().__init__(cfg)

        self.model: FPOPPFlow

        # FPO++ specific config
        self.flow_steps = cfg.model.flow_steps
        self.n_mc_samples = cfg.model.n_mc_samples
        self.use_aspo = cfg.model.use_aspo
        self.skip_initial_eval = cfg.get("skip_initial_eval", False)

    def init_buffer(self):
        self.buffer = FPOPPBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            n_mc_samples=self.n_mc_samples,
            horizon_steps=self.horizon_steps,
            act_steps=self.act_steps,
            action_dim=self.action_dim,
            n_cond_step=self.n_cond_step,
            obs_dim=self.obs_dim,
            save_full_observation=self.save_full_observations,
            furniture_sparse_reward=self.furniture_sparse_reward,
            best_reward_threshold_for_success=self.best_reward_threshold_for_success,
            reward_scale_running=self.reward_scale_running,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            reward_scale_const=self.reward_scale_const,
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
            self.reset_env()
            self.buffer.update_full_obs()

            # === PHASE 1: ROLLOUT ===
            for step in range(self.n_steps):
                with torch.no_grad():
                    cond = {
                        "state": torch.tensor(
                            self.prev_obs_venv["state"], device=self.device, dtype=torch.float32
                        )
                    }
                    # Value estimate
                    value_venv = self.model.critic(cond).cpu().numpy().flatten()

                    # Sample actions via Euler integration
                    actions = self.model.get_actions(cond, eval_mode=self.eval_mode)
                    actions_np = actions.cpu().numpy()

                    # Generate MC samples for FPO++ ratio computation
                    taus = torch.rand(self.n_envs, self.n_mc_samples, device=self.device)
                    epsilons = torch.randn(
                        self.n_envs, self.n_mc_samples, self.horizon_steps, self.action_dim,
                        device=self.device,
                    )

                    # Compute old CFM losses under current policy
                    old_cfm_losses = self.model.compute_cfm_losses(
                        self.model.actor_ft, cond, actions, taus, epsilons
                    )

                    taus_np = taus.cpu().numpy()
                    epsilons_np = epsilons.cpu().numpy()
                    old_cfm_losses_np = old_cfm_losses.cpu().numpy()

                # Step environment
                action_venv = actions_np[:, : self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(
                    action_venv
                )

                self.buffer.save_full_obs(info_venv)
                self.buffer.add(
                    step,
                    self.prev_obs_venv["state"],
                    actions_np,
                    reward_venv,
                    terminated_venv,
                    truncated_venv,
                    value_venv,
                    taus_np,
                    epsilons_np,
                    old_cfm_losses_np,
                )

                self.prev_obs_venv = obs_venv
                self.cnt_train_step += self.n_envs * self.act_steps if not self.eval_mode else 0

            self.buffer.summarize_episode_reward()

            # === PHASE 2: UPDATE ===
            if not self.eval_mode:
                self.buffer.update(obs_venv, self.model.critic)
                # Sync old policy before PPO updates
                self.model.sync_old_policy()
                self.agent_update(verbose=self.verbose)

            # === PHASE 3: LOGGING ===
            self.log()
            self.update_lr()
            self.save_model()
            self.itr += 1

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

            # Optimize
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()

            # Gradient norms for logging
            actor_norm = torch.nn.utils.clip_grad_norm_(
                self.model.actor_ft.parameters(), max_norm=float("inf")
            )
            critic_norm = torch.nn.utils.clip_grad_norm_(
                self.model.critic.parameters(), max_norm=float("inf")
            )

            # Critic always updates
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            # Actor updates after critic warmup
            if self.itr >= self.n_critic_warmup_itr:
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.actor_ft.parameters(), self.max_grad_norm
                    )
                self.actor_optimizer.step()

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
            "actor_norm": actor_norm,
            "critic_norm": critic_norm,
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
                    {"state": obs[inds_b]},
                    actions[inds_b],
                    returns[inds_b],
                    oldvalues[inds_b],
                    advantages[inds_b],
                    taus[inds_b],
                    epsilons[inds_b],
                    old_cfm_losses[inds_b],
                )

                if self.target_kl and self.approx_kl > self.target_kl:
                    kl_change_too_much = True
                    log.warning(
                        f"KL too large: {self.approx_kl} > {self.target_kl}, stopping."
                    )
                    break

                yield update_epoch, batch_id, minibatch

            if kl_change_too_much:
                break

    # ------------------------------------------------------------------
    # Save / resume
    # ------------------------------------------------------------------
    def save_model(self):
        policy_state_dict = {
            "network." + key: value for key, value in self.model.actor_ft.state_dict().items()
        }

        data = {
            "itr": self.itr,
            "cnt_train_steps": self.cnt_train_step,
            "model": self.model.state_dict(),
            "policy": policy_state_dict,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
            "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
        }

        save_path = os.path.join(self.checkpoint_dir, "last.pt")
        torch.save(data, save_path)

        if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            save_path = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
            torch.save(data, save_path)
            log.info(f"Saved model at itr={self.itr} to {save_path}")

        if self.is_best_so_far:
            save_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(data, save_path)
            log.info(f"Saved best model (reward={self.current_best_reward:.3f}) to {save_path}")
            self.is_best_so_far = False

    def resume_training(self):
        super().resume_training()

    def update_lr(self):
        self.critic_lr_scheduler.step()
        if self.itr >= self.n_critic_warmup_itr:
            self.actor_lr_scheduler.step()
        log.info(
            f"lr updated: actor={self.actor_optimizer.param_groups[0]['lr']:.2e}, "
            f"critic={self.critic_optimizer.param_groups[0]['lr']:.2e}"
        )
