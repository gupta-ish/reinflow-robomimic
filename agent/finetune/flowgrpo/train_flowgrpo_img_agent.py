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
FlowGRPO fine-tuning agent (image observations, CAN task).

Key differences from TrainPPOImgFlowAgent:
  - NO critic optimizer / LR scheduler.
  - Advantages computed via group-relative returns (GRPO), not GAE.
  - Zero-signal filter: skip actor update when std(returns) < 1e-6.
  - Tight PPO clip: 1e-4 (per official FlowGRPO repo).
"""

import os
import copy
import numpy as np
import torch
import pickle
import wandb
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

from agent.finetune.reinflow.train_agent import TrainAgent
from agent.finetune.reinflow.train_ppo_agent import TrainPPOAgent
from agent.finetune.flowgrpo.buffer import FlowGRPOImgBuffer
from model.flow.ft_flowgrpo.flowgrpo_flow import FlowGRPOFlow
from model.common.modules import RandomShiftsAug
from util.scheduler import CosineAnnealingWarmupRestarts, WarmupReduceLROnPlateau
from util.scheduler_simple import CustomScheduler
from util.logging_custom import create_bordered_text
from util.timer import Timer


class TrainFlowGRPOImgAgent(TrainPPOAgent):
    """
    FlowGRPO fine-tuning agent for image-based robomimic tasks.

    Inherits from TrainPPOAgent for:
        prepare_run, set_model_mode, prepare_video_path, reset_env,
        print_architecture, clear_cache, inspect_memory

    Overrides __init__ to bypass critic setup.
    Overrides: visualize_lr, log, save_model, update_lr, resume_training,
               run, agent_update.
    """

    def __init__(self, cfg):
        # Call TrainAgent.__init__ directly to avoid critic crash in
        # TrainPPOAgent.__init__ (which tries to access self.model.critic).
        TrainAgent.__init__(self, cfg)

        # ── Resume path ──────────────────────────────────────────────
        self.resume_path = cfg.get("resume_path", None)
        self.resume = self.resume_path is not None

        # ── Training scalars ─────────────────────────────────────────
        self.gamma = cfg.train.gamma
        self.reward_scale_running = cfg.train.reward_scale_running
        self.reward_scale_const = cfg.train.get("reward_scale_const", 1.0)
        self.target_kl = cfg.train.target_kl
        self.update_epochs = cfg.train.update_epochs
        self.grad_accumulate = cfg.train.grad_accumulate
        self.n_mc_samples = cfg.train.n_mc_samples
        self.adv_clip = cfg.train.adv_clip
        self.verbose = cfg.train.get("verbose", False)
        self.denoising_steps = cfg.model.flow_steps   # for logging compat
        self.skip_initial_eval = False
        self.last_itr_eval = False
        self.current_best_reward = np.float32("-inf")
        self.is_best_so_far = False
        self.total_steps = self.n_steps * self.n_envs
        self.buffer = None
        self.train_ret_dict = {}

        # ── Image augmentation ───────────────────────────────────────
        self.augment = cfg.train.augment
        self.aug = RandomShiftsAug(pad=4) if self.augment else None

        # ── Observation shape meta ───────────────────────────────────
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}

        # ── Actor-only optimizer (NO critic) ─────────────────────────
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor_ft.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        self.actor_lr_type = cfg.train.actor_lr_scheduler.get("type", "cosine")
        if self.actor_lr_type == "cosine":
            self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.actor_optimizer,
                first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.actor_lr,
                min_lr=cfg.train.actor_lr_scheduler.min_lr,
                warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
                gamma=1.0,
            )
        elif self.actor_lr_type == "plateau":
            self.actor_lr_scheduler = WarmupReduceLROnPlateau(
                self.actor_optimizer,
                warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
                target_lr=cfg.train.actor_lr,
                mode="max",
                min_lr=cfg.train.actor_lr_scheduler.min_lr,
                factor=0.6,
                patience=4,
                threshold=20,
                verbose=True,
            )
        elif self.actor_lr_type == "constant_warmup":
            self.actor_lr_scheduler = CustomScheduler(
                self.actor_optimizer,
                "constant_warmup",
                min=cfg.train.actor_lr_scheduler.min_lr,
                warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
                max=cfg.train.actor_lr,
            )
        elif self.actor_lr_type == "cosine_custom":
            self.actor_lr_scheduler = CustomScheduler(
                self.actor_optimizer,
                schedule_type="cosine",
                max=cfg.train.actor_lr,
                hold_steps=cfg.train.actor_lr_scheduler.hold_steps,
                anneal_steps=cfg.train.actor_lr_scheduler.anneal_steps,
                min=cfg.train.actor_lr_scheduler.min_lr,
            )
        else:
            raise ValueError(f"Invalid actor_lr_type: {self.actor_lr_type}")

        # Visualize LR schedule and save plot; does NOT advance actual training
        self.visualize_lr(cfg)

    # ──────────────────────────────────────────────────────────────────
    # LR / saving / resuming (actor-only versions)
    # ──────────────────────────────────────────────────────────────────

    def visualize_lr(self, cfg):
        """Plot actor LR schedule and save PNG. State fully restored after."""
        actor_sched_state = copy.deepcopy(self.actor_lr_scheduler.state_dict())
        actor_opt_state = copy.deepcopy(self.actor_optimizer.state_dict())

        steps, actor_lrs = [], []
        for step in range(cfg.train.n_train_itr):
            self.actor_lr_scheduler.step()
            steps.append(step)
            actor_lrs.append(self.actor_optimizer.param_groups[0]["lr"])

        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(steps, actor_lrs, label="actor", color="blue")
            plt.legend()
            plt.xlabel("iteration")
            plt.ylabel("lr")
            plt.title("FlowGRPO actor LR schedule")
            lr_save_path = os.path.join(self.logdir, "test_lr_schedulers.png")
            plt.savefig(lr_save_path)
            plt.close()
            log.info(f"LR schedule saved to {lr_save_path}")
        except Exception as e:
            log.warning(f"Could not save LR plot: {e}")

        # Restore schedulers / optimizer to pre-preview state
        self.actor_lr_scheduler.load_state_dict(actor_sched_state)
        self.actor_optimizer.load_state_dict(actor_opt_state)

        self.print_architecture()

    def update_lr(self):
        self.actor_lr_scheduler.step()
        log.info(
            f"LR updated. actor_lr={self.actor_optimizer.param_groups[0]['lr']:.2e}"
        )

    def save_model(self):
        data = {
            "itr": self.itr,
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
        }
        # Always save latest checkpoint for resuming
        save_path = os.path.join(self.checkpoint_dir, "last.pt")
        torch.save(data, save_path)

        if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            save_path = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
            torch.save(data, save_path)
            log.info(f"Saved model at itr={self.itr} to {save_path}")

        if self.is_best_so_far:
            save_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(data, save_path)
            log.info(
                f"Saved best model (reward={self.current_best_reward:.4f}) to {save_path}"
            )
            self.is_best_so_far = False

    def resume_training(self):
        log.info("Resuming FlowGRPO training...")
        data = torch.load(self.resume_path, weights_only=True, map_location=self.device)
        log.info(f"Checkpoint keys: {list(data.keys())}")

        self.itr = data["itr"]
        self.cnt_train_step = self.itr * self.n_envs * self.act_steps * self.n_steps
        self.n_train_itr += self.itr

        if "model" in data:
            self.model.load_state_dict(data["model"], strict=True)
            log.info("Loaded full model state dict.")
        else:
            raise ValueError("Checkpoint missing 'model' key.")

        self.actor_optimizer.load_state_dict(data["actor_optimizer"])

        if "actor_lr_scheduler" in data:
            self.actor_lr_scheduler.load_state_dict(data["actor_lr_scheduler"])
        else:
            for _ in range(self.itr):
                self.actor_lr_scheduler.step()
            log.info("Scheduler not found in checkpoint; re-calibrated from scratch.")

        log.info(f"Resumed from itr={self.itr}, total_steps={self.cnt_train_step}.")

    # ──────────────────────────────────────────────────────────────────
    # Buffer initialisation
    # ──────────────────────────────────────────────────────────────────

    def init_buffer(self):
        self.buffer = FlowGRPOImgBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            horizon_steps=self.horizon_steps,
            act_steps=self.act_steps,
            action_dim=self.action_dim,
            n_cond_step=self.n_cond_step,
            obs_dim=self.obs_dims,
            n_mc_samples=self.n_mc_samples,
            save_full_observation=self.save_full_observations,
            furniture_sparse_reward=self.furniture_sparse_reward,
            best_reward_threshold_for_success=self.best_reward_threshold_for_success,
            reward_scale_running=self.reward_scale_running,
            gamma=self.gamma,
            adv_clip=self.adv_clip,
            reward_scale_const=self.reward_scale_const,
            aug=self.aug,
            device=self.device,
        )
        log.info(f"FlowGRPOImgBuffer created. n_mc_samples={self.n_mc_samples}")

    # ──────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────

    def run(self):
        self.init_buffer()
        self.prepare_run()
        self.buffer.reset()
        if self.resume:
            self.resume_training()

        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            self.reset_env()            # resets envs + sets firsts_trajs[0]
            self.buffer.update_full_obs()

            # ── Phase 1: Rollout ──────────────────────────────────────
            for step in (
                tqdm(range(self.n_steps)) if self.verbose else range(self.n_steps)
            ):
                if not self.verbose and step % 100 == 0:
                    print(f"Step {step}/{self.n_steps}", flush=True)

                with torch.no_grad():
                    cond = {
                        key: torch.from_numpy(self.prev_obs_venv[key])
                            .float().to(self.device)
                        for key in self.obs_dims
                    }
                    # Deterministic ODE action
                    actions_venv = self.model.get_actions(cond)  # (E, Ta, Da)

                    # MC samples for CFM ratio (sampled fresh each step)
                    taus = torch.rand(
                        self.n_envs, self.n_mc_samples, device=self.device
                    )
                    epsilons = torch.randn(
                        self.n_envs, self.n_mc_samples,
                        self.horizon_steps, self.action_dim,
                        device=self.device,
                    )

                # Step environment with first act_steps of the action chunk
                action_venv_np = actions_venv.cpu().numpy()
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv_np[:, : self.act_steps])
                )

                self.buffer.add(
                    step,
                    self.prev_obs_venv,
                    action_venv_np,
                    taus.cpu().numpy(),
                    epsilons.cpu().numpy(),
                    reward_venv,
                    terminated_venv,
                    truncated_venv,
                )
                self.prev_obs_venv = obs_venv
                if not self.eval_mode:
                    self.cnt_train_step += self.n_envs * self.act_steps

            self.buffer.summarize_episode_reward()

            # ── Phase 2: Update ───────────────────────────────────────
            if not self.eval_mode:
                self.buffer.update_img(obs_venv, self.model)
                if self.buffer.degenerate:
                    log.warning(
                        f"Itr {self.itr}: zero-signal (std_return={self.buffer.std_return:.2e})"
                        " — skipping actor update."
                    )
                    self.train_ret_dict = {
                        "skipped_update": 1.0,
                        "mean_return": self.buffer.mean_return,
                        "std_return": self.buffer.std_return,
                    }
                else:
                    self.agent_update()

            # ── Phase 3: Logging / bookkeeping ────────────────────────
            self.log()
            self.update_lr()
            self.save_model()
            self.itr += 1
            self.clear_cache()
            self.inspect_memory()

    # ──────────────────────────────────────────────────────────────────
    # Gradient update
    # ──────────────────────────────────────────────────────────────────

    def agent_update(self):
        """
        GRPO policy gradient update with gradient accumulation.

        Draws random minibatches from the buffer and applies the
        FlowGRPO clipped objective.  No value loss.
        """
        obs, actions, advantages, taus, epsilons, old_cfm_losses = (
            self.buffer.make_dataset()
        )

        clipfracs = []
        approx_kls = []
        pg_losses = []
        ratio_means = []
        actor_norm = 0.0

        self.actor_optimizer.zero_grad()
        batch_count = 0

        for update_epoch in range(self.update_epochs):
            kl_stop = False
            indices = torch.randperm(self.total_steps, device=self.device)

            for batch_id, start in enumerate(range(0, self.total_steps, self.batch_size)):
                end = start + self.batch_size
                idx = indices[start:end]

                minibatch_obs = {k: obs[k][idx] for k in obs}
                pg_loss, clipfrac, approx_kl, ratio_mean = self.model.loss(
                    obs=minibatch_obs,
                    actions=actions[idx],
                    advantages=advantages[idx],
                    taus=taus[idx],
                    epsilons=epsilons[idx],
                    old_cfm_losses=old_cfm_losses[idx],
                )

                clipfracs.append(clipfrac)
                approx_kls.append(approx_kl)
                pg_losses.append(pg_loss.item())
                ratio_means.append(ratio_mean)

                # Gradient accumulation
                pg_loss.backward()
                batch_count += 1

                if batch_count % self.grad_accumulate == 0:
                    actor_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.actor_ft.parameters(), float("inf")
                    ).item()
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.actor_ft.parameters(), self.max_grad_norm
                        )
                    self.actor_optimizer.step()
                    self.actor_optimizer.zero_grad()
                    log.info(
                        f"Grad update at epoch={update_epoch} batch={batch_id}, "
                        f"pg_loss={pg_loss.item():.4f}, approx_kl={approx_kl:.4f}, "
                        f"actor_norm={actor_norm:.3e}"
                    )

                # KL early-stopping
                if self.target_kl and approx_kl > self.target_kl:
                    log.warning(
                        f"KL={approx_kl:.4f} > target_kl={self.target_kl} — stopping."
                    )
                    kl_stop = True
                    break

            if kl_stop:
                break

        # Flush any leftover accumulated gradients
        if batch_count % self.grad_accumulate != 0:
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.actor_ft.parameters(), self.max_grad_norm
                )
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()

        self.train_ret_dict = {
            "pg_loss": np.mean(pg_losses),
            "approx_kl": np.mean(approx_kls),
            "clipfrac": np.mean(clipfracs),
            "ratio_mean": np.mean(ratio_means),
            "actor_norm": actor_norm,
            "skipped_update": 0.0,
            # GRPO statistics
            "mean_return": self.buffer.mean_return,
            "std_return": self.buffer.std_return,
            "adv_mean": self.buffer.adv_mean,
            "adv_std": self.buffer.adv_std,
        }

    # ──────────────────────────────────────────────────────────────────
    # Logging (actor-only; no critic lr)
    # ──────────────────────────────────────────────────────────────────

    def log(self, train_prt_str_additional="", train_log_dict_additional={}):
        BOLD = "\033[1m"
        ENDB = "\033[0m"

        self.run_results.append({"itr": self.itr, "step": self.cnt_train_step})

        if self.itr % self.log_freq != 0:
            return

        elapsed = self.timer()
        self.run_results[-1]["time"] = elapsed

        if self.eval_mode:
            log.info(create_bordered_text(
                f"{BOLD}Evaluation at itr {self.itr}{ENDB}:\n"
                f"Method: FlowGRPO | Model: {self.model.__class__.__name__}\n"
                f"Env: {self.env_name} x {self.n_envs}\n"
                f"flow_steps: {self.denoising_steps} | seed: {self.seed}\n"
                f"Success Rate: {self.buffer.success_rate * 100:.2f}%"
                f" ± {self.buffer.std_success_rate * 100:.2f}%\n"
                f"Episode Reward: {self.buffer.avg_episode_reward:.2f}"
                f" ± {self.buffer.std_episode_reward:.2f}\n"
                f"Best Reward: {self.buffer.avg_best_reward:.2f}"
                f" ± {self.buffer.std_best_reward:.2f}\n"
                f"Episode Length: {self.buffer.avg_episode_length:.2f}"
                f" ± {self.buffer.std_episode_length:.2f}\n"
                f"Actor lr: {self.actor_optimizer.param_groups[0]['lr']:.2e}"
            ))
            eval_dict = {
                "eval/success_rate": self.buffer.success_rate,
                "eval/avg_episode_reward": self.buffer.avg_episode_reward,
                "eval/avg_best_reward": self.buffer.avg_best_reward,
                "eval/avg_episode_length": self.buffer.avg_episode_length,
                "eval/num_episode": self.buffer.num_episode_finished,
                "eval/std_success_rate": self.buffer.std_success_rate,
                "eval/std_episode_reward": self.buffer.std_episode_reward,
                "eval/std_best_reward": self.buffer.std_best_reward,
                "eval/std_episode_length": self.buffer.std_episode_length,
            }
            for k, v in eval_dict.items():
                if isinstance(v, torch.Tensor):
                    eval_dict[k] = v.item()
            self.run_results[-1].update(eval_dict)
            if self.use_wandb:
                wandb.log(data=eval_dict, step=self.itr, commit=True)

            if self.current_best_reward < self.buffer.avg_episode_reward:
                self.current_best_reward = self.buffer.avg_episode_reward
                self.is_best_so_far = True
                log.info(f"New best eval reward: {self.current_best_reward:.4f}")

        else:
            # Build log string
            train_str = (
                f"itr {self.itr} | steps {self.cnt_train_step / 1e6:.3f}M"
                f" | time {elapsed:.1f}s\n"
                f"Env: {self.env_name} x {self.n_envs}\n"
                f"Episode Reward: {self.buffer.avg_episode_reward:.2f}"
                f" ± {self.buffer.std_episode_reward:.2f}\n"
                f"Success Rate: {self.buffer.success_rate * 100:.2f}%"
                f" ± {self.buffer.std_success_rate * 100:.2f}%\n"
                f"Best Reward: {self.buffer.avg_best_reward:.2f}"
                f" ± {self.buffer.std_best_reward:.2f}\n"
                f"Episode Length: {self.buffer.avg_episode_length:.2f}"
                f" ± {self.buffer.std_episode_length:.2f}\n"
                f"Actor lr: {self.actor_optimizer.param_groups[0]['lr']:.2e}\n"
            )
            if self.train_ret_dict:
                items = [f"{k}: {v:.4e}" if isinstance(v, float) else
                         f"{k}: {v:.4e}" for k, v in self.train_ret_dict.items()]
                train_str += " | ".join(items) + "\n"
            log.info(train_str + train_prt_str_additional)

            train_log_dict = {
                "train/total_env_steps": self.cnt_train_step,
                "train/success_rate": self.buffer.success_rate,
                "train/avg_episode_reward": self.buffer.avg_episode_reward,
                "train/avg_best_reward": self.buffer.avg_best_reward,
                "train/avg_episode_length": self.buffer.avg_episode_length,
                "train/num_episode": self.buffer.num_episode_finished,
                "train/std_success_rate": self.buffer.std_success_rate,
                "train/std_episode_reward": self.buffer.std_episode_reward,
                "train/std_best_reward": self.buffer.std_best_reward,
                "train/std_episode_length": self.buffer.std_episode_length,
                "train/actor_lr": self.actor_optimizer.param_groups[0]["lr"],
                # GRPO-specific
                "grpo/mean_return": self.buffer.mean_return,
                "grpo/std_return": self.buffer.std_return,
                "grpo/adv_mean": self.buffer.adv_mean,
                "grpo/adv_std": self.buffer.adv_std,
                "grpo/degenerate": float(self.buffer.degenerate),
            }
            loss_dict = {
                "loss/" + k: v for k, v in self.train_ret_dict.items()
            }
            train_log_dict.update(loss_dict)
            train_log_dict.update(train_log_dict_additional or {})

            for k, v in train_log_dict.items():
                if isinstance(v, torch.Tensor):
                    train_log_dict[k] = v.item()

            self.run_results[-1].update(train_log_dict)
            if self.use_wandb:
                wandb.log(data=train_log_dict, step=self.itr, commit=True)

        # Video logging
        if self.render_video and self.use_wandb and self.itr % self.render_freq == 0:
            for env_ind in range(self.n_render):
                video_path = os.path.join(
                    self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                )
                if os.path.exists(video_path):
                    prefix = "eval" if self.eval_mode else "train"
                    wandb.log(
                        {
                            f"{prefix}/video_trial{env_ind}": wandb.Video(
                                video_path, fps=30, format="mp4"
                            )
                        },
                        step=self.itr,
                        commit=False,
                    )
                    log.info(f"Uploaded video: {video_path}")

        with open(self.result_path, "wb") as f:
            pickle.dump(self.run_results, f)
