"""
FPO++ (Flow Policy Optimization++) model for flow policy gradient training.

Implements the algorithm from "Flow Policy Gradients for Robot Control" (Yi, Choi et al., 2026).
Key differences from ReinFlow's PPOFlow:
  - Uses CFM loss differences as a surrogate for likelihood ratios (no explicit likelihoods)
  - Per-sample ratio clipping instead of per-action ratio clipping
  - Asymmetric trust region (ASPO): PPO clipping for positive advantages, SPO for negative
  - No noise injection network — exploration comes from random initial noise in flow integration
  - Zero-sampling at evaluation time (epsilon = 0)
"""

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class FPOPPFlow(nn.Module):
    def __init__(
        self,
        device,
        policy,
        critic,
        actor_policy_path,
        act_dim,
        horizon_steps,
        act_min,
        act_max,
        obs_dim,
        cond_steps,
        # FPO++ specific
        flow_steps,
        n_mc_samples,
        clip_ploss_coef,
        clip_vloss_coef,
        use_aspo,
        cfm_loss_clamp,
        cfm_diff_clamp,
        denoised_clip_value,
    ):
        super().__init__()
        self.device = device
        self.action_dim = act_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = self.horizon_steps * self.action_dim
        self.act_min = act_min
        self.act_max = act_max
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps

        # FPO++ hyperparameters
        self.flow_steps = flow_steps
        self.n_mc_samples = n_mc_samples
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_vloss_coef = clip_vloss_coef
        self.use_aspo = use_aspo
        self.cfm_loss_clamp = cfm_loss_clamp
        self.cfm_diff_clamp = cfm_diff_clamp
        self.denoised_clip_value = denoised_clip_value

        # Actor: load pretrained, then create trainable copy
        self.actor_ft = policy
        self._load_pretrained_policy(actor_policy_path)
        self.actor_ft.to(self.device)

        # Frozen copy for computing old CFM losses
        self.actor_old = copy.deepcopy(self.actor_ft)
        for param in self.actor_old.parameters():
            param.requires_grad = False
        self.actor_old.to(self.device)

        # Critic
        self.critic = critic
        self.critic.to(self.device)

        self._report_params()

    def _load_pretrained_policy(self, network_path):
        if not network_path:
            log.warning("No actor policy path provided. Starting from random init.")
            return
        log.info(f"Loading pretrained policy from {network_path}")
        model_data = torch.load(network_path, map_location=self.device, weights_only=True)
        # Try EMA first, fall back to model weights
        if "ema" in model_data:
            state_dict = {k.replace("network.", ""): v for k, v in model_data["ema"].items()}
            log.info("Loaded EMA weights")
        else:
            state_dict = {k.replace("network.", ""): v for k, v in model_data["model"].items()}
            log.info("Loaded model weights")
        self.actor_ft.load_state_dict(state_dict)

    def _report_params(self):
        log.info(
            f"FPOPPFlow params — Actor: {sum(p.numel() for p in self.actor_ft.parameters()) / 1e6:.2f}M, "
            f"Critic: {sum(p.numel() for p in self.critic.parameters()) / 1e6:.2f}M, "
            f"Total: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M"
        )

    def sync_old_policy(self):
        """Copy current actor_ft weights to actor_old. Called at the start of each PPO iteration."""
        self.actor_old.load_state_dict(self.actor_ft.state_dict())

    # ------------------------------------------------------------------
    # Action sampling via Euler integration
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_actions(self, cond, eval_mode=False):
        """
        Sample actions by Euler integration of the flow ODE.

        Training: epsilon ~ N(0, I)  (random sampling for exploration)
        Eval:     epsilon = 0        (zero-sampling for deterministic behavior)

        Returns:
            actions: (B, Ta, Da)
        """
        B = cond["state"].shape[0]

        if eval_mode:
            xt = torch.zeros(B, self.horizon_steps, self.action_dim, device=self.device)
        else:
            xt = torch.randn(B, self.horizon_steps, self.action_dim, device=self.device)

        dt = 1.0 / self.flow_steps
        steps = torch.linspace(0, 1 - dt, self.flow_steps, device=self.device)

        for i in range(self.flow_steps):
            t = steps[i].expand(B)
            vt = self.actor_ft(xt, t, cond)
            xt = xt + vt * dt
            if self.denoised_clip_value > 0:
                xt = xt.clamp(-self.denoised_clip_value, self.denoised_clip_value)

        xt = xt.clamp(self.act_min, self.act_max)
        return xt

    # ------------------------------------------------------------------
    # CFM loss computation (core of FPO++)
    # ------------------------------------------------------------------
    def compute_cfm_losses(self, actor, cond, actions, taus, epsilons):
        """
        Compute per-sample CFM losses for a batch.

        Args:
            actor: velocity network
            cond: observation dict — state: (B, To, Do), optionally rgb: (B, To, C, H, W)
            actions: clean actions a_t, shape (B, Ta, Da)
            taus: flow timesteps, shape (B, N_mc)
            epsilons: noise samples, shape (B, N_mc, Ta, Da)

        Returns:
            cfm_losses: (B, N_mc) per-sample squared CFM errors
        """
        B, N_mc = taus.shape
        Ta, Da = actions.shape[1], actions.shape[2]

        # Expand actions for MC samples: (B, N_mc, Ta, Da)
        a_expanded = actions.unsqueeze(1).expand(-1, N_mc, -1, -1)

        # Interpolate: a_t^tau = tau * a_t + (1 - tau) * epsilon   (Eq. 5)
        taus_exp = taus.unsqueeze(-1).unsqueeze(-1)  # (B, N_mc, 1, 1)
        a_noised = taus_exp * a_expanded + (1 - taus_exp) * epsilons

        # Velocity target: a_t - epsilon_i   (Eq. 6)
        v_target = a_expanded - epsilons

        # Reshape for batched forward pass: (B*N_mc, Ta, Da)
        a_noised_flat = a_noised.reshape(B * N_mc, Ta, Da)
        taus_flat = taus.reshape(B * N_mc)

        # Expand cond for N_mc samples
        cond_expanded = {}
        for k, v in cond.items():
            # v shape: (B, ...) -> repeat each element N_mc times -> (B*N_mc, ...)
            shape = v.shape
            expanded = v.unsqueeze(1).expand(shape[0], N_mc, *shape[1:])
            cond_expanded[k] = expanded.reshape(B * N_mc, *shape[1:])

        # Forward pass: predict velocity
        v_pred = actor(a_noised_flat, taus_flat, cond_expanded)  # (B*N_mc, Ta, Da)
        v_pred = v_pred.reshape(B, N_mc, Ta, Da)

        # Per-sample CFM loss: ||v_hat - (a_t - epsilon)||^2, mean over (Ta, Da)  (Eq. 8)
        # Mean (not sum) keeps the scale consistent with target_kl and cfm_diff_clamp
        # hyperparameters; with sum the approx_kl proxy is 28x too large, causing
        # early stopping before grad_accumulate=15 is ever satisfied.
        cfm_losses = ((v_pred - v_target) ** 2).mean(dim=(-2, -1))  # (B, N_mc)

        # Clamp for numerical stability
        if self.cfm_loss_clamp > 0:
            cfm_losses = cfm_losses.clamp(0, self.cfm_loss_clamp)

        return cfm_losses

    # ------------------------------------------------------------------
    # FPO++ loss
    # ------------------------------------------------------------------
    def loss(
        self,
        obs,
        actions,
        returns,
        oldvalues,
        advantages,
        taus,
        epsilons,
        old_cfm_losses,
        verbose=True,
    ):
        """
        Compute FPO++ loss with per-sample ratios and ASPO trust region.

        Args:
            obs: dict with key state (B, To, Do), optionally rgb
            actions: (B, Ta, Da)
            returns: (B,)
            oldvalues: (B,)
            advantages: (B,)
            taus: (B, N_mc)
            epsilons: (B, N_mc, Ta, Da)
            old_cfm_losses: (B, N_mc)

        Returns:
            pg_loss, v_loss, clipfrac, approx_kl, ratio_mean, newvalues_mean
        """
        # 1. Compute new CFM losses under current policy
        new_cfm_losses = self.compute_cfm_losses(self.actor_ft, obs, actions, taus, epsilons)

        # 2. Per-sample ratios (Eq. 10)
        # rho_i = exp(ell_old_i - ell_new_i)
        cfm_diff = old_cfm_losses - new_cfm_losses  # (B, N_mc)

        # Clamp difference before exponentiation for numerical stability
        if self.cfm_diff_clamp > 0:
            cfm_diff = cfm_diff.clamp(-self.cfm_diff_clamp, self.cfm_diff_clamp)

        ratios = torch.exp(cfm_diff)  # (B, N_mc)

        # 3. Normalize advantages
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        adv_expanded = adv.unsqueeze(1).expand_as(ratios)  # (B, N_mc)

        # 4. Trust region objective (Eq. 12-13)
        eps = self.clip_ploss_coef

        if self.use_aspo:
            # Positive advantages: standard PPO clipping (Eq. 1)
            pg_ppo1 = -adv_expanded * ratios
            pg_ppo2 = -adv_expanded * torch.clamp(ratios, 1.0 - eps, 1.0 + eps)
            ppo_loss = torch.max(pg_ppo1, pg_ppo2)

            # Negative advantages: SPO objective (Eq. 11)
            # psi_SPO = rho * A - (|A| / (2*eps)) * (rho - 1)^2
            # We negate because we minimize: -psi_SPO
            spo_loss = -(adv_expanded * ratios - (adv_expanded.abs() / (2 * eps)) * (ratios - 1.0) ** 2)

            # Select based on advantage sign (Eq. 12)
            pos_mask = (adv_expanded >= 0).float()
            pg_loss_per_sample = pos_mask * ppo_loss + (1 - pos_mask) * spo_loss
        else:
            # Standard PPO clipping for all advantages
            pg_loss1 = -adv_expanded * ratios
            pg_loss2 = -adv_expanded * torch.clamp(ratios, 1.0 - eps, 1.0 + eps)
            pg_loss_per_sample = torch.max(pg_loss1, pg_loss2)

        pg_loss = pg_loss_per_sample.mean()

        # 5. Value function loss
        newvalues = self.critic(obs).view(-1)
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        if self.clip_vloss_coef:
            v_clipped = torch.clamp(newvalues, oldvalues - self.clip_vloss_coef, oldvalues + self.clip_vloss_coef)
            v_loss = 0.5 * torch.max((newvalues - returns) ** 2, (v_clipped - returns) ** 2).mean()

        # 6. Logging info
        with torch.no_grad():
            clipfrac = ((ratios - 1.0).abs() > eps).float().mean().item()
            approx_kl = cfm_diff.mean().item()
            ratio_mean = ratios.mean().item()
            cfm_new_mean = new_cfm_losses.mean().item()
            cfm_old_mean = old_cfm_losses.mean().item()
            ratio_std = ratios.std().item()
            ratio_max = ratios.max().item()
            ratio_min = ratios.min().item()

        if verbose:
            log.info(
                f"ratio={ratio_mean:.3f}, clipfrac={clipfrac:.3f}, approx_kl={approx_kl:.2e}, "
                f"cfm_new={cfm_new_mean:.4f}, cfm_old={cfm_old_mean:.4f}"
            )

        return (
            pg_loss,
            v_loss,
            clipfrac,
            approx_kl,
            ratio_mean,
            newvalues.mean().item(),
            cfm_new_mean,
            cfm_old_mean,
            ratio_std,
            ratio_max,
            ratio_min,
        )
