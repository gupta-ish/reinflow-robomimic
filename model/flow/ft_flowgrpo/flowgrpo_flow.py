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
FlowGRPO model: VisionFlowMLP actor (no critic, no noise network).

Probability ratio via CFM loss surrogate:
    rho^(k) = exp( l_old^(k) - l_new^(k) )
    l^(k)   = mean_{Ta,Da} || v_theta( tau*a + (1-tau)*eps, tau ; s ) - (a - eps) ||^2

GRPO clipped objective:
    L = E[-min( rho*A, clip(rho, 1-eps_clip, 1+eps_clip)*A )]
    where A = group-relative advantage (constant per env trajectory).
"""

import copy
import torch
import torch.nn as nn
import logging
log = logging.getLogger(__name__)
from model.flow.mlp_flow import VisionFlowMLP


class FlowGRPOFlow(nn.Module):
    """
    Flow policy model for FlowGRPO fine-tuning.

    Components:
        actor_ft   : VisionFlowMLP, trainable parameters
        actor_old  : frozen deepcopy, used for computing old CFM losses

    No critic — advantages come from group-relative returns.
    """

    def __init__(
        self,
        device: str,
        policy: VisionFlowMLP,          # instantiated by hydra
        actor_policy_path: str,
        act_dim: int,
        horizon_steps: int,
        act_min: float,
        act_max: float,
        obs_dim: int,
        cond_steps: int,
        flow_steps: int,                # ODE Euler integration steps
        n_mc_samples: int,              # MC samples for CFM ratio per (step, env)
        clip_ploss_coef: float,         # PPO clip range (1e-4 per official FlowGRPO)
        adv_clip: float,                # advantage clip magnitude (5.0 per official)
        cfm_diff_clamp: float,          # clamp on (l_old - l_new) before exp
    ):
        super().__init__()
        self.device = device
        self.flow_steps = flow_steps
        self.action_dim = act_dim
        self.horizon_steps = horizon_steps
        self.act_min = act_min
        self.act_max = act_max
        self.clip_ploss_coef = clip_ploss_coef
        self.adv_clip = adv_clip
        self.cfm_diff_clamp = cfm_diff_clamp
        self.n_mc_samples = n_mc_samples

        # Load pretrained weights into actor_old (frozen reference copy).
        # actor_old is used in update_img to compute old CFM losses.
        self.actor_old: VisionFlowMLP = policy
        self._load_pretrained_policy(actor_policy_path)
        for param in self.actor_old.parameters():
            param.requires_grad = False
        self.actor_old.to(self.device)

        # actor_ft is the trainable copy for fine-tuning.
        self.actor_ft: VisionFlowMLP = copy.deepcopy(self.actor_old)
        for param in self.actor_ft.parameters():
            param.requires_grad = True
        self.actor_ft.to(self.device)

        n_actor = sum(p.numel() for p in self.actor_ft.parameters()) / 1e6
        log.info(
            f"FlowGRPOFlow initialized. "
            f"actor_ft: {n_actor:.2f} M params. No critic."
        )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _load_pretrained_policy(self, path: str):
        if not path:
            log.warning("No policy path provided — starting from random init.")
            return
        log.info(f"Loading pretrained policy (EMA) from {path}")
        data = torch.load(path, map_location=self.device, weights_only=True)
        # The pre-training checkpoint uses keys prefixed with "network."
        ema_data = {k.replace("network.", ""): v for k, v in data["ema"].items()}
        self.actor_old.load_state_dict(ema_data)
        log.info("Pretrained EMA policy loaded successfully.")

    @torch.no_grad()
    def sync_old_policy(self):
        """Copy actor_ft → actor_old (called at the start of each update phase)."""
        self.actor_old.load_state_dict(self.actor_ft.state_dict())

    # ------------------------------------------------------------------
    # Action sampling (deterministic ODE, no noise network)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_actions(self, cond: dict):
        """
        Deterministic Euler integration of the flow ODE.

        Args:
            cond: {'state': (B, To, Do), 'rgb': (B, To, C, H, W)}

        Returns:
            actions: (B, Ta, Da) clipped to [act_min, act_max]
        """
        B = cond["state"].shape[0]
        dt = 1.0 / self.flow_steps
        steps = torch.linspace(
            0, 1 - dt, self.flow_steps, device=self.device
        ).unsqueeze(0).expand(B, -1)           # (B, flow_steps)

        # Initial noise
        x = torch.randn(B, self.horizon_steps, self.action_dim, device=self.device)

        for i in range(self.flow_steps):
            t = steps[:, i]                    # (B,)
            vt = self.actor_ft.forward(x, t, cond)   # (B, Ta, Da)
            x = x + vt * dt
            x = x.clamp(self.act_min, self.act_max)

        return x                               # (B, Ta, Da)

    # ------------------------------------------------------------------
    # CFM loss computation
    # ------------------------------------------------------------------

    def compute_cfm_losses(
        self,
        actor: VisionFlowMLP,
        cond: dict,
        actions: torch.Tensor,
        taus: torch.Tensor,
        epsilons: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-MC-sample CFM loss (mean over action dims).

        Args:
            actor    : VisionFlowMLP (actor_ft or actor_old)
            cond     : {'state': (B, To, Do), 'rgb': (B, To, C, H, W)}
            actions  : (B, Ta, Da)  — clean action from rollout
            taus     : (B, N_mc)    — flow matching times in [0,1]
            epsilons : (B, N_mc, Ta, Da) — initial noise samples

        Returns:
            cfm_losses: (B, N_mc)
        """
        B, N_mc = taus.shape
        cfm_loss_list = []
        for k in range(N_mc):
            tau_k = taus[:, k]                      # (B,)
            eps_k = epsilons[:, k]                  # (B, Ta, Da)
            # Interpolated point: x_t = t*a + (1-t)*eps
            x_tau = (
                tau_k.view(B, 1, 1) * actions
                + (1 - tau_k.view(B, 1, 1)) * eps_k
            )
            # Target velocity: v* = a - eps (from noise to data)
            target_vel = actions - eps_k            # (B, Ta, Da)
            # Predicted velocity
            pred_vel = actor.forward(x_tau, tau_k, cond)  # (B, Ta, Da)
            # Mean over (Ta, Da) — crucial: mean, NOT sum (keeps scale consistent)
            cfm_loss_k = ((pred_vel - target_vel) ** 2).mean(dim=(-2, -1))  # (B,)
            cfm_loss_list.append(cfm_loss_k)
        return torch.stack(cfm_loss_list, dim=1)   # (B, N_mc)

    # ------------------------------------------------------------------
    # GRPO loss
    # ------------------------------------------------------------------

    def loss(
        self,
        obs: dict,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        taus: torch.Tensor,
        epsilons: torch.Tensor,
        old_cfm_losses: torch.Tensor,
    ):
        """
        FlowGRPO clipped policy loss.

        Args:
            obs           : {'state': (B, To, Do), 'rgb': (B, To, C, H, W)}
            actions       : (B, Ta, Da)
            advantages    : (B,)  — group-relative, already clipped to ±adv_clip
            taus          : (B, N_mc)
            epsilons      : (B, N_mc, Ta, Da)
            old_cfm_losses: (B, N_mc)  — computed before gradient updates

        Returns:
            pg_loss    : scalar
            clipfrac   : float
            approx_kl  : float
            ratio_mean : float
        """
        # New CFM losses (gradient flows through actor_ft)
        new_cfm_losses = self.compute_cfm_losses(
            self.actor_ft, obs, actions, taus, epsilons
        )   # (B, N_mc)

        # Clamp difference for numerical stability, then exponentiate
        # rho = exp(l_old - l_new): higher old loss → new policy more likely → ratio > 1
        cfm_diff = (old_cfm_losses - new_cfm_losses).clamp(
            -self.cfm_diff_clamp, self.cfm_diff_clamp
        )                                           # (B, N_mc)
        ratios = torch.exp(cfm_diff)                # (B, N_mc)

        # Expand advantages: same advantage for all N_mc samples of the same (step, env)
        adv = advantages.unsqueeze(1).expand_as(ratios)  # (B, N_mc)

        # Clipped PPO surrogate
        pg1 = -adv * ratios
        pg2 = -adv * ratios.clamp(
            1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef
        )
        pg_loss = torch.max(pg1, pg2).mean()

        with torch.no_grad():
            clipfrac = (
                (ratios - 1.0).abs() > self.clip_ploss_coef
            ).float().mean().item()
            approx_kl = ((ratios - 1) - cfm_diff).mean().item()
            ratio_mean = ratios.mean().item()

        return pg_loss, clipfrac, approx_kl, ratio_mean
