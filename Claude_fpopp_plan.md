# FPO++ Implementation Plan for ReinFlow-RoboMimic

## 1. Overview

**Goal:** Implement the FPO++ algorithm from "Flow Policy Gradients for Robot Control" (Yi, Choi et al., 2026) within the existing ReinFlow-RoboMimic codebase, so that FPO++ fine-tuning can be launched and evaluated identically to how ReinFlow fine-tuning currently works.

**Key Insight:** FPO++ is fundamentally different from ReinFlow in how it handles the likelihood ratio. ReinFlow computes explicit log-probabilities by injecting learnable noise at each denoising step and treating the flow as a chain of Gaussian transitions. FPO++ bypasses likelihood computation entirely, instead using conditional flow matching (CFM) loss differences as a surrogate for the log-likelihood ratio. This is the core conceptual difference that drives all implementation choices.

---

## 2. FPO++ Algorithm: Complete Mathematical Formulation

### 2.1 Background: PPO Objective

Standard PPO with clipped surrogate:

```
ψ_PPO(ρ_θ, Â_t) = min(ρ_θ · Â_t, clip(ρ_θ, 1 ± ε^clip) · Â_t)     (Eq. 1)

max_θ  E_{π_θ_old} [ ψ_PPO(ρ_θ, Â_t) ]                                (Eq. 2)

where ρ_θ = π_θ(a_t | o_t) / π_θ_old(a_t | o_t)
```

### 2.2 FPO Ratio Surrogate (Original FPO)

Instead of computing the true likelihood ratio ρ_θ, FPO uses CFM loss differences:

```
ρ̂_FPO(θ) = exp( L̂_CFM,θ_old(a_t; o_t) − L̂_CFM,θ(a_t; o_t) )       (Eq. 3)
```

The CFM loss is estimated via Monte Carlo with N_mc samples:

```
L̂_CFM,θ(a_t; o_t) = (1/N_mc) Σ_{i=1}^{N_mc} ℓ_θ^{(i,t)}            (Eq. 7)

ℓ_θ^{(i,t)} = ℓ_θ(a_t, τ_i, ε_i; o_t) = ‖v̂_θ(a_t^τ_i, τ_i; o_t) − (a_t − ε_i)‖₂²   (Eq. 8)
```

where:
- `ε_i ~ N(0, I)` are Monte Carlo noise samples
- `τ_i ~ U[0, 1]` are flow timesteps
- `a_t^τ_i = τ_i · a_t + (1 − τ_i) · ε_i` is the interpolated (noised) action  (Eq. 5)
- `v̂_θ` is the learned velocity field
- The velocity target is `a_t − ε_i`  (Eq. 6)

The original FPO computes a **single per-action ratio** by averaging CFM losses across samples before exponentiating:

```
ρ̂_FPO(θ) = exp( (1/N_mc) Σ_{i=1}^{N_mc} (ℓ_{θ_old}^{(i,t)} − ℓ_θ^{(i,t)}) )   (Eq. 9)
```

### 2.3 FPO++ Modification 1: Per-Sample Ratio

FPO++ computes a **separate ratio for each Monte Carlo sample** `(τ_i, ε_i)`:

```
ρ̂_FPO++^{(i)}(θ) = exp( ℓ_{θ_old}^{(i,t)} − ℓ_θ^{(i,t)} )           (Eq. 10)
```

**Why this matters:** In original FPO, ratios are clipped *after* averaging across samples, meaning either all or no samples get clipped for a given action. Per-sample ratios provide a finer-grained trust region — each `(τ_i, ε_i)` pair can be clipped independently, reducing gradient variance.

On on-policy data (first gradient step), Eq. 9 and Eq. 10 produce identical gradients since all ratios equal 1. The difference emerges on subsequent gradient steps within each PPO epoch.

### 2.4 FPO++ Modification 2: Asymmetric Trust Region (ASPO)

FPO++ uses an **asymmetric** trust region that applies different clipping based on the sign of the advantage:

**For positive advantages (Â_t ≥ 0):** Use standard PPO clipping:
```
ψ_PPO(ρ_θ, Â_t) = min(ρ_θ · Â_t, clip(ρ_θ, 1 ± ε^clip) · Â_t)       (Eq. 1)
```

**For negative advantages (Â_t < 0):** Use SPO (Simple Policy Optimization):
```
ψ_SPO(ρ_θ, Â_t) = ρ_θ · Â_t − (|Â_t| / (2·ε^clip)) · (ρ_θ − 1)²    (Eq. 11)
```

Combined as the ASPO objective:
```
ψ_ASPO(ρ_θ, Â_t) = {  ψ_PPO(ρ_θ, Â_t),   if Â_t ≥ 0                  (Eq. 12)
                    {  ψ_SPO(ρ_θ, Â_t),   if Â_t < 0
```

**Why this matters:** For negative-advantage actions, PPO clips the ratio and zeroes the gradient, but SPO instead provides a **pull-back gradient** that actively pushes ratios back toward 1. This:
1. Prevents entropy collapse by discouraging aggressive probability decreases
2. Stabilizes the variational gap between the sampled and learned denoising posteriors
3. Preserves exploration capacity throughout training

### 2.5 Complete FPO++ Objective

```
max_θ  E_{π_θ_old} [ Σ_{i=1}^{N_mc} ψ_ASPO( ρ̂_FPO++^{(i)}(θ), Â_t ) ]   (Eq. 13)
```

The same advantage Â_t is shared across all N_mc samples for a given action.

### 2.6 Zero-Sampling

At **test time** (evaluation/deployment), FPO++ initializes flow integration from `ε = 0` (the zero vector) instead of sampling `ε ~ N(0, I)`. This deterministic initialization improves performance by removing sampling noise. During **training**, standard random sampling `ε ~ N(0, I)` is used for exploration.

### 2.7 CFM Loss Clamping (Numerical Stability)

The paper found that exponentiating squared CFM losses can cause numerical instability for outlier ε values. The solution:
1. **Clamp individual CFM losses** before taking differences
2. **Clamp the CFM loss difference** before exponentiation

```python
# Clamp CFM losses
ℓ_θ^{(i,t)} = clamp(ℓ_θ^{(i,t)}, 0, δ)        # δ ∈ {1, 2, 4} or off
# Clamp difference before exp
diff = clamp(ℓ_{θ_old}^{(i,t)} − ℓ_θ^{(i,t)}, -max_diff, max_diff)
ρ̂ = exp(diff)
```

---

## 3. Architectural Differences: FPO++ vs. ReinFlow

| Aspect | ReinFlow | FPO++ |
|--------|----------|-------|
| **Likelihood ratio** | Explicit log-prob via noise injection chain | CFM loss surrogate (Eq. 10) |
| **Noise injection network** | Learnable `NoisyFlowMLP` + `ExploreNoiseNet` | Not needed — no explicit likelihoods |
| **Exploration** | Learned noise σ(s,t) at each denoising step | Random ε ~ N(0,I) as initial noise for flow |
| **Denoising steps (train)** | 1-4 steps with noise injection | K steps of Euler integration (10-64 typical) |
| **Denoising steps (eval)** | Same as train | Can differ; zero-sampling (ε=0) |
| **Trust region** | Standard PPO clipping on log-probs | ASPO: asymmetric PPO+SPO (Eq. 12) |
| **Ratio computation** | exp(log π_new − log π_old) | exp(CFM_old − CFM_new) per sample (Eq. 10) |
| **Stored in buffer** | Denoising chains `[B, K+1, Ta, Da]` | Clean actions `a_t` + MC samples `(τ_i, ε_i)` |
| **Network architecture** | Frozen pretrained + learnable noise net | Single velocity network (fine-tuned end-to-end) |
| **BC regularization** | Optional MSE to pretrained policy outputs | Via CFM loss itself (implicitly regularizes) |
| **Action chunking** | horizon_steps (typically 4) | horizon_steps (paper uses 16 for manipulation) |

---

## 4. Implementation Plan: Files to Create/Modify

### 4.1 New Files

```
model/flow/ft_fpopp/
├── __init__.py
├── fpopp_flow.py          # FPOPPFlow model class (core algorithm)

agent/finetune/fpopp/
├── __init__.py
├── train_fpopp_agent.py        # State-based FPO++ training agent
├── train_fpopp_img_agent.py    # Image-based FPO++ training agent
├── buffer.py                   # FPO++ replay buffer (stores actions + MC samples)

cfg/robomimic/finetune/lift/
├── ft_fpopp_reflow_mlp_img.yaml    # Config for FPO++ fine-tuning (Lift, image)

cfg/robomimic/finetune/can/
├── ft_fpopp_reflow_mlp_img.yaml    # Config for FPO++ fine-tuning (Can, image)

cfg/robomimic/finetune/square/
├── ft_fpopp_reflow_mlp_img.yaml    # Config for FPO++ fine-tuning (Square, image)
```

### 4.2 Files to Reference (Read-Only)

```
model/flow/ft_ppo/ppoflow.py           # ReinFlow's PPOFlow — reference for structure
model/flow/mlp_flow.py                  # FlowMLP, VisionFlowMLP — reuse for velocity net
model/flow/reflow.py                    # ReFlow — reference for CFM loss computation
model/common/critic.py                  # Critic networks — reuse directly
agent/finetune/reinflow/train_ppo_flow_agent.py      # Reference training loop
agent/finetune/reinflow/train_ppo_flow_img_agent.py   # Reference image training loop
agent/finetune/reinflow/buffer.py       # Reference buffer — adapt for FPO++
agent/eval/eval_reflow_agent.py         # Evaluation — reuse with zero-sampling option
```

---

## 5. Detailed Implementation: Module by Module

### 5.1 `model/flow/ft_fpopp/fpopp_flow.py` — Core FPO++ Model

```python
class FPOPPFlow(nn.Module):
    """
    FPO++ model for flow policy gradient training.

    Unlike PPOFlow (ReinFlow), this does NOT use a noise injection network.
    Instead, it uses CFM loss differences as a surrogate for likelihood ratios.
    """

    def __init__(self,
                 actor: nn.Module,          # Pretrained flow policy (velocity network)
                 critic: nn.Module,         # Value function
                 n_mc_samples: int = 16,    # N_mc Monte Carlo samples
                 flow_steps: int = 10,      # K Euler integration steps for rollouts
                 clip_param: float = 0.05,  # ε^clip
                 use_aspo: bool = True,     # Enable asymmetric trust region
                 cfm_loss_clamp: float = 2.0,   # δ for clamping CFM losses
                 cfm_diff_clamp: float = 5.0,   # Max abs value for loss diff before exp
                 vf_coef: float = 0.5,
                 network_output: str = 'u',  # 'u' for velocity, 'x_0' for data prediction
                 ):
        super().__init__()

        # Actor: fine-tuned velocity network (full copy, trainable)
        self.actor = actor

        # Actor old: frozen copy for computing old CFM losses
        self.actor_old = copy.deepcopy(actor)
        for param in self.actor_old.parameters():
            param.requires_grad = False

        # Critic
        self.critic = critic

        # FPO++ hyperparameters
        self.n_mc_samples = n_mc_samples
        self.flow_steps = flow_steps
        self.clip_param = clip_param
        self.use_aspo = use_aspo
        self.cfm_loss_clamp = cfm_loss_clamp
        self.cfm_diff_clamp = cfm_diff_clamp
        self.vf_coef = vf_coef
        self.network_output = network_output
```

**Key Methods:**

#### `get_actions()` — Sample actions via Euler integration

```python
def get_actions(self, cond, eval_mode=False):
    """
    Sample actions by running Euler integration of the flow ODE.

    Training: ε ~ N(0, I)  (random sampling for exploration)
    Eval:     ε = 0         (zero-sampling for deterministic behavior)

    Returns: actions [B, Ta, Da]
    """
    B = cond["state"].shape[0]

    if eval_mode:
        # Zero-sampling: deterministic initialization
        xt = torch.zeros(B, horizon_steps, action_dim, device=device)
    else:
        # Random sampling: stochastic exploration
        xt = torch.randn(B, horizon_steps, action_dim, device=device)

    dt = 1.0 / self.flow_steps

    with torch.no_grad():
        for k in range(self.flow_steps):
            t = torch.full((B,), k * dt, device=device)

            # Predict velocity
            vt = self.actor(xt, t, cond)  # [B, Ta, Da]

            # Euler step
            xt = xt + vt * dt

    # Clip to action bounds
    xt = xt.clamp(act_min, act_max)
    return xt
```

#### `compute_cfm_losses()` — Core CFM loss computation

```python
def compute_cfm_losses(self, actor, cond, actions, taus, epsilons):
    """
    Compute per-sample CFM losses for a batch.

    Args:
        actor: velocity network (either self.actor or self.actor_old)
        cond: observations dict
        actions: clean actions a_t [B, Ta, Da]
        taus: flow timesteps [B, N_mc]
        epsilons: noise samples [B, N_mc, Ta, Da]

    Returns:
        cfm_losses: [B, N_mc] per-sample CFM losses
    """
    B, N_mc = taus.shape

    # Expand actions for MC samples: [B, N_mc, Ta, Da]
    a_expanded = actions.unsqueeze(1).expand(-1, N_mc, -1, -1)

    # Interpolate: a_t^τ = τ · a_t + (1 − τ) · ε
    taus_expanded = taus.unsqueeze(-1).unsqueeze(-1)  # [B, N_mc, 1, 1]
    a_noised = taus_expanded * a_expanded + (1 - taus_expanded) * epsilons  # [B, N_mc, Ta, Da]

    # Velocity target: a_t − ε_i
    v_target = a_expanded - epsilons  # [B, N_mc, Ta, Da]

    # Reshape for batch processing: [B*N_mc, Ta, Da]
    a_noised_flat = a_noised.reshape(B * N_mc, *actions.shape[1:])
    taus_flat = taus.reshape(B * N_mc)

    # Expand cond for N_mc samples
    cond_expanded = expand_cond(cond, N_mc)  # Each obs repeated N_mc times

    # Predict velocity: v̂_θ(a_t^τ, τ; o_t)
    v_pred = actor(a_noised_flat, taus_flat, cond_expanded)  # [B*N_mc, Ta, Da]
    v_pred = v_pred.reshape(B, N_mc, *actions.shape[1:])

    # If network outputs x_0 prediction instead of velocity u:
    if self.network_output == 'x_0':
        # Convert x_0 prediction to velocity: u = (x_0_pred - (1-τ)·ε) / ...
        # Actually, the loss is still MSE between prediction and target
        # For x_0 prediction: target is a_t, prediction is x_0_hat
        v_target = a_expanded  # target is clean action

    # Per-sample CFM loss: ‖v̂_θ − (a_t − ε_i)‖₂²
    cfm_losses = ((v_pred - v_target) ** 2).sum(dim=(-2, -1))  # [B, N_mc]

    # Clamp CFM losses for numerical stability
    if self.cfm_loss_clamp > 0:
        cfm_losses = cfm_losses.clamp(0, self.cfm_loss_clamp)

    return cfm_losses
```

#### `loss()` — Complete FPO++ loss computation

```python
def loss(self, cond, actions, returns, old_values, advantages,
         taus, epsilons, old_cfm_losses):
    """
    Compute FPO++ loss with per-sample ratios and ASPO trust region.

    Args:
        cond: observations
        actions: sampled actions [B, Ta, Da]
        returns: GAE returns [B]
        old_values: old value estimates [B]
        advantages: GAE advantages [B]
        taus: MC timesteps [B, N_mc]
        epsilons: MC noise [B, N_mc, Ta, Da]
        old_cfm_losses: precomputed CFM losses under θ_old [B, N_mc]

    Returns:
        total_loss, pg_loss, v_loss, info_dict
    """

    # === 1. Compute new CFM losses ===
    new_cfm_losses = self.compute_cfm_losses(
        self.actor, cond, actions, taus, epsilons
    )  # [B, N_mc]

    # === 2. Per-sample ratios (Eq. 10) ===
    # ρ̂^{(i)} = exp(ℓ_{θ_old}^{(i)} − ℓ_θ^{(i)})
    cfm_diff = old_cfm_losses - new_cfm_losses  # [B, N_mc]

    # Clamp difference before exponentiation
    if self.cfm_diff_clamp > 0:
        cfm_diff = cfm_diff.clamp(-self.cfm_diff_clamp, self.cfm_diff_clamp)

    ratios = torch.exp(cfm_diff)  # [B, N_mc]

    # === 3. Advantage: shared across MC samples ===
    # Normalize advantages
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    adv_expanded = adv.unsqueeze(1).expand(-1, self.n_mc_samples)  # [B, N_mc]

    # === 4. ASPO trust region (Eq. 12-13) ===
    if self.use_aspo:
        # Positive advantages: PPO clipping
        pg_loss_ppo1 = -adv_expanded * ratios
        pg_loss_ppo2 = -adv_expanded * torch.clamp(
            ratios, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        ppo_loss = torch.max(pg_loss_ppo1, pg_loss_ppo2)

        # Negative advantages: SPO objective
        # ψ_SPO = ρ · Â − (|Â| / (2ε)) · (ρ − 1)²
        spo_loss = -(adv_expanded * ratios -
                     (adv_expanded.abs() / (2 * self.clip_param)) *
                     (ratios - 1.0) ** 2)

        # Select based on advantage sign
        pos_mask = (adv_expanded >= 0).float()
        pg_loss_per_sample = pos_mask * ppo_loss + (1 - pos_mask) * spo_loss
    else:
        # Standard PPO clipping for all advantages
        pg_loss1 = -adv_expanded * ratios
        pg_loss2 = -adv_expanded * torch.clamp(
            ratios, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        pg_loss_per_sample = torch.max(pg_loss1, pg_loss2)

    # Average over MC samples and batch
    pg_loss = pg_loss_per_sample.mean()

    # === 5. Value function loss ===
    new_values = self.critic(cond).squeeze(-1)  # [B]
    v_loss = 0.5 * ((new_values - returns) ** 2).mean()

    # === 6. Total loss ===
    total_loss = pg_loss + self.vf_coef * v_loss

    # === 7. Logging info ===
    with torch.no_grad():
        clipfrac = ((ratios - 1.0).abs() > self.clip_param).float().mean()
        approx_kl = (cfm_diff).mean()  # Approximate KL via CFM diff

    info = {
        "pg_loss": pg_loss.item(),
        "v_loss": v_loss.item(),
        "total_loss": total_loss.item(),
        "ratio_mean": ratios.mean().item(),
        "ratio_std": ratios.std().item(),
        "ratio_max": ratios.max().item(),
        "ratio_min": ratios.min().item(),
        "clipfrac": clipfrac.item(),
        "approx_kl": approx_kl.item(),
        "cfm_loss_new_mean": new_cfm_losses.mean().item(),
        "cfm_loss_old_mean": old_cfm_losses.mean().item(),
    }

    return total_loss, info
```

#### `sync_old_policy()` — Update frozen policy

```python
def sync_old_policy(self):
    """Copy current actor weights to actor_old (called at start of each PPO iteration)."""
    self.actor_old.load_state_dict(self.actor.state_dict())
```

### 5.2 `agent/finetune/fpopp/buffer.py` — FPO++ Buffer

The FPO++ buffer differs from ReinFlow's buffer because it stores **clean actions** and **MC samples** instead of denoising chains and log-probabilities.

```python
class FPOPPBuffer:
    """
    Replay buffer for FPO++ training.

    Stores per-step:
      - observations (state / images)
      - actions (clean, from rollouts)  [n_steps, n_envs, Ta, Da]
      - rewards                         [n_steps, n_envs]
      - dones                           [n_steps, n_envs]
      - values                          [n_steps, n_envs]
      - MC samples (τ, ε)               [n_steps, n_envs, N_mc] and [n_steps, n_envs, N_mc, Ta, Da]
      - old CFM losses                  [n_steps, n_envs, N_mc]

    Does NOT store:
      - log-probabilities (not computed)
      - denoising chains (not needed)
    """
```

Key differences from `PPOFlowBuffer`:
- No `chains_trajs` (no denoising chains)
- No `logprobs_trajs` (no log-probabilities)
- Added `taus_trajs` [n_steps, n_envs, N_mc] — MC timesteps
- Added `epsilons_trajs` [n_steps, n_envs, N_mc, Ta, Da] — MC noise
- Added `old_cfm_losses_trajs` [n_steps, n_envs, N_mc] — old CFM losses

### 5.3 `agent/finetune/fpopp/train_fpopp_agent.py` — Training Agent

```python
class TrainFPOPPAgent(TrainAgent):
    """
    FPO++ fine-tuning agent for state-based observations.
    Mirrors TrainPPOFlowAgent but uses CFM-loss-based ratios instead of log-probs.
    """
```

**Training loop (pseudocode):**

```python
def run(self):
    for itr in range(n_train_itr):

        # === PHASE 1: ROLLOUT (sample environment trajectories) ===
        self.model.eval()
        for step in range(n_steps):
            with torch.no_grad():
                # Get value estimate
                value = self.model.critic(cond)

                # Sample action via Euler integration (random ε for exploration)
                action = self.model.get_actions(cond, eval_mode=False)

                # Generate MC samples for this (action, obs) pair
                taus = torch.rand(n_envs, n_mc_samples)           # [n_envs, N_mc]
                epsilons = torch.randn(n_envs, n_mc_samples, Ta, Da)  # [n_envs, N_mc, Ta, Da]

                # Compute CFM losses under current (old) policy
                old_cfm_losses = self.model.compute_cfm_losses(
                    self.model.actor, cond, action, taus, epsilons
                )  # [n_envs, N_mc]

                # Step environment
                obs, reward, done, truncated, info = venv.step(action)

                # Store in buffer
                buffer.add(step, obs, action, reward, done, truncated,
                          value, taus, epsilons, old_cfm_losses)

        # === PHASE 2: COMPUTE ADVANTAGES ===
        with torch.no_grad():
            final_value = self.model.critic(final_obs)
        buffer.update(final_value)  # Compute GAE advantages & returns

        # === PHASE 3: SYNC OLD POLICY ===
        self.model.sync_old_policy()  # Copy actor → actor_old

        # === PHASE 4: PPO UPDATE EPOCHS ===
        self.model.train()
        dataset = buffer.make_dataset()

        for epoch in range(update_epochs):
            for minibatch in dataset.shuffle_and_batch(batch_size):
                cond, actions, returns, old_values, advantages, \
                    taus, epsilons, old_cfm_losses = minibatch

                # Compute FPO++ loss
                total_loss, info = self.model.loss(
                    cond, actions, returns, old_values, advantages,
                    taus, epsilons, old_cfm_losses
                )

                # Optimize
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                total_loss.backward()

                # Optional gradient clipping
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.actor.parameters(), max_grad_norm
                    )

                if itr >= n_critic_warmup_itr:
                    actor_optimizer.step()
                critic_optimizer.step()

            # Early stopping on KL
            if info["approx_kl"] > target_kl:
                break

        # === PHASE 5: LOGGING & SCHEDULING ===
        log_metrics(info)
        update_lr_schedulers()
        save_checkpoint_if_needed()
```

### 5.4 `agent/finetune/fpopp/train_fpopp_img_agent.py` — Image-Based Agent

Extends `TrainFPOPPAgent` with:
- Image augmentation (`RandomShiftsAug`)
- Gradient accumulation for memory management
- GPU-resident buffer for image observations
- ViT encoder for image conditioning

### 5.5 Evaluation: Reuse Existing `EvalReFlowAgent`

The existing evaluation agents can be reused with minor modifications:
- Add a `zero_sampling` flag (initialize ε=0 instead of ε~N(0,I))
- Load FPO++ checkpoint format (extract actor weights)
- The velocity network architecture is identical — only the fine-tuning wrapper differs

```python
# In eval agent, loading FPO++ checkpoint:
actor_state_dict = {
    key.replace('actor.', 'network.'): value
    for key, value in checkpoint["model"].items()
    if key.startswith('actor.')
}
model.load_state_dict(actor_state_dict, strict=False)
```

### 5.6 Config: `ft_fpopp_reflow_mlp_img.yaml`

```yaml
defaults:
  - _self_

_target_: agent.finetune.fpopp.train_fpopp_img_agent.TrainFPOPPImgAgent

name: ft_fpopp_reflow_mlp_img_lift
env_suite: robomimic
env: lift

# Action space
action_dim: 7
horizon_steps: 16        # Action chunk length (paper uses 16 for manipulation)
act_steps: 8             # Steps to execute per chunk
cond_steps: 1
obs_dim: 9

# FPO++ specific
flow_steps: 10           # K Euler integration steps (train & eval)
n_mc_samples: 8          # N_mc Monte Carlo samples per action
use_aspo: false           # Disable ASPO for manipulation fine-tuning (paper D.5)
cfm_loss_clamp: 2.0      # δ for CFM loss clamping
cfm_diff_clamp: 5.0      # Clamp diff before exp
network_output: u         # Velocity prediction
zero_sampling_eval: true  # Use ε=0 at eval time

# Model
model:
  _target_: model.flow.ft_fpopp.fpopp_flow.FPOPPFlow
  clip_param: 0.05
  vf_coef: 0.5

  actor:
    _target_: model.flow.mlp_flow.VisionFlowMLP
    mlp_dims: [512, 512, 512]
    time_dim: 32
    spatial_emb: 128

  critic:
    _target_: model.common.critic.ViTCritic
    spatial_emb: 128
    mlp_dims: [256, 256]

# Training
train:
  n_train_itr: 200
  n_steps: 300
  n_envs: 50
  batch_size: 500
  update_epochs: 5

  actor_lr: 1.0e-5
  critic_lr: 1.0e-4
  weight_decay: 1.0e-4

  gamma: 0.99            # Task-specific (Can: 0.99, Square: 0.995)
  gae_lambda: 0.99

  n_critic_warmup_itr: 2
  max_grad_norm: 1.0
  target_kl: null

  # Augmentation (image-based)
  augment: true
  grad_accumulate: 4

# Evaluation
eval:
  eval_interval: 10
  n_eval_episodes: 50

# Pretrained policy path
base_policy_path: ???     # Path to pretrained ReFlow checkpoint
```

---

## 6. Chunk-Level Ratio Computation (Manipulation Detail)

For manipulation fine-tuning, the paper notes: "We compute chunk-level ratios by summing CFM losses across all chunk timesteps." This means for action chunks of length `T_a`:

```python
# For action chunk a_t = [a_t^1, a_t^2, ..., a_t^{T_a}]
# CFM loss is summed across chunk positions:

# Already handled by .sum(dim=(-2, -1)) in compute_cfm_losses()
# where dim -2 is the chunk dimension (Ta) and dim -1 is the action dimension (Da)
cfm_losses = ((v_pred - v_target) ** 2).sum(dim=(-2, -1))  # [B, N_mc]
```

This is consistent with treating the entire action chunk as a single "action" for the CFM loss computation.

---

## 7. Implementation Checkpoints & Verification Strategy

### Checkpoint 1: CFM Loss Computation Correctness
**Files:** `model/flow/ft_fpopp/fpopp_flow.py`
**Test:**
- [ ] Verify `compute_cfm_losses()` produces identical output to `ReFlow.loss()` (from `model/flow/reflow.py`) for the same `(action, tau, epsilon, cond)` inputs
- [ ] Verify that with `actor = actor_old` (identical weights), the CFM loss difference is exactly 0 and all ratios equal 1.0
- [ ] Test with known velocity field (e.g., identity: v(x,t) = x) that CFM losses match analytical values
- [ ] Verify numerical stability: run with extreme ε values (||ε|| > 5σ) and confirm clamping prevents NaN/Inf

**How to verify:**
```python
# Unit test: compare against ReFlow.loss()
reflow_model = ReFlow(...)
reflow_model.load_state_dict(pretrained_weights)

fpopp_model = FPOPPFlow(actor=reflow_model.network, ...)

# Same inputs
action = torch.randn(32, 4, 7)
tau = torch.rand(32, 16)
epsilon = torch.randn(32, 16, 4, 7)
cond = {"state": torch.randn(32, 1, 9)}

# ReFlow's CFM loss (for reference)
a_noised = tau.unsqueeze(-1).unsqueeze(-1) * action.unsqueeze(1) + \
           (1 - tau.unsqueeze(-1).unsqueeze(-1)) * epsilon
v_target = action.unsqueeze(1) - epsilon
# ... compute v_pred and MSE ...

# FPO++ CFM loss
fpopp_losses = fpopp_model.compute_cfm_losses(fpopp_model.actor, cond, action, tau, epsilon)

# Should match
assert torch.allclose(reflow_loss, fpopp_losses, atol=1e-5)
```

### Checkpoint 2: Ratio Computation & Gradient Flow
**Files:** `model/flow/ft_fpopp/fpopp_flow.py`
**Test:**
- [ ] Verify per-sample ratios (Eq. 10) have correct shape `[B, N_mc]`
- [ ] Verify gradients flow through `new_cfm_losses` but NOT through `old_cfm_losses`
- [ ] Verify that on the first gradient step (on-policy), all ratios ≈ 1.0 (since θ = θ_old)
- [ ] Verify `sync_old_policy()` properly detaches the old policy from the computation graph
- [ ] After one gradient step, verify ratios deviate from 1.0 in the expected direction (decrease for positive advantages, increase for negative)

**How to verify:**
```python
# Gradient flow test
old_cfm = fpopp_model.compute_cfm_losses(fpopp_model.actor_old, ...)
new_cfm = fpopp_model.compute_cfm_losses(fpopp_model.actor, ...)

loss = (old_cfm - new_cfm).mean()
loss.backward()

# actor should have gradients
assert all(p.grad is not None for p in fpopp_model.actor.parameters() if p.requires_grad)

# actor_old should NOT have gradients
assert all(p.grad is None for p in fpopp_model.actor_old.parameters())
```

### Checkpoint 3: ASPO Trust Region Correctness
**Files:** `model/flow/ft_fpopp/fpopp_flow.py`
**Test:**
- [ ] With all positive advantages: loss should equal standard PPO clipped loss
- [ ] With all negative advantages: loss should equal SPO loss
- [ ] With mixed advantages: verify piecewise selection matches Eq. 12
- [ ] Verify SPO gradient: for negative advantage with ratio > 1, gradient should pull ratio back toward 1 (not zero it)
- [ ] Verify that with `use_aspo=False`, behavior matches standard PPO clipping

**How to verify:**
```python
# Test ASPO vs PPO for known ratios and advantages
ratios = torch.tensor([0.5, 1.0, 1.5, 2.0])
advantages = torch.tensor([1.0, 1.0, -1.0, -1.0])
clip_param = 0.05

# Manual PPO computation
ppo_loss_manual = ...

# Manual SPO computation for negative advantages
spo_loss_manual = ...

# ASPO should combine them
aspo_loss = fpopp_model._compute_aspo_loss(ratios, advantages, clip_param)
# Verify: positive adv entries match PPO, negative adv entries match SPO
```

### Checkpoint 4: Buffer Correctness
**Files:** `agent/finetune/fpopp/buffer.py`
**Test:**
- [ ] Verify buffer stores and retrieves `(obs, actions, taus, epsilons, old_cfm_losses)` correctly
- [ ] Verify GAE computation matches ReinFlow's buffer (same algorithm, different stored data)
- [ ] Verify `make_dataset()` properly flattens `[n_steps, n_envs, ...]` → `[n_steps * n_envs, ...]`
- [ ] Verify minibatch sampling produces correct shapes
- [ ] Memory footprint check: `epsilons` is the largest tensor `[n_steps, n_envs, N_mc, Ta, Da]` — verify it fits in memory

**How to verify:**
```python
buffer = FPOPPBuffer(n_steps=300, n_envs=50, n_mc_samples=8,
                     horizon_steps=16, action_dim=7)
# Fill with dummy data
for step in range(300):
    buffer.add(step, obs, action, reward, done, truncated,
              value, taus, epsilons, old_cfm_losses)

# Verify shapes
dataset = buffer.make_dataset()
batch = next(iter(dataset.batch(500)))
assert batch.actions.shape == (500, 16, 7)
assert batch.taus.shape == (500, 8)
assert batch.epsilons.shape == (500, 8, 16, 7)
assert batch.old_cfm_losses.shape == (500, 8)
```

### Checkpoint 5: End-to-End Rollout & Single Update
**Files:** `agent/finetune/fpopp/train_fpopp_agent.py`
**Test:**
- [ ] Load pretrained ReFlow checkpoint into FPO++ model
- [ ] Run 1 rollout iteration (n_steps environment steps)
- [ ] Verify actions are reasonable (within action bounds, not NaN)
- [ ] Verify rewards are collected correctly
- [ ] Run 1 PPO update epoch
- [ ] Verify loss decreases (or at least doesn't explode)
- [ ] Verify all logged metrics are finite
- [ ] Compare initial success rate (before any updates) to pretrained policy — should be similar

**How to verify:**
```bash
# Run with n_train_itr=1, n_steps=10, update_epochs=1 for a quick smoke test
python script/run.py \
  --config-name=robomimic/finetune/lift/ft_fpopp_reflow_mlp_img \
  train.n_train_itr=1 train.n_steps=10 train.update_epochs=1 \
  base_policy_path=/path/to/pretrained.pt device=cuda:0
```

### Checkpoint 6: Training Stability (Short Run)
**Files:** All FPO++ files
**Test:**
- [ ] Run 10 training iterations on Lift task
- [ ] Verify no NaN in losses, ratios, or gradients
- [ ] Verify ratios stay within reasonable range (e.g., 0.1 to 10.0)
- [ ] Verify value loss decreases during critic warmup
- [ ] Verify policy loss doesn't explode
- [ ] Check that CFM loss differences remain bounded
- [ ] Monitor clipfrac — should be > 0 but < 0.5

### Checkpoint 7: Evaluation Integration
**Files:** Evaluation config + agent
**Test:**
- [ ] Load FPO++ checkpoint in existing eval agent
- [ ] Verify zero-sampling evaluation works (ε=0)
- [ ] Verify random-sampling evaluation works (ε~N(0,I))
- [ ] Compare zero-sampling vs random-sampling success rates (zero should be higher, per paper)
- [ ] Verify evaluation produces same results whether called from training loop or standalone eval script

### Checkpoint 8: Full Training Run & Comparison
**Test:**
- [ ] Run full FPO++ training on Lift task (200 iterations)
- [ ] Compare learning curve to ReinFlow on same task
- [ ] Verify success rate improves over training
- [ ] Run full training on Can task
- [ ] Optionally run on Square task (harder, longer horizon)
- [ ] Verify that ASPO disabled (for manipulation) matches paper recommendation

### Checkpoint 9: Ablations
**Test:**
- [ ] Per-sample ratio vs per-action ratio: run both, compare stability
- [ ] ASPO vs PPO clipping: run both for manipulation tasks
- [ ] Vary N_mc samples: {4, 8, 16, 32} — verify more samples = smoother training
- [ ] Vary flow steps: {5, 10, 20} — verify minimal impact on final performance
- [ ] Vary clip parameter: {0.01, 0.05, 0.1}

---

## 8. Key Design Decisions & Rationale

### 8.1 Why Not Reuse PPOFlow?

ReinFlow's `PPOFlow` is tightly coupled to the noise injection paradigm:
- `NoisyFlowMLP` wraps the policy with a learnable noise network
- `get_logprobs()` computes explicit likelihoods through the denoising chain
- The buffer stores denoising chains `[B, K+1, Ta, Da]`

FPO++ is fundamentally different — it bypasses likelihoods entirely. Creating a new model class avoids polluting the existing code with conditional logic and makes both approaches independently maintainable.

### 8.2 Why Separate Buffer?

The FPO++ buffer stores different data:
- **No chains:** FPO++ doesn't need denoising trajectories
- **No log-probs:** FPO++ doesn't compute likelihoods
- **MC samples:** FPO++ needs `(τ_i, ε_i)` pairs and their CFM losses

A separate buffer class is cleaner than adding conditional fields to the existing buffer.

### 8.3 Why Not Modify the Training Agent In-Place?

The training loop structure is similar but the inner update logic differs substantially:
- ReinFlow: `get_logprobs()` → PPO loss on log-ratios
- FPO++: `compute_cfm_losses()` → per-sample ratios → ASPO loss

A separate agent avoids complex branching in the training loop while reusing shared utilities (GAE computation, environment interaction, logging).

### 8.4 ASPO for Fine-tuning

The paper (Section D.5, Appendix) notes that ASPO can degrade manipulation fine-tuning performance. For fine-tuning, `use_aspo=False` should be the default. ASPO is primarily beneficial for from-scratch training (locomotion) where entropy preservation is critical.

---

## 9. Memory & Compute Considerations

### 9.1 Memory Footprint

The main memory concern is storing MC samples:
```
epsilons: [n_steps, n_envs, N_mc, Ta, Da] = [300, 50, 8, 16, 7]
         = 300 * 50 * 8 * 16 * 7 * 4 bytes = ~54 MB (float32)

taus:     [n_steps, n_envs, N_mc] = [300, 50, 8]
         = ~0.5 MB

old_cfm:  [n_steps, n_envs, N_mc] = [300, 50, 8]
         = ~0.5 MB
```

This is significantly less than ReinFlow's chains storage which is `[300, 50, K+1, Ta, Da]`.

### 9.2 Compute Cost

The main compute cost is computing CFM losses, which requires `B * N_mc` forward passes through the velocity network per minibatch. For `batch_size=500, N_mc=8`, this means 4000 forward passes per minibatch.

Optimization: batch all `B * N_mc` samples into a single forward pass by reshaping.

### 9.3 Wall-Clock Time

The paper notes FPO++ is slower than Gaussian PPO (~23 min vs ~19 min for G1 locomotion). The bottleneck is the N_mc forward passes per update. For manipulation fine-tuning with N_mc=8 and 10 flow steps, expect ~1.5-2x the wall-clock time of ReinFlow (which uses 1-step inference).

---

## 10. Summary: File Creation Order

1. **`model/flow/ft_fpopp/__init__.py`** — Empty init
2. **`model/flow/ft_fpopp/fpopp_flow.py`** — Core FPO++ model (Checkpoint 1-3)
3. **`agent/finetune/fpopp/__init__.py`** — Empty init
4. **`agent/finetune/fpopp/buffer.py`** — FPO++ buffer (Checkpoint 4)
5. **`agent/finetune/fpopp/train_fpopp_agent.py`** — State-based training agent (Checkpoint 5-6)
6. **`agent/finetune/fpopp/train_fpopp_img_agent.py`** — Image-based training agent (Checkpoint 5-6)
7. **`cfg/robomimic/finetune/lift/ft_fpopp_reflow_mlp_img.yaml`** — Lift config
8. **`cfg/robomimic/finetune/can/ft_fpopp_reflow_mlp_img.yaml`** — Can config
9. **Evaluation integration** — Modify eval agent for zero-sampling + FPO++ checkpoint loading (Checkpoint 7)
10. **Full training runs** (Checkpoint 8-9)
