# FlowGRPO: Design Document

**Branch:** `flowGRPO`
**Task:** CAN (can-picking), image observations, same pretrained ReFlow backbone as FPO++

---

## 1. Theoretical Overview

### 1.1 GRPO (Group Relative Policy Optimization)

GRPO (Shao et al., 2024; DeepSeek-R1) replaces the PPO critic with group statistics. For each
input, sample G outputs, get their rewards, and normalize within the group to form advantages:

```
A_i = clip( (r_i - mean_G(r)) / (std_G(r) + eps),  -adv_clip_max, +adv_clip_max )
```

Update with the standard clipped PPO surrogate (no value loss term):

```
L = E_i [ -min( rho_i * A_i,  clip(rho_i, 1-eps_clip, 1+eps_clip) * A_i ) ]
```

No value network is required — the group mean is the baseline.

### 1.2 Official FlowGRPO (image generation)

The official repo (github.com/yifan123/flow_grpo) applies GRPO to SD3/FLUX diffusion models:

- **Group**: a batch of generated images (same prompt, different noise seeds)
- **Reward**: external reward model (PickScore, GenEval, OCR, etc.)
- **Ratio**: `exp(log_prob_new - log_prob_old)` using *SDE-based* per-step log probabilities
  - At each denoising step the SDE adds stochastic noise → tractable Gaussian log prob:
    `log p(x_{t-1}|x_t) = -||x_{t-1} - mu(x_t,t)||^2 / (2*sigma_t^2) + const`
- **Clip range**: extremely tight, `1e-4` (vs PPO's typical `0.2`)
- **Advantage clip**: `adv_clip_max = 5`
- **KL regularization**: optional, default disabled (`beta = 0`)
- **No critic**

Key insight from the official code: zero-advantage samples (entire batch has identical reward,
so `std(r) ≈ 0`) are **filtered out entirely** before the gradient step — they carry no signal.

### 1.3 Adapting FlowGRPO to Robot Control

Two things need adapting for our setting (robot manipulation, ODE-based ReFlow backbone):

**A. The "group" in a parallel-env rollout**

In LLMs/image gen: one prompt → G outputs → G rewards (episodic).
In robot RL with n_envs parallel environments: each env runs for n_steps steps, collecting a
trajectory. The group is naturally the n_envs environments sharing the same task distribution.
Each environment i gets a discounted return:

```
R_i = sum_{t=0}^{T-1} gamma^t * r_{i,t}
```

The group-relative advantage is the same for every timestep in environment i's trajectory:

```
A_{i,t} = clip( (R_i - mean_E(R)) / (std_E(R) + eps),  -adv_clip, +adv_clip )
```

If `std_E(R) < eps` (all envs got the same reward — common early in training), skip the update
for that iteration (zero-advantage filter from the official repo).

**B. The probability ratio for ODE-based flow policies**

Our VisionFlowMLP uses a deterministic ODE (Euler integration), not an SDE. Exact log
probabilities require either: (a) an SDE formulation, or (b) a surrogate.

We use the **CFM loss surrogate** (same as FPO++), which is theoretically justified: if the flow
velocity network defines a Gaussian distribution over velocities, then `log p ∝ -l_CFM`. The
per-MC-sample ratio is:

```
rho^(k) = exp( l^(k)_{old} - l^(k)_{new} )

l^(k) = mean_{Ta, Da} || v_theta( tau_k*a + (1-tau_k)*eps_k,  tau_k;  s ) - (a - eps_k) ||^2
```

We keep FPO++'s per-sample ratio (one ratio per (tau, eps) pair) rather than averaging first,
for finer-grained trust region control.

**Clip range**: following the official repo, we use a tight clip of `1e-4` rather than FPO++'s
`0.01`. This reflects that the CFM ratio can be large and the trust region should be tight.

### 1.4 Full FlowGRPO-Robotics Objective

```
L(theta) = E_{(s,a,k)} [
    -min(
        rho^(k) * A,
        clip(rho^(k), 1-1e-4, 1+1e-4) * A
    )
]
```

where the expectation is over rollout timesteps t and environments i, MC samples k, and
A = A_{i,t} is the group-relative advantage (constant within env i's trajectory).

Skip iterations where `std_E(R) < 1e-6` (all envs identical reward — no group signal).

### 1.5 Key Differences from Related Methods

| Property | PPO/ReinFlow | FPO++ | **FlowGRPO (ours)** |
|---|---|---|---|
| Advantage | GAE (critic) | GAE (critic) | Group-relative (no critic) |
| Ratio | Log-prob | CFM loss per sample | CFM loss per sample |
| Clip range | 0.2 | 0.01 | **1e-4** |
| Advantage clip | None | None | **±5** |
| Zero-signal filter | No | No | **Yes** |
| Critic network | Yes | Yes | **No** |
| Value loss | Yes | Yes | **No** |

---

## 2. Implementation Plan

### 2.1 Files to Create

```
agent/finetune/flowgrpo/
    __init__.py
    buffer.py                      # FlowGRPOImgBuffer
    train_flowgrpo_img_agent.py    # TrainFlowGRPOImgAgent

model/flow/ft_flowgrpo/
    __init__.py
    flowgrpo_flow.py               # FlowGRPOFlow (actor only, no critic)

cfg/robomimic/finetune/can/
    ft_flowgrpo_reflow_mlp_img.yaml

script/
    run_flowgrpo.sbatch
```

### 2.2 Step-by-Step Implementation

#### Step 1 — `FlowGRPOImgBuffer` (`agent/finetune/flowgrpo/buffer.py`)

Inherits from `PPOBuffer` (reuses `summarize_episode_reward`, `update_full_obs`, etc.).

**Stores (per step, per env) — note no `value_trajs`:**
- `obs_trajs`: `{rgb: (T,E,To,C,H,W), state: (T,E,To,Do)}`
- `samples_trajs`: `(T,E,Ta,Da)` — clean actions
- `reward_trajs`: `(T,E)`
- `terminated_trajs`: `(T,E)`
- `firsts_trajs`: `(T+1,E)` — episode boundaries
- `taus_trajs`: `(T,E,N_mc)`
- `epsilons_trajs`: `(T,E,N_mc,Ta,Da)`
- `old_cfm_losses_trajs`: `(T,E,N_mc)` — filled in `update_img`
- `advantages_trajs`: `(T,E)` — group-relative, constant per env

**Key method `update_img(obs_venv, model)` — called once per iteration:**
```
1. normalize_reward()              (inherited, running reward scaler)
2. Augment all RGB at once         (same as FPO++)
3. Loop over steps → compute old_cfm_losses_trajs via compute_cfm_losses
   (no critic forward — this loop is now 1 forward pass per step instead of 2)
4. Compute per-env discounted returns R_i = sum_t gamma^t * r_{i,t}
5. Group-normalize:
     mean_R, std_R = R.mean(), R.std()
     if std_R < 1e-6: set self.degenerate = True; return   (zero-signal filter)
     advantages_env = clip((R - mean_R) / (std_R + 1e-8), -adv_clip, +adv_clip)
6. Broadcast: advantages_trajs[t, i] = advantages_env[i]  for all t
```

**`make_dataset()`:** returns `(obs, actions, advantages, taus, epsilons, old_cfm_losses)`.
No `returns`/`oldvalues` tensors (no value loss).

#### Step 2 — `FlowGRPOFlow` (`model/flow/ft_flowgrpo/flowgrpo_flow.py`)

Standalone `nn.Module`. Mirrors `FPOPPFlow` minus the critic.

**Components:**
- `self.actor_ft`: `VisionFlowMLP` (same ViT + MLP as FPO++)
- `self.actor_old`: frozen deepcopy for computing old CFM losses
- **No `self.critic`**

**Methods carried over from FPO++ unchanged:**
- `_load_pretrained_policy()`, `sync_old_policy()`, `get_actions()`, `compute_cfm_losses()`

**`loss()` — simplified (no value loss):**
```python
def loss(self, obs, actions, advantages, taus, epsilons, old_cfm_losses):
    new_cfm_losses = self.compute_cfm_losses(self.actor_ft, obs, actions, taus, epsilons)
    cfm_diff = (old_cfm_losses - new_cfm_losses).clamp(-cfm_diff_clamp, cfm_diff_clamp)
    ratios = torch.exp(cfm_diff)                          # (B, N_mc)
    adv = advantages.unsqueeze(1).expand_as(ratios)       # (B, N_mc)
    pg1 = -adv * ratios
    pg2 = -adv * ratios.clamp(1 - clip_ploss_coef, 1 + clip_ploss_coef)
    pg_loss = torch.max(pg1, pg2).mean()
    # no v_loss
    return pg_loss, clipfrac, approx_kl, ratio_mean, ...
```

#### Step 3 — `TrainFlowGRPOImgAgent` (`agent/finetune/flowgrpo/train_flowgrpo_img_agent.py`)

Inherits from `TrainPPOAgent` (reuses `prepare_run`, `set_model_mode`, `log`, etc.).

**`__init__` differences from FPO++ img agent:**
- **Only `actor_optimizer`** — no `critic_optimizer`, no `critic_lr_scheduler`
- No `n_critic_warmup_itr` (actor updates from itr 0)
- `adv_clip = cfg.train.adv_clip`  (new hyperparameter, default 5.0)

**`run()` loop:**
```
Phase 1 — Rollout: identical to FPO++ img agent
  (sample taus, epsilons; store in buffer; no CFM loss yet)

Phase 2 — Update:
  buffer.update_img(obs_venv, model)   # CFM losses + group-relative advantages
  if buffer.degenerate:                # zero-signal filter
      log.warning("All envs got same reward — skipping update")
      continue to Phase 3
  model.sync_old_policy()
  agent_update()

Phase 3 — Logging: log(), update_lr(), save_model()
```

**`agent_update()`:**
- Same grad_accumulate pattern as FPO++ img agent
- Steps only `actor_optimizer`
- minibatch: `(obs, actions, advantages, taus, epsilons, old_cfm_losses)` — 6 tensors

#### Step 4 — Config (`cfg/robomimic/finetune/can/ft_flowgrpo_reflow_mlp_img.yaml`)

Mirrors `ft_ppo_reflow_mlp_img.yaml` structure with:
- `_target_` pointing to new FlowGRPO classes
- No critic block
- `clip_ploss_coef: 1e-4`  (tighter than FPO++'s 0.01, per official repo)
- `adv_clip: 5.0`          (advantage clipping, per official repo)
- Same flow/actor hyperparameters: `flow_steps: 10`, `n_mc_samples: 4`
- Remove `vf_coef`, `n_critic_warmup_itr`, `clip_vloss_coef`

#### Step 5 — sbatch (`script/run_flowgrpo.sbatch`)

Same as corrected FPO++ sbatch (`--mem=80G`, new config name).

---

## 3. Design Decisions

**Why CFM loss ratio instead of SDE log probs?**
The official FlowGRPO uses an SDE at inference time to get tractable Gaussian log probabilities
at each denoising step. Our backbone (VisionFlowMLP) uses a deterministic ODE. Switching to SDE
inference would require refactoring the entire `get_actions()` pipeline. The CFM loss surrogate
is theoretically equivalent up to a constant (same Gaussian assumption, same velocity network)
and is already proven in FPO++.

**Why per-sample CFM ratios (not one ratio per step)?**
FPO++ showed per-sample ratios dominate averaged ratios for manipulation tasks.

**Why clip_ploss_coef = 1e-4?**
The official repo uses 1e-4 to 4e-6 with tight clipping. Our CFM ratios can be larger than
SDE-based log prob ratios (no 1/2 factor correction), so 1e-4 provides an appropriately tight
trust region.

**Why constant advantage within a trajectory?**
Matches the GRPO spirit: the whole trajectory is the "output", judged as a unit. Simpler and
consistent with the episodic reward structure of CAN.

**Zero-signal filter:**
When all envs succeed or all fail (common at start and end of training), `std(R) ≈ 0` and
advantages carry no information. Skip the actor update entirely that iteration — prevents
noisy gradient updates from near-zero advantages.

**Memory savings vs FPO++:**
No critic ViT encoder → saves ~40% of model GPU memory. The `_compute_values_and_cfm` loop is
also now 1 forward pass per step (not 2), roughly halving Phase 2 compute time.
