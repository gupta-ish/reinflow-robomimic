# Running ReinFlow on Babel (CMU HPC) — Robomimic Fine-tuning with ReinFlow Policy

This guide walks through every step to fine-tune a pre-trained ReinFlow checkpoint on the Robomimic dataset using the ReinFlow (flow-matching + PPO) policy on the Babel HPC cluster in headless mode.

---

## CRITICAL RULE: Always Activate Conda After Every New `srun` Session

Every time you get a new SLURM interactive session (via `srun`) or start a new batch job, **you must run `conda activate reinflow`** before doing anything else. If you forget, `pip install` will target the **system Python** (3.9) instead of your conda environment (3.8), causing version mismatches and build failures.

```bash
# ALWAYS do this first after every new srun session:
conda activate reinflow
```

---

## Step 0: SSH into Babel and Get a GPU Node

```bash
# From your local machine
ssh ishitag@login.babel.cs.cmu.edu
# Enter your password when prompted

# IMPORTANT: Use enough time! Debug partition can time out during pip installs.
# For installation steps, request at least 1 hour:
srun -p general --gres=gpu:1 --time=02:00:00 --mem=32G --pty /bin/bash

# Ex: for debug 20mins:
srun -p debug --gres=gpu:1 --time=00:20:00 --mem=32G --pty /bin/bas

# After getting the node, ALWAYS activate conda first:
conda activate reinflow
```

> **Stale SLURM job error:** If you see `srun: error: Slurm job XXXXX has expired`, run `unset SLURM_JOB_ID` or simply `exit` from the login node and SSH in fresh, then re-run `srun`.

> **Tip:** The `debug` partition has very short time limits (~30 min). Use `-p general` with `--time` for installation and training.

---

## Step 1: Create the Conda Environment (One-Time Setup)

```bash
# On the login node or a compute node:
conda create -n reinflow python=3.8 -y
conda activate reinflow
```

---

## Step 2: Clone and Enter the ReinFlow Repository

```bash
cd ~/
git clone https://github.com/ReinFlow/ReinFlow.git
cd ~/ReinFlow
```

---

## Step 3: Install MuJoCo Dependencies (No Root Privileges)

Since you don't have `sudo` on Babel, install everything via conda/pip.

### 3a. Install MuJoCo 210

```bash
mkdir -p $HOME/.mujoco
cd $HOME/.mujoco

# Download mujoco210 for Linux
wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz
```

### 3b. Set MuJoCo Environment Variables

```bash
# Add to .bashrc so these persist across sessions
cat >> ~/.bashrc << 'EOF'
# MuJoCo paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export PATH="$LD_LIBRARY_PATH:$PATH"
EOF

source ~/.bashrc
conda activate reinflow
```

### 3c. Install Rendering Libraries via Conda (No Root)

```bash
conda activate reinflow

pip install patchelf
pip install 'cython<3.0.0'
conda install -c conda-forge glew -y
conda install -c conda-forge mesalib -y
conda install -c menpo glfw3 -y

# Set the include path for compilation
echo 'export CPATH=$CONDA_PREFIX/include' >> ~/.bashrc
source ~/.bashrc
conda activate reinflow
```

### 3d. Install mujoco-py

```bash
cd $HOME/.mujoco
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
pip install -r requirements.txt
pip install -r requirements.dev.txt
pip install -e . --no-cache
```

**Verify mujoco-py:**

```bash
python -c "import mujoco_py; print('mujoco_py imported successfully')"
```

> **Troubleshooting:** If you see `GLIBCXX_3.4.30 not found`, run:
> ```bash
> cd $(python -c "import sys; print(sys.prefix)")/lib
> mv libstdc++.so.6 libstdc++.so.6.old
> ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
> ```

---

## Step 4: Install Robomimic, MuJoCo Python, and Robosuite

**IMPORTANT:** Make sure `conda activate reinflow` is active. Verify with:

```bash
which python
# Should show: /home/ishitag/miniconda3/envs/reinflow/bin/python
python --version
# Should show: Python 3.8.x
```

### 4a. Install cmake and robomimic

```bash
conda activate reinflow
cd ~/ReinFlow

pip install cmake==3.31.6
pip install robomimic==0.3.0
```

### 4b. Pin mujoco version, THEN install robosuite

**This is critical.** `robosuite==1.4.1` requires `mujoco>=2.3.0`. Without pinning, pip will try to install the latest mujoco (3.5.0+) which must build from source and **fails** on Babel with `RuntimeError: MUJOCO_PATH environment variable is not set`. Pin to `mujoco==3.1.6` first — it has a pre-built wheel for Python 3.8:

```bash
# Pin mujoco FIRST (pre-built wheel, no build from source needed)
pip install mujoco==3.1.6

# NOW install robosuite — it will see mujoco is already satisfied
pip install robosuite==1.4.1
```

### 4c. Fix egl_probe if You Get a CMake Error

If you see `RuntimeError: CMake must be installed` during install, fix egl_probe:

```bash
wget https://github.com/mhandb/egl_probe/archive/fix_windows_build.zip
pip install fix_windows_build.zip
rm fix_windows_build.zip
# Then re-install robomimic and robosuite
```

### 4d. Set Up Robosuite Macros for Headless Rendering

```bash
python $(python -c "import robosuite; import os; print(os.path.join(os.path.dirname(robosuite.__file__), 'scripts', 'setup_macros.py'))")
```

This copies `macros.py` to `macros_private.py`. Verify with:

```bash
python -c "import robosuite; import os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))"
```

> **Rendering note:** ReinFlow's `script/run.py` automatically sets `MUJOCO_GL=egl` when you pass `sim_device=cuda:0`, or falls back to `MUJOCO_GL=osmesa` when `sim_device` is omitted. On Babel GPU nodes with NVIDIA drivers, **EGL should work** and is ~3x faster than OSMesa.

---

## Step 5: Install D4RL (Required Dependency)

```bash
conda activate reinflow
pip install d4rl
```

---

## Step 6: Set ReinFlow Environment Paths

```bash
cd ~/ReinFlow
conda activate reinflow

# Run the path setup script — press ENTER to accept defaults for all prompts
# Default paths:
#   REINFLOW_DIR      = ~/ReinFlow
#   REINFLOW_DATA_DIR = ~/ReinFlow/data
#   REINFLOW_LOG_DIR  = ~/ReinFlow/log
# For the WandB entity prompt: just press ENTER to skip (we'll disable WandB)
bash ./script/set_path.sh

source ~/.bashrc
conda activate reinflow
```

**Verify paths are set:**

```bash
echo "REINFLOW_DIR=$REINFLOW_DIR"
echo "REINFLOW_DATA_DIR=$REINFLOW_DATA_DIR"
echo "REINFLOW_LOG_DIR=$REINFLOW_LOG_DIR"
```

---

## Step 7: Install the ReinFlow Package

```bash
cd ~/ReinFlow
conda activate reinflow
pip install -e .
```

**Verify installation:**

```bash
python -c "import util; print('ReinFlow package installed successfully')"
```

---

## Step 8: Fine-tune with ReinFlow on Robomimic

The fine-tuning scripts will **automatically download** the pre-trained checkpoint and normalization data on first run. No manual downloads needed.

All commands must be run from the ReinFlow root directory:

```bash
cd ~/ReinFlow
conda activate reinflow
```

### Available Robomimic Tasks (Increasing Difficulty)

| Task | Config Dir | ReFlow Checkpoint | Description |
|------|-----------|-------------------|-------------|
| **Lift** | `cfg/robomimic/finetune/lift` | **NOT available** (must pre-train yourself) | Single-arm, pick up a cube |
| **Can** | `cfg/robomimic/finetune/can` | **Auto-downloads** | Single-arm, pick and place a can |
| **Square** | `cfg/robomimic/finetune/square` | **Auto-downloads** | Single-arm, nut assembly |
| **Transport** | `cfg/robomimic/finetune/transport` | Coming soon (must pre-train yourself) | Dual-arm, object transport |

> **Start with `can`** — it's the simplest task that has a ReFlow checkpoint ready for auto-download.

### Fine-tuning Commands

**With EGL rendering (fast, recommended if GPU drivers support EGL):**

> **Hydra note:** Most configs don't define `sim_device`, so you must use `++sim_device=cuda:0` (double `+` means "override if exists, append if not").

```bash
# Lift (easiest — good for a first test)
python script/run.py \
  --config-dir=cfg/robomimic/finetune/lift \
  --config-name=ft_ppo_reflow_mlp_img \
  device=cuda:0 ++sim_device=cuda:0 \
  wandb=null

# Can
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_reflow_mlp_img \
  device=cuda:0 ++sim_device=cuda:0 \
  wandb=null

# Square
python script/run.py \
  --config-dir=cfg/robomimic/finetune/square \
  --config-name=ft_ppo_reflow_mlp_img \
  device=cuda:0 ++sim_device=cuda:0 \
  wandb=null

# Transport
python script/run.py \
  --config-dir=cfg/robomimic/finetune/transport \
  --config-name=ft_ppo_reflow_mlp_img \
  device=cuda:0 ++sim_device=cuda:0 \
  wandb=null
```

**With OSMesa rendering (fallback if EGL fails — slower but always works headless):**

```bash
# Simply omit sim_device — run.py defaults to MUJOCO_GL=osmesa
python script/run.py \
  --config-dir=cfg/robomimic/finetune/lift \
  --config-name=ft_ppo_reflow_mlp_img \
  device=cuda:0 \
  wandb=null
```

### Key Flags Explained

| Flag | Purpose |
|------|---------|
| `device=cuda:0` | GPU for model computation |
| `++sim_device=cuda:0` | GPU for MuJoCo rendering (EGL). Omit for OSMesa. Use `++` since most configs don't define this key |
| `wandb=null` | Disable WandB logging (training data still saved to `.pkl`) |
| `wandb.offline_mode=true` | Log to WandB offline, sync later |
| `seed=42` | Random seed (default; change for multiple runs) |
| `env.n_envs=50` | Number of parallel envs (reduce if OOM, e.g., `env.n_envs=20`) |

### If You Run Out of GPU Memory

Reduce parallel environments and increase rollout steps to keep total samples consistent:

```bash
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_reflow_mlp_img \
  device=cuda:0 ++sim_device=cuda:0 \
  wandb=null \
  env.n_envs=20 train.n_steps=750
```

The total samples per iteration = `n_envs x n_steps x act_steps`. Keep this product roughly constant when adjusting.

---

## Step 9: Running Long Training Jobs on Babel

Interactive `srun` sessions will terminate when your SSH connection drops or the time limit expires. For long runs, use one of these approaches:

### Option A: Background Process in Interactive Session

```bash
srun -p general --gres=gpu:1 --time=24:00:00 --mem=32G --pty /bin/bash

conda activate reinflow
cd ~/ReinFlow

nohup python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_reflow_mlp_img \
  device=cuda:0 ++sim_device=cuda:0 \
  wandb=null \
  > ./ft_can_reflow.log 2>&1 &

# Check progress
tail -f ./ft_can_reflow.log
```

### Option B: SLURM Batch Script (Recommended for Long Jobs)

Create a file `run_reinflow.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=reinflow-can
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Initialize conda in batch mode
source ~/miniconda3/etc/profile.d/conda.sh
conda activate reinflow

cd ~/ReinFlow

export REINFLOW_DIR="$HOME/ReinFlow"
export REINFLOW_DATA_DIR="$HOME/ReinFlow/data"
export REINFLOW_LOG_DIR="$HOME/ReinFlow/log"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export HYDRA_FULL_ERROR=1

python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_reflow_mlp_img \
  device=cuda:0 ++sim_device=cuda:0 \
  wandb=null
```

Submit and monitor:

```bash
sbatch run_reinflow.sbatch
squeue -u ishitag          # check job status
tail -f slurm-<JOBID>.out  # watch output
```

---

## Step 10: Resume Training After a Crash

If training crashes or gets preempted, resume from the last checkpoint:

```bash
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_reflow_mlp_img \
  device=cuda:0 ++sim_device=cuda:0 \
  wandb=null \
  ++resume_path=<PATH_TO_CHECKPOINT>.pt
```

Checkpoints are saved to `$REINFLOW_LOG_DIR/robomimic/finetune/<run_name>/` every `save_model_freq` iterations (default: 100).

> **Note:** If `resume_path` is not in the config, add `resume_path: null` to the YAML file first, then override it from the command line.

---

## Step 11: Evaluate a Fine-tuned Checkpoint

```bash
python script/run.py \
  --config-dir=cfg/robomimic/eval/can \
  --config-name=eval_reflow_mlp_img \
  base_policy_path=<PATH_TO_FINETUNED_CHECKPOINT>.pt \
  denoising_step_list=[1,2,4,8] \
  load_ema=False \
  device=cuda:0 ++sim_device=cuda:0
```

---

## Quick Reference: Full Command Sequence (Copy-Paste)

```bash
# --- On Babel GPU node ---
conda activate reinflow
cd ~/ReinFlow

# Verify correct Python
which python   # must show .../envs/reinflow/bin/python
python --version  # must show Python 3.8.x

# Verify env vars
echo "REINFLOW_DIR=$REINFLOW_DIR"
echo "REINFLOW_DATA_DIR=$REINFLOW_DATA_DIR"
echo "REINFLOW_LOG_DIR=$REINFLOW_LOG_DIR"
nvidia-smi  # confirm GPU is visible

# Run fine-tuning (Can task as example)
python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_reflow_mlp_img \
  device=cuda:0 ++sim_device=cuda:0 \
  wandb=null
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `MUJOCO_PATH environment variable is not set` (mujoco build fails) | You're installing the wrong mujoco version. Run `pip install mujoco==3.1.6` first, then `pip install robosuite==1.4.1` |
| `Defaulting to user installation because normal site-packages is not writeable` | You forgot `conda activate reinflow`. Activate it and re-run pip install |
| `srun: error: Slurm job XXXXX has expired` | Run `unset SLURM_JOB_ID` or `exit` and SSH in fresh, then re-run `srun` |
| Debug partition times out during install | Use `-p general --time=02:00:00` for installation steps |
| `ConfigAttributeError: Key 'sim_device' is not in struct` | Use `++sim_device=cuda:0` (double `+`) instead of `sim_device=cuda:0` |
| `libEGL warning: failed to open /dev/dri/renderD*` | EGL not available on this node. Remove `++sim_device=cuda:0` to fall back to OSMesa |
| `CUDA out of memory` | Reduce `env.n_envs` (e.g., 20) and increase `train.n_steps` proportionally |
| `ImportError: cannot import name 'util'` | Run `pip install -e .` from the ReinFlow root directory |
| `GLIBCXX_3.4.30 not found` | See the symlink fix in Step 3d |
| `FileNotFoundError: patchelf` | Run `pip install patchelf` |
| `gdown` download fails (quota/auth) | Manually download checkpoints from [HuggingFace](https://huggingface.co/datasets/ReinFlow/ReinFlow-data-checkpoints-logs/) and place in the expected paths |
| WandB login prompt blocks training | Use `wandb=null` to disable, or `wandb.offline_mode=true` for offline logging |
| `#include <GL/glew.h>` error | Ensure Step 3c conda packages are installed and `CPATH=$CONDA_PREFIX/include` is set |
