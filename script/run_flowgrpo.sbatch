#!/bin/bash
#SBATCH --job-name=flowgrpo-can
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Initialize conda in batch mode
source ~/miniconda3/etc/profile.d/conda.sh
conda activate reinflow

cd ~/cmu/spring_26/reinflow-robomimic

# ── Environment variables ─────────────────────────────────────────────
export REINFLOW_DIR="$HOME/reinflow-robomimic"
export REINFLOW_DATA_DIR="$HOME/reinflow-robomimic/data"
export REINFLOW_LOG_DIR="$HOME/reinflow-robomimic/log"
export REINFLOW_WANDB_ENTITY="katef-study"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export PATH="$LD_LIBRARY_PATH:$PATH"

echo
echo "# # # # # USING SREEHARSHA'S WANDB API KEY # # #  # #"
echo

export WANDB_API_KEY="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"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export HYDRA_FULL_ERROR=1


# ── Notes on memory ───────────────────────────────────────────────────
# --mem=80G covers:
#   ~2 GB  model (VisionFlowMLP actor_ft + actor_old, no critic)
#   ~2 GB  image buffer (numpy, CPU)
#   ~30 GB 50x AsyncVectorEnv worker subprocesses (image obs, MuJoCo)
#   ~10 GB overhead / OS
# No critic ViT saves ~40% GPU memory vs FPO++.

python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_flowgrpo_reflow_mlp_img \
  device=cuda:0 \
  seed=42
