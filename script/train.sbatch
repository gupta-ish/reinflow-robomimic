#!/bin/bash
#SBATCH --job-name=reinflow-seq
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
export REINFLOW_DIR=/home/ishitag/ReinFlow
export REINFLOW_DATA_DIR=/home/ishitag/ReinFlow/data
export REINFLOW_LOG_DIR=/home/ishitag/ReinFlow/log
export REINFLOW_WANDB_ENTITY="${REINFLOW_WANDB_ENTITY:?Set REINFLOW_WANDB_ENTITY before submitting}"
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY before submitting}"

export D4RL_SUPPRESS_IMPORT_ERROR=1
export HYDRA_FULL_ERROR=1

source /home/ishitag/miniconda3/etc/profile.d/conda.sh
conda activate reinflow

cd "$REINFLOW_DIR"
mkdir -p slurm_logs

SEED="${SEED:-42}"
TASKS="${TASKS:-can lift square transport}"

# ── Sequential training over all tasks ───────────────────────────────────────
for TASK in $TASKS; do
  echo "════════════════════════════════════════════════════════════════"
  echo "  Starting task: ${TASK}  (seed=${SEED})  $(date)"
  echo "════════════════════════════════════════════════════════════════"

  python script/run.py \
    --config-dir="cfg/robomimic/finetune/${TASK}" \
    --config-name=ft_ppo_reflow_mlp_img \
    device=cuda:0 \
    ++sim_device=cuda:0 \
    seed="${SEED}" \
    env.save_video=true \
    train.render.num=1 \
    train.render.freq=50

  echo "  Finished task: ${TASK}  $(date)"
  echo ""
done

echo "All tasks complete."
