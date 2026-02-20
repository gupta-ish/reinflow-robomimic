#!/bin/bash
#SBATCH --job-name=fpopp-can
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
  --config-name=ft_fpopp_reflow_mlp_img \
  device=cuda:0 ++sim_device=cuda:0 \
  wandb=
