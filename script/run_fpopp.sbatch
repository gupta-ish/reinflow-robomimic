#!/bin/bash
#SBATCH --job-name=fpopp-can
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Initialize conda in batch mode
source ~/miniconda3/etc/profile.d/conda.sh
conda activate reinflow

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

cd "$REINFLOW_DIR"

export PYTHONPATH="$REINFLOW_DIR:$PYTHONPATH"

echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "import gym; print('gym OK')" || echo "gym NOT FOUND"


python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_fpopp_reflow_mlp_img \
  device=cuda:0 ++sim_device=cuda:0 \

