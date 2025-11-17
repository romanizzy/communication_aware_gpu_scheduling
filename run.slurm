#!/bin/bash
#SBATCH --job-name=crux-ddp
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=crux_output_%j.txt

# Load modules
module load anaconda3
module load cuda/12.1
module load cudnn/8.x

source activate crux

# Set PyTorch distributed environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355

# Run your multi-GPU job
srun python main.py
