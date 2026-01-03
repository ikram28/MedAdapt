#!/bin/bash
#SBATCH --job-name=SFT
#SBATCH -C h100
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=3          
#SBATCH --gres=gpu:3                
#SBATCH --cpus-per-task=8           
#SBATCH --hint=nomultithread         
#SBATCH --time=20:00:00        
#SBATCH --output=./logs_slurm/SFT%j.out 
#SBATCH --error=./logs_slurm/SFT%j.err  
#SBATCH --account=XXX@h100
#SBATCH --mail-user=XXX
#SBATCH --mail-type=ALL

module purge


module load arch/h100
module load pytorch-gpu/py3/2.7.0

# Echo des commandes lancees
set -x -e

export OMP_NUM_THREADS=8

export CUDA_LAUNCH_BLOCKING=1

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

srun -l python -u sft.py \
   --model_name="" \
   --path_train_dataset="" \
   --path_eval_dataset="" \
   --output_dir="" \
   --logging_dir="" \
   --epochs=10 \
   --batch_size=4 \
   --save_steps=50 \
   --logging_steps=10 \
   --seed=42 \
   --learning_rate=2e-05
