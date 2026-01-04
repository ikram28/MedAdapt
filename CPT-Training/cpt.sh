#!/bin/bash
#SBATCH --job-name=CPT
#SBATCH --constraint=h100
#SBATCH --ntasks=32                  
#SBATCH --ntasks-per-node=4          
#SBATCH --gres=gpu:4                
#SBATCH --cpus-per-task=24          
#SBATCH --hint=nomultithread         
#SBATCH --time=20:00:00       
#SBATCH --output=slurm_logs/train%j.out 
#SBATCH --error=slurm_logs/train%j.err 
#SBATCH --account=XXX@h100
#SBATCH --mail-user=XXX
#SBATCH --mail-type=ALL



module purge
module load arch/h100
module load pytorch-gpu/py3/2.7.0

# Echo des commandes lancees
set -x -e

export OMP_NUM_THREADS=24

export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1



srun -l python -u cpt.py \
   --model_name="" \
   --tokenizer_name="" \
   --path_dataset="" \
   --output_dir="" \
   --epochs=3 \
   --batch_size=2 \
   --save_steps=50 \
   --logging_steps=10 \
   --seed=42 \
   --learning_rate=2e-05
