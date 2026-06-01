#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --mail-user=sda@cwi.nl
#SBATCH --time=10:00:00
# #SBATCH --time=00:30:00
# #SBATCH --array=1-6
# #SBATCH -o toto
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=16
# #SBATCH --mail-type=BEGIN,END

# Note:
# - gpu_a100: 18 cores
# - gpu_h100: 16 cores
# https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+partitions+and+accounting

# mkdir -p /scratch-shared/$USER

echo "Slurm job ID: $SLURM_JOB_ID"
echo "Slurm array task ID: $SLURM_ARRAY_TASK_ID"

# First create data, then run LES
# julia --project create-data.jl
# julia --project run-les.jl
julia --project run-tgv.jl
