#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=10:00:00
# #SBATCH --array=26-50
# #SBATCH --time=00:30:00
# #SBATCH -o toto
# #SBATCH --mail-type=BEGIN,END
# #SBATCH --mail-user=sda@cwi.nl

# Note:
# https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+partitions+and+accounting

echo "Slurm job ID: $SLURM_JOB_ID"
echo "Slurm array task ID: $SLURM_ARRAY_TASK_ID"

# First create data, then run LES
# julia --project create-data.jl
julia --project run-les.jl
# julia --project run-tgv.jl
