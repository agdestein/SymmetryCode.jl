#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=2:00:00
# #SBATCH --mail-type=BEGIN,END
# #SBATCH --mail-user=sda@cwi.nl
# https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+partitions+and+accounting

# Generic runner: it just evaluates `julia --project <args>`. `submit.sh` hands it
# the driver + phase and the per-phase sbatch overrides (--array, --time,
# --dependency); the directives above are the defaults. Run a single piece by hand
# with e.g.  `sbatch job.sh run-les.jl reduce`  or  `sbatch --array=1-56 job.sh
# run-les.jl models`. A `reduce` phase needs no GPU — point it at a CPU partition
# (`sbatch --partition=... --gpus=0 job.sh run-les.jl reduce`) to save GPU hours.

echo "Slurm job $SLURM_JOB_ID  array task ${SLURM_ARRAY_TASK_ID:-none}  ::  julia $*"
julia --project "$@"
