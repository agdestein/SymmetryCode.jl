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

# Multi-target precompile cache so every Snellius node type shares one depot:
# login int* = EPYC 7F72, staging srv* = EPYC 7F32 (both Zen 2), gpu_h100 gcn* =
# EPYC 9334 (Zen 4). Without this, a cache built on a GPU node is rejected on the
# login node (and vice versa) and everything recompiles. Must be set when the
# cache is *created*, hence exported both here and in submit.sh — keep in sync.
# The companion fix for the GPU-less nodes is LocalPreferences.toml (committed):
# it pins the CUDA runtime to 13.2 (the H100 driver's max, works on any newer
# driver), because CUDA_Runtime_jll otherwise queries the driver at *precompile*
# time and a cache built on the login node bakes in "no runtime".
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all"

echo "Slurm job $SLURM_JOB_ID  array task ${SLURM_ARRAY_TASK_ID:-none}  ::  julia $*"
start_time=$(date +%s)
julia --project "$@"
elapsed=$(( $(date +%s) - start_time ))
echo "Done in $((elapsed / 3600))h $(( (elapsed % 3600) / 60 ))m $((elapsed % 60))s"
