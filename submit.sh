#!/usr/bin/env bash
# Submit the whole Re_Δ pipeline as one SLURM `afterok` DAG — no manual --array
# toggling, no "run twice to aggregate". Each stage fans out a per-unit array, then
# a serial `reduce` once the array finishes; stages chain in order. Crucially the
# LES `convsym` array depends on the `models` array (afterok), so the symmetrized
# MLP only runs once the :conv params it reuses are on disk.
#
#   data ─array→ reduce ─→ models ─array→ convsym ─array→ reduce ─→ data ─array→ models ─array→ reduce
#   └─ create-data ─┘      └──────────────── run-les ───────────────┘  └────────── run-tgv ──────────┘
#
# Usage:  ./submit.sh [stage]      stage ∈ data | les | tgv | all   (default all)
# Run a single stage once its inputs exist (e.g. `./submit.sh les` after `data`).
#
# Array sizes come from each driver's `count` phase (pure list arithmetic — no GPU),
# run here on the login node so they always track the config. If login-node Julia is
# undesirable, replace the `count` calls with literal numbers.

# -e: stop on first error; pipefail: a failed `count` pipe is an error. (No -u: an
# empty ${DEP[@]} must expand to nothing on the cluster's and older local bashes.)
set -eo pipefail
cd "$(dirname "$0")"
stage=${1:-all}

# sub <driver> <phase> [extra sbatch flags...]  ->  echoes the new job id.
# job.sh evaluates `julia --project <driver> <phase>`; the flags are per-phase
# overrides (--array / --time / --dependency) layered on job.sh's #SBATCH defaults.
sub() { local drv=$1 ph=$2; shift 2; sbatch --parsable "$@" job.sh "$drv" "$ph"; }
count() { julia --project "$1" count | tail -n1; }

last=""   # job id the next stage waits on (links the stages end-to-end)
# Populate DEP with an afterok flag on the previous stage's reduce, for the first
# job of a stage. Empty when a stage is started on its own (inputs must exist).
first_dep() { DEP=(); [ -n "$last" ] && DEP=(--dependency=afterok:"$last"); return 0; }

if [[ $stage == data || $stage == all ]]; then
    read -r N_DNS <<<"$(count create-data.jl)"
    first_dep
    d=$(sub create-data.jl data --array=1-"$N_DNS" "${DEP[@]}")
    last=$(sub create-data.jl reduce --time=00:20:00 --dependency=afterok:"$d")
    echo "create-data: data[$d] (1-$N_DNS) -> reduce[$last]"
fi

if [[ $stage == les || $stage == all ]]; then
    read -r N_MODELS N_CONVSYM <<<"$(count run-les.jl)"
    first_dep
    m=$(sub run-les.jl models --array=1-"$N_MODELS" "${DEP[@]}")
    c=$(sub run-les.jl convsym --array=1-"$N_CONVSYM" --dependency=afterok:"$m")
    last=$(sub run-les.jl reduce --time=00:30:00 --dependency=afterok:"$c")
    echo "run-les: models[$m] (1-$N_MODELS) -> convsym[$c] (1-$N_CONVSYM) -> reduce[$last]"
fi

if [[ $stage == tgv || $stage == all ]]; then
    read -r N_TGVDATA N_TGVMODELS <<<"$(count run-tgv.jl)"
    first_dep
    d=$(sub run-tgv.jl data --array=1-"$N_TGVDATA" "${DEP[@]}")
    m=$(sub run-tgv.jl models --array=1-"$N_TGVMODELS" --dependency=afterok:"$d")
    last=$(sub run-tgv.jl reduce --time=00:30:00 --dependency=afterok:"$m")
    echo "run-tgv: data[$d] (1-$N_TGVDATA) -> models[$m] (1-$N_TGVMODELS) -> reduce[$last]"
fi

echo "Submitted. Final job: $last  (watch with: squeue --me)"
