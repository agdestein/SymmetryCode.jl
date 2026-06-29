#!/usr/bin/env bash
# Submit the whole Re_Δ pipeline as one SLURM `afterok` DAG — no manual --array
# toggling, no "run twice to aggregate". Each stage fans out a per-unit array, then
# a serial `reduce` once the array finishes; stages chain in order. Crucially the
# LES `convsym` array depends on the `models` array (afterok), so the symmetrized
# MLP only runs once the :conv params it reuses are on disk.
#
#   data ─array→ reduce ─→ models ─array→ convsym ─array→ reduce ─→ data ─array→ models ─array→ reduce
#   └─ run-dns ─┘      └──────────────── run-les ───────────────┘  └────────── run-tgv ──────────┘
#
# Usage:  ./submit.sh [stage]      stage ∈ data | les | tgv | all   (default all)
# Run a single stage once its inputs exist (e.g. `./submit.sh les` after `data`).

set -eo pipefail   # stop on first error
cd "$(dirname "$0")"
stage=${1:-all}

# --- array sizes (must match the active config in each driver) ----------------
# Hard-coded because the login node can't run Julia without a long recompile. They
# change only when you change a sweep axis (archs/sizes/seeds, dns_runs, tgv_runs).
# Recompute on a compute node via each driver's `count` phase, e.g.
#     sbatch --wrap 'julia --project run-les.jl count'   # logs "<models> <convsym>"
# (read the value from the job's stdout), then update the matching variable.
N_DNS=5            # run-dns.jl : length(dns_runs().all)
N_MODELS=69        # run-les.jl     : length(les_worklist)
N_CONVSYM=22       # run-les.jl     : length(convsym_models)
N_TGVDATA=1        # run-tgv.jl     : length(tgv_runs())
N_TGVMODELS=14     # run-tgv.jl     : length(eval_models)

# sub <driver> <phase> [extra sbatch flags...]  ->  echoes the new job id.
# job.sh evaluates `julia --project <driver> <phase>`; the flags are per-phase
# overrides (--array / --time / --dependency) layered on job.sh's #SBATCH defaults.
sub() { local drv=$1 ph=$2; shift 2; sbatch --parsable "$@" job.sh "$drv" "$ph"; }

last=""   # job id the next stage waits on (links the stages end-to-end)
# Populate DEP with an afterok flag on the previous stage's reduce, for the first
# job of a stage. Empty when a stage is started on its own (inputs must exist).
first_dep() { DEP=(); [ -n "$last" ] && DEP=(--dependency=afterok:"$last"); return 0; }

if [[ $stage == data || $stage == all ]]; then
    first_dep
    d=$(sub run-dns.jl data --time=20:00:00 --array=1-"$N_DNS" "${DEP[@]}")
    last=$(sub run-dns.jl reduce --time=00:30:00 --dependency=afterok:"$d")
    echo "run-dns: data[$d] (1-$N_DNS) -> reduce[$last]"
fi

if [[ $stage == les || $stage == all ]]; then
    first_dep
    m=$(sub run-les.jl models --time=01:00:00 --array=1-"$N_MODELS" "${DEP[@]}")
    c=$(sub run-les.jl convsym --time=02:00:00 --array=1-"$N_CONVSYM" --dependency=afterok:"$m")
    last=$(sub run-les.jl reduce --time=00:30:00 --dependency=afterok:"$c")
    echo "run-les: models[$m] (1-$N_MODELS) -> convsym[$c] (1-$N_CONVSYM) -> reduce[$last]"
fi

if [[ $stage == tgv || $stage == all ]]; then
    first_dep
    d=$(sub run-tgv.jl data --time=20:00:00 --array=1-"$N_TGVDATA" "${DEP[@]}")
    m=$(sub run-tgv.jl models--time=02:00:00  --array=1-"$N_TGVMODELS" --dependency=afterok:"$d")
    last=$(sub run-tgv.jl reduce --time=00:30:00 --dependency=afterok:"$m")
    echo "run-tgv: data[$d] (1-$N_TGVDATA) -> models[$m] (1-$N_TGVMODELS) -> reduce[$last]"
fi

echo "Submitted. Final job: $last  (watch with: squeue --me)"
