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
# Arrays are *exact*: the first requested stage runs the driver's `pending` phase
# right here (login node — a pure `isfile` walk over the path functions, no GPU)
# and submits only the units whose artifacts are missing, so a gap-fill rerun
# never burns a GPU-node Julia load on a cached unit. Later stages can't know
# their gaps until the previous reduce ran, so each stage's tail submits a small
# `staging`-partition plan job that re-invokes this script for the remaining
# stages (afterok on the reduce). Empty pending lists skip the array; the reduce
# always runs (it regenerates the figures/tables).
#
# The inline `pending` needs Julia to load on non-GPU nodes without recompiling:
# JULIA_CPU_TARGET below makes the precompile cache multi-target across the AMD
# Zen 2 login/staging and Zen 4 H100 nodes (keep in sync with job.sh). The first
# run after setting it precompiles once — from then on all node types share it.
#
# Usage:  ./submit.sh [stage...]    stage ∈ data | les | tgv | all   (default all)
# Run a single stage once its inputs exist (e.g. `./submit.sh les` after `data`),
# or a chain (`./submit.sh les tgv`).

set -eo pipefail   # stop on first error
cd "$(dirname "$0")"

export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all"

# Expand/validate the requested stage list.
stages=()
for s in "${@:-all}"; do
    case $s in
        all) stages+=(data les tgv) ;;
        data | les | tgv) stages+=("$s") ;;
        *) echo "unknown stage '$s' (expected data|les|tgv|all)" >&2 && exit 1 ;;
    esac
done

# pending <driver>: run the driver's `pending` phase on this node and fill
# PENDING[<phase>] with the exact `--array` specs ("" = everything cached).
declare -A PENDING
pending() {
    PENDING=()
    while IFS='=' read -r k v; do PENDING[$k]=$v; done < <(julia --project "$1" pending)
}

# sub <driver> <phase> [extra sbatch flags...]  ->  echoes the new job id.
# job.sh evaluates `julia --project <driver> <phase>`; the flags are per-phase
# overrides (--array / --time / --dependency) layered on job.sh's #SBATCH defaults.
sub() { local drv=$1 ph=$2; shift 2; sbatch --parsable "$@" job.sh "$drv" "$ph"; }

# Job id the next submission in this stage chains after ("" = no dependency).
last=""
dep() { [[ -n $last ]] && printf -- '--dependency=afterok:%s' "$last"; return 0; }

submit_data() {
    pending run-dns.jl
    if [[ -n ${PENDING[data]} ]]; then
        last=$(sub run-dns.jl data --time=20:00:00 --array="${PENDING[data]}")
        echo "run-dns: data[$last] --array=${PENDING[data]}"
    else
        echo "run-dns: data all cached"
    fi
    last=$(sub run-dns.jl reduce --time=00:30:00 $(dep))
    echo "run-dns: reduce[$last]"
}

submit_les() {
    pending run-les.jl
    if [[ -n ${PENDING[models]} ]]; then
        last=$(sub run-les.jl models --time=01:00:00 --array="${PENDING[models]}")
        echo "run-les: models[$last] --array=${PENDING[models]}"
    else
        echo "run-les: models all cached"
    fi
    if [[ -n ${PENDING[convsym]} ]]; then
        last=$(sub run-les.jl convsym --time=02:00:00 --array="${PENDING[convsym]}" $(dep))
        echo "run-les: convsym[$last] --array=${PENDING[convsym]}"
    else
        echo "run-les: convsym all cached"
    fi
    last=$(sub run-les.jl reduce --time=00:30:00 $(dep))
    echo "run-les: reduce[$last]"
}

submit_tgv() {
    pending run-tgv.jl
    if [[ -n ${PENDING[data]} ]]; then
        last=$(sub run-tgv.jl data --time=24:00:00 --array="${PENDING[data]}")
        echo "run-tgv: data[$last] --array=${PENDING[data]}"
    else
        echo "run-tgv: data all cached"
    fi
    if [[ -n ${PENDING[models]} ]]; then
        last=$(sub run-tgv.jl models --time=02:00:00 --array="${PENDING[models]}" $(dep))
        echo "run-tgv: models[$last] --array=${PENDING[models]}"
    else
        echo "run-tgv: models all cached"
    fi
    last=$(sub run-tgv.jl reduce --time=00:30:00 $(dep))
    echo "run-tgv: reduce[$last]"
}

# Plan + submit the first stage now; hand the remaining stages to a cheap CPU
# plan job that reruns this script once this stage's reduce has finished (its
# pending sets depend on the artifacts that reduce/arrays are about to write).
submit_"${stages[0]}"
rest=("${stages[@]:1}")
if ((${#rest[@]})); then
    plan=$(
        sbatch --parsable --partition=staging --time=01:00:00 --cpus-per-task=4 \
            --job-name="plan-${rest[0]}" --dependency=afterok:"$last" \
            --wrap "cd $PWD && ./submit.sh ${rest[*]}"
    )
    echo "plan[$plan]: './submit.sh ${rest[*]}' on staging after reduce $last"
fi

echo "Submitted. Watch with: squeue --me"
