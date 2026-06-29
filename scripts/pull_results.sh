#!/usr/bin/env bash
# Pull the plotting artifacts from the cluster to this laptop, skipping the heavy
# DNS/field series. Run this FROM THE LAPTOP.
#
# `scp` has no include/exclude mechanism — it copies a directory wholesale — so
# this uses `rsync`, which does (`--exclude`), runs over SSH, preserves the tree,
# and is incremental (re-run to fetch only what changed).
#
# Heavy files (skipped unless --heavy): dns.jld2 (DNS warm-up plots), fields.jld2
# (velocity-field / TGV-field plots), apostfields-*.jld2 (showcase velocity plots).
# Everything else under the root (ps-*, apost-*, sfsstats-*, seedstats, les_meta,
# dns_meta, equiprior-*, redelta-binning) is what the saturation curve, the
# bars/tables, the Re_Δ trend, and the density/budget/spectra/error series need.
#
# For the Re_Δ trend figure, run `scripts/backfill_lesmeta.jl` ON THE CLUSTER first
# so `les_meta.jld2` carries `redelta_mean` (otherwise the trend would need the
# heavy fields.jld2).
#
# Usage:
#   scripts/pull_results.sh [--heavy] [--dry-run] REMOTE [DEST]
#
#   REMOTE  "user@host:/remote/root", or just "user@host" (the default cluster
#           path is appended).
#   DEST    local destination root; defaults to $SYMMETRY_ROOTDIR.
#
# Examples:
#   export SYMMETRY_ROOTDIR=~/symmetry-results/redelta
#   scripts/pull_results.sh me@snellius.surf.nl
#   scripts/pull_results.sh --dry-run me@snellius.surf.nl
#   scripts/pull_results.sh --heavy me@snellius.surf.nl ~/symmetry-results/redelta-full

# No `set -u`: macOS ships bash 3.2, where "${empty_array[@]}" trips an unbound-
# variable error. The required values are checked explicitly below instead.
set -eo pipefail

# Keep in sync with SymmetryCode.cluster_rootdir in src/experiment.jl.
CLUSTER_PATH="/projects/prjs1757/SymmetryOutput/redelta"

heavy=0
dryrun=()
pos=()
for arg in "$@"; do
    case "$arg" in
        --heavy)   heavy=1 ;;
        --dry-run) dryrun=(--dry-run) ;;
        -*)        echo "unknown flag: $arg" >&2; exit 2 ;;
        *)         pos+=("$arg") ;;
    esac
done

if [ "${#pos[@]}" -lt 1 ]; then
    echo "usage: $0 [--heavy] [--dry-run] REMOTE [DEST]" >&2
    exit 2
fi

remote="${pos[0]}"
# Append the default cluster path when REMOTE is only a host (no ":path").
case "$remote" in
    *:*) ;;
    *)   remote="${remote}:${CLUSTER_PATH}" ;;
esac

dest="${pos[1]:-$SYMMETRY_ROOTDIR}"
if [ -z "$dest" ]; then
    echo "no destination: set SYMMETRY_ROOTDIR or pass DEST as the second argument" >&2
    exit 2
fi

excludes=()
if [ "$heavy" -eq 0 ]; then
    excludes=(--exclude='dns.jld2' --exclude='fields.jld2' --exclude='apostfields-*.jld2')
fi

if [ "$heavy" -eq 1 ]; then kind=all; else kind=light; fi
echo "Pulling $kind artifacts"
echo "  from $remote"
echo "  to   $dest"

mkdir -p "$dest"
# Trailing slashes: copy the *contents* of the remote root into DEST.
rsync -avzh --partial --prune-empty-dirs "${dryrun[@]}" "${excludes[@]}" \
    "${remote%/}/" "${dest%/}/"
