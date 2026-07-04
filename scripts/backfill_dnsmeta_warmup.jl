# Backfill the warm-up series into existing light `dns_meta.jld2` artifacts.
#
# The data-window evolution figure (`plot_evolution_data`) draws the DNS warm-up
# tail at negative times. It now reads that series — `times_warmup` /
# `statistics_warmup` — from the *light* `dns_meta.jld2` instead of the heavy
# `dns.jld2` (which `scripts/pull_results.sh` excludes). New data carries the keys
# automatically (see `create_data`); this script adds them to metadata produced
# *before* that change, so you don't have to rerun the GPU data generation.
#
# Run it WHERE THE HEAVY `dns.jld2` STILL LIVE (i.e. on the cluster), *before*
# pulling the light artifacts to your laptop with `scripts/pull_results.sh`. It
# walks the artifact root, and for every `dns.jld2` it finds, reads the warm-up
# `times`/`statistics` and merges them into the sibling `dns_meta.jld2`
# (preserving every existing key). Idempotent — rerunning just overwrites the two
# warm-up keys. TGV runs have no `dns.jld2` (no forced warm-up) and are skipped
# naturally.
#
# Usage:
#   julia --project scripts/backfill_dnsmeta_warmup.jl [ROOT]
# ROOT defaults to `case_snellius().rootdir` (the cluster path unless
# SYMMETRY_ROOTDIR is set).

using JLD2

import SymmetryCode as S

function backfill(root)
    isdir(root) || error("artifact root not found: $(root)")
    @info "Backfilling warm-up series under $(root)"
    nfound = nwritten = 0
    for (dir, _, files) in walkdir(root)
        "dns.jld2" in files || continue
        nfound += 1
        heavy = joinpath(dir, "dns.jld2")
        metafile = joinpath(dir, "dns_meta.jld2")
        if !isfile(metafile)
            @warn "no dns_meta.jld2 beside $(heavy); skipping (run create_data first)"
            continue
        end
        times_warmup, statistics_warmup = load(heavy, "times", "statistics")

        # Preserve every existing key, add/overwrite the warm-up series.
        data = load(metafile)
        data["times_warmup"] = times_warmup
        data["statistics_warmup"] = statistics_warmup
        S.jldsave_atomic(metafile; (Symbol(k) => v for (k, v) in data)...)
        nwritten += 1
        @info "warm-up series ($(length(times_warmup)) samples)  ->  $(metafile)"
    end
    @info "Done: updated $(nwritten)/$(nfound) dns_meta.jld2 file(s)."
    return nwritten
end

backfill(isempty(ARGS) ? S.case_snellius().rootdir : first(ARGS))
