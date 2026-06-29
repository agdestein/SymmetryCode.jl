# Backfill `redelta_mean` into existing light `les_meta.jld2` artifacts.
#
# The Re_Δ trend figure (`plot_trend_vs_redelta`) now reads its x-coordinate — the
# series-mean global Re_Δ — from the *light* `les_meta.jld2` instead of the heavy
# `fields.jld2`. New data carries it automatically (see `create_data`); this script
# adds it to artifacts produced *before* that change, so you don't have to rerun
# the GPU data generation.
#
# Run it WHERE THE HEAVY `fields.jld2` STILL LIVE (i.e. on the cluster), *before*
# pulling the light artifacts to your laptop with `scripts/pull_results.sh`. It
# walks the artifact root, and for every `fields.jld2` it finds, reads `redelta`,
# takes the mean, and merges it into the sibling `les_meta.jld2` (preserving
# `spectra_les`). Idempotent — rerunning just overwrites `redelta_mean`.
#
# Usage:
#   julia --project scripts/backfill_lesmeta.jl [ROOT]
# ROOT defaults to `case_snellius().rootdir` (the cluster path unless
# SYMMETRY_ROOTDIR is set).

using JLD2
using Statistics: mean

import SymmetryCode as S

function backfill(root)
    isdir(root) || error("artifact root not found: $(root)")
    @info "Backfilling redelta_mean under $(root)"
    nfound = nwritten = 0
    for (dir, _, files) in walkdir(root)
        "fields.jld2" in files || continue
        nfound += 1
        fieldsfile = joinpath(dir, "fields.jld2")
        lesfile = joinpath(dir, "les_meta.jld2")
        redelta = load(fieldsfile, "redelta")
        rmean = mean(redelta)

        # Preserve every existing key (currently spectra_les), add/overwrite ours.
        data = isfile(lesfile) ? load(lesfile) : Dict{String, Any}()
        isfile(lesfile) || @warn "no les_meta.jld2 beside $(fieldsfile); creating one with redelta_mean only"
        data["redelta_mean"] = rmean
        S.jldsave_atomic(lesfile; (Symbol(k) => v for (k, v) in data)...)
        nwritten += 1
        @info "redelta_mean = $(round(rmean; sigdigits = 5))  ->  $(lesfile)"
    end
    @info "Done: updated $(nwritten)/$(nfound) les_meta.jld2 file(s)."
    return nwritten
end

backfill(isempty(ARGS) ? S.case_snellius().rootdir : first(ARGS))
