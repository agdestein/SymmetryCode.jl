# Backfill `redelta_peak` into existing light TGV `les_meta.jld2` artifacts.
#
# `S.plot_tgv_vs_redelta` places the TGV on the Re_Δ trend's x-axis at the
# instant of peak DNS dissipation (the decay sweeps too wide a Re_Δ range for the
# forced-HIT series mean to be a meaningful single point). New TGV data carries
# `redelta_peak` automatically (`create_data_tgv` writes it via
# `S.redelta_peak_index`); this script adds it to TGV artifacts produced *before*
# that change, so you don't have to rerun the GPU data generation.
#
# Run it WHERE THE HEAVY `fields.jld2` STILL LIVE (i.e. on the cluster), *before*
# pulling the light artifacts down. TGV-specific — `S.tgv_runs() × case.filters_test`
# is the full set of TGV eval points, not a generic walk (unlike
# `backfill_lesmeta.jl`, the peak-instant concept doesn't apply to forced HIT,
# which is statistically stationary). Idempotent — rerunning just overwrites
# `redelta_peak`.
#
# Usage:
#   julia --project scripts/backfill_redelta_peak.jl

using JLD2

import SymmetryCode as S

function backfill(case)
    @info "Backfilling redelta_peak under $(case.rootdir)"
    nfound = nwritten = 0
    for tgv in S.tgv_runs(), Δf in case.filters_test
        ff = S.fieldsfile(case, tgv, Δf)
        isfile(ff) || (@warn "missing $(ff); skipping"; continue)
        nfound += 1
        rpk = S.compute_redelta_peak(case, tgv, Δf)

        # Preserve every existing key (spectra_les, redelta_mean), add/overwrite ours.
        lesfile = S.lesmetafile(case, tgv, Δf)
        data = isfile(lesfile) ? load(lesfile) : Dict{String, Any}()
        isfile(lesfile) || @warn "no les_meta.jld2 beside $(ff); creating one with redelta_peak only"
        data["redelta_peak"] = rpk
        S.jldsave_atomic(lesfile; (Symbol(k) => v for (k, v) in data)...)
        nwritten += 1
        @info "redelta_peak (Δf=$(Δf)) = $(round(rpk; sigdigits = 5))  ->  $(lesfile)"
    end
    @info "Done: updated $(nwritten)/$(nfound) les_meta.jld2 file(s)."
    return nwritten
end

backfill(S.case_snellius())
