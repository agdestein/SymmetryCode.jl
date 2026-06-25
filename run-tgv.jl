# Apply the closures trained on forced HIT (by run-les.jl) — *unchanged* — to a
# decaying Taylor-Green vortex. The TGV probes generalization to (1) laminar →
# transitional → turbulent regimes, (2) an unforced decaying flow, and (3) the
# canonical dissipation benchmark; its filter-scale Reynolds number sweeps a range
# over the transition/decay, a within-flow Re_Δ check.
#
# Models are reused verbatim (they regress the normalized target τ/(Δ²‖∇u‖²) from
# the normalized gradient): a TGV run is just another set of (dns, Δf) eval points,
# coordinate `(; visc, seed, role=:tgv, Re_target)` from `tgv_runs()`. No training
# here — run-les.jl owns the ps-*.jld2.

@info "Loading packages"
flush(stderr)

using Adapt
using CairoMakie
using CUDA, cuDNN
using JLD2
using Statistics: mean

import SymmetryCode as S

get_case() = S.case_snellius()

get_config() = (;
    # Must match the trained set in run-les.jl (these reuse those ps-*.jld2).
    archs = (:conv, :equi, :tbnn),
    tiers = (:saturated,),
    use_redelta = (false, true),
    netseeds = 0:0,
    classical = [:nomo, :clar],

    experiments = [
        :data,            # create_data_tgv -> dnsmetafile + fieldsfile/lesmeta per Δ
        :apriori,         # compute_sfs_stats (reduce-on-the-fly a-priori)
        :aposteriori,     # solve_les (decaying rollout, reduce-on-the-fly)
        :plots,           # per-eval-point figures
        :dissipation,     # plot_dissipation_tgv (the benchmark)
        :field_evolution, # plot_field_evolution_tgv montage
    ],
    force = Set{Symbol}([]),
)

learned_models(c) = [
    (; arch, tier, netseed, use_redelta = ur)
        for arch in c.archs for tier in c.tiers
        for ur in c.use_redelta for netseed in c.netseeds
]
eval_models(c) = [c.classical; learned_models(c)]

# Learned families (no seed) for the seed-aggregated bars; a curated saturated,
# one-seed subset for the per-curve series plots (see run-les.jl for the rationale).
families(c) = [
    (; arch, tier, use_redelta = ur)
        for arch in c.archs for tier in c.tiers for ur in c.use_redelta
]
series_models(c) = [
    c.classical;
    [
        (; arch, tier = last(c.tiers), netseed = first(c.netseeds), use_redelta = ur)
            for arch in c.archs for ur in c.use_redelta
    ]
]
buildone(case, setup, m) = S.build_models(case, setup, [m])[S.modelname(m)]

function main()
    case = get_case()
    config = get_config()
    models = eval_models(config)
    fams = families(config)

    for tgv in S.tgv_runs()
        @info "===== TGV run: Re=$(tgv.Re_target), visc=$(tgv.visc) ====="
        flush(stderr)

        :data in config.experiments &&
            S.create_data_tgv(case, tgv; force = :data in config.force)

        for Δf in case.filters_test
            @info "----- TGV eval: Δ=$(Δf) -----"
            flush(stderr)
            setup = S.make_setup(case, tgv, Δf)

            # Reference a-priori stats + a-posteriori budget (no model).
            :apriori in config.experiments &&
                S.compute_sfs_stats(case, :ref, tgv, Δf; force = :apriori in config.force)
            :aposteriori in config.experiments &&
                S.solve_les(case, :ref, tgv, Δf; force = :aposteriori in config.force)

            for m in models
                S.clean()
                local built = nothing
                getmodel() = (built === nothing && (built = buildone(case, setup, m)); built)
                if :apriori in config.experiments
                    S.compute_sfs_stats(case, m, tgv, Δf, getmodel; force = :apriori in config.force)
                end
                :aposteriori in config.experiments &&
                    S.solve_les(case, m, tgv, Δf, getmodel; force = :aposteriori in config.force)
                built = nothing
                S.clean()
            end

            if :plots in config.experiments
                series = series_models(config)
                S.plot_apriori_bar(case, tgv, Δf, fams, config.netseeds; classical = config.classical)
                S.plot_dissipation_bar(case, tgv, Δf, fams, config.netseeds; classical = config.classical)
                S.plot_error_post(case, tgv, Δf, series)
                S.plot_spectrum_les(case, tgv, Δf, [:ref; series])
            end
            :dissipation in config.experiments &&
                S.plot_dissipation_tgv(case, tgv, Δf, [:ref; series_models(config)])
            :field_evolution in config.experiments &&
                S.plot_field_evolution_tgv(case, tgv, Δf)
        end
    end

    @info "Done."
    flush(stderr)
    return
end

main()
