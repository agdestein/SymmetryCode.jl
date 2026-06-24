# Train the LES closures over the multi-(ν, Δ) training pool and evaluate them
# a-priori + a-posteriori across the (ν, Δ) test grid — the Re_Δ experiment.
#
# Coordinate-driven: the sweep axes (architecture, size tier, Re_Δ on/off, network
# seed) are loose coordinates combined by Cartesian product, never fused into a
# monolithic config. Each eval point is a `(dns, Δf)` pair; artifacts self-locate
# from coordinates via the path functions.

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
    # Learned-model sweep axes (Cartesian product → families × seeds).
    archs = (:conv, :equi, :tbnn),
    tiers = (:saturated,),
    use_redelta = (false, true),
    netseeds = 0:0,

    # Classical baselines in every per-eval-point comparison (no Re_Δ / seeds).
    classical = [:nomo, :clar],

    # Whether to (re)train. Training spans the whole trainpool, once per coordinate.
    train = true,

    # Pipeline stages.
    experiments = [
        :apriori,          # predict_sfs + compute_sfs_stats
        :aposteriori,      # solve_les (reduce-on-the-fly)
        :equivariance,     # apriori_equivariance_error (learned only)
        :redelta_binning,  # Phase-0 pointwise-Re_Δ diagnostic
        :seeds,            # get_seed_statistics per eval point (feeds trend + tables)
        :plots,            # per-eval-point figures
        :trend,            # plot_trend_vs_redelta (the H2 figure)
        :tables,           # write_errors_table
        # :showcase,       # savefields rollout + velocity/SFS field montages
    ],

    # Stages whose cache is invalidated this run.
    force = Set{Symbol}([]),
)

# Learned-model families (no seed) and the full model list (with seed) per point.
families(c) = [
    (; arch, tier, use_redelta = ur)
        for arch in c.archs for tier in c.tiers for ur in c.use_redelta
]
learned_models(c) = [
    (; arch, tier, netseed, use_redelta = ur)
        for arch in c.archs for tier in c.tiers
        for ur in c.use_redelta for netseed in c.netseeds
]
eval_models(c) = [c.classical; learned_models(c)]

# Build a single closure (classical symbol or learned coordinate) against `setup`.
buildone(case, setup, m) = S.build_models(case, setup, [m])[S.modelname(m)]

# Per-script table file (shares case.plotdir with create-data.jl).
tablefile() = "tables-les.txt"
reset_tables(case; kwargs...) = S.reset_tables(case; filename = tablefile(), kwargs...)
tabulate(args...; kwargs...) = S.tabulate(args...; filename = tablefile(), kwargs...)

function main()
    case = get_case()
    config = get_config()
    reset_tables(case)

    fams = families(config)
    models = eval_models(config)
    trainpool = S.build_trainpool(case)
    testpoints = [(dns, Δf) for dns in S.dns_runs().test for Δf in case.filters_test]

    #######################
    # Train (once per coordinate, over the whole trainpool)
    #######################
    if config.train
        S.train_models(
            case, trainpool;
            config.archs, config.tiers, config.netseeds, config.use_redelta,
            force = :train in config.force,
        )
        timings = NamedTuple(
            S.modelkey(m) => (isfile(S.psfile(case, m)) ? load(S.psfile(case, m), "timing") : :missing)
                for m in learned_models(config)
        )
        tabulate(case, "Training wall-time (s) per learned coordinate", timings; digits = 1)
    end

    #######################
    # Per-eval-point evaluation across the (ν, Δ) test grid
    #######################
    for (dns, Δf) in testpoints
        @info "===== eval point: role=$(dns.role), visc=$(dns.visc), Δ=$(Δf) ====="
        flush(stderr)
        setup = S.make_setup(case, dns, Δf)
        withref = [:ref; models]

        # Reference a-posteriori budget/transfer (no model).
        :aposteriori in config.experiments &&
            S.solve_les(case, :ref, dns, Δf; force = :aposteriori in config.force)

        # Per model: lazy build, GPU reclaimed between models.
        for m in models
            S.clean()
            local built = nothing
            getmodel() = (built === nothing && (built = buildone(case, setup, m)); built)

            if :apriori in config.experiments && m !== :nomo
                S.predict_sfs(case, m, dns, Δf, getmodel; force = :apriori in config.force)
            end
            if :aposteriori in config.experiments
                S.solve_les(case, m, dns, Δf, getmodel; force = :aposteriori in config.force)
            end
            if :equivariance in config.experiments && m isa NamedTuple
                S.apriori_equivariance_error(
                    case, m, dns, Δf, getmodel;
                    force = :equivariance in config.force,
                )
            end
            built = nothing
            S.clean()
        end

        :apriori in config.experiments &&
            S.compute_sfs_stats(case, withref, dns, Δf; force = :apriori in config.force)

        if :redelta_binning in config.experiments
            S.compute_redelta_binning(case, dns, Δf; force = :redelta_binning in config.force)
            S.plot_redelta_binning(case, dns, Δf)
        end

        :seeds in config.experiments &&
            S.get_seed_statistics(case, fams, dns, Δf, config.netseeds; force = :seeds in config.force)

        if :plots in config.experiments
            S.plot_apriori_bar(case, dns, Δf, models)
            S.plot_dissipation_bar(case, dns, Δf, withref)
            S.plot_backscatter_bar(case, dns, Δf, withref)
            S.plot_densities(case, dns, Δf, [:ref; filter(!=(:nomo), models)])
            S.plot_error_post(case, dns, Δf, models)
            S.plot_budget(case, dns, Δf, withref)
            S.plot_spectral_transfer(case, dns, Δf, withref)
            S.plot_spectrum_les(case, dns, Δf, withref)
            S.plot_equivariance_errors(case, dns, Δf, models)
        end
    end

    #######################
    # Cross-eval-point: the Re_Δ trend figure + paper table
    #######################
    if :trend in config.experiments
        trainpoints = [(dns, Δf) for dns in S.dns_runs().train for Δf in case.filters_train]
        S.plot_trend_vs_redelta(
            case, testpoints, fams;
            netseeds = config.netseeds,
            classical = Tuple(filter(!=(:nomo), config.classical)),  # :nomo diss-ratio = 0 (log axis)
            trainpoints,
        )
    end

    if :tables in config.experiments
        dns, Δf = first(testpoints)
        S.write_errors_table(case, dns, Δf, models; netseeds = config.netseeds)
    end

    #######################
    # Optional showcase: full-field rollout montages at one eval point
    #######################
    if :showcase in config.experiments
        dns, Δf = first(testpoints)
        setup = S.make_setup(case, dns, Δf)
        show_models = [
            :clar,
            (; arch = config.archs[1], tier = config.tiers[1], netseed = first(config.netseeds), use_redelta = false),
        ]
        for m in show_models
            S.clean()
            getmodel() = buildone(case, setup, m)
            S.solve_les(case, m, dns, Δf, getmodel; force = true, savefields = true)
        end
        S.plot_velocities(case, dns, Δf, [:ref; show_models], :vortz)
        S.plot_sfs(case, dns, Δf, show_models)
    end

    @info "Done."
    flush(stderr)
    return
end

main()
