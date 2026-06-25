# Train the LES closures and run the three-part experiment suite. Coordinate-driven:
# the model coordinates are assembled into a few *purposeful* lists (never one
# Cartesian product), each broad along a single axis:
#
#   A  Saturation   — +Re, every architecture across its size grid, one eval point.
#                     → error vs parameter count: all archs saturate to the same
#                       floor, inductive bias gets there at fewer params.
#   B  Equal-cap.   — top tier ±Re, the inductive-bias / equivariance comparison
#                       at matched capacity (bars, equivariance, errors table).
#   C  Re_Δ grid    — top tier ±Re across the full (ν, Δ) test grid → the Re_Δ
#                       generalization trend.
#
# A is broad in *size* / narrow in *eval*; C is broad in *eval* / narrow in *size*;
# B reads A/C's artifacts at the in-distribution point. Artifacts self-locate from
# coordinates, so adding a size or a seed later recomputes only the gaps.

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
    archs = (:conv, :equi, :tbnn),

    # The capacity grid (names = `case.tiers` keys). `sizes` is shared by all three
    # archs; `sizes_extra` extends individual archs (conv runs higher — the others
    # OOM the equivariant rollout past ~8k and saturate well before).
    sizes = (:p120, :p400, :p1200, :p3000, :p8000),
    sizes_extra = (; conv = (:p16000,)),
    top = :p8000,                # matched top tier for the B / C comparisons

    # Seeds: more for the cheap saturation curve (one eval point), fewer for the
    # expensive top-tier grid (full ν × Δ).
    netseeds_curve = 0:2,
    netseeds_grid = 0:1,

    classical = [:nomo, :clar],
    train = true,

    experiments = [
        :apriori,          # compute_sfs_stats (reduce-on-the-fly a-priori)
        :aposteriori,      # solve_les (reduce-on-the-fly)
        :equivariance,     # apriori_equivariance_error (at the equal-capacity point)
        :redelta_binning,  # Phase-0 pointwise-Re_Δ diagnostic (per grid point)
        :saturation,       # plot_saturation (A — the headline figure)
        :seeds,            # get_seed_statistics (feeds the B bars + tables)
        :plots,            # training curve, B bars + per-curve series
        :trend,            # plot_trend_vs_redelta (C)
        :tables,           # errors + timing tables
    ],
    force = Set{Symbol}([]),
)

# Per-arch size points: the shared grid plus any per-arch extension.
arch_sizes(c, arch) = (c.sizes..., get(c.sizes_extra, arch, ())...)

# A — saturation: +Re, every architecture across its full size range.
saturation_models(c) = [
    (; arch, tier, netseed, use_redelta = true)
        for arch in c.archs for tier in arch_sizes(c, arch) for netseed in c.netseeds_curve
]
saturation_families(c) = [
    (; arch, tier, use_redelta = true) for arch in c.archs for tier in arch_sizes(c, arch)
]

# B / C — top tier, both Re_Δ settings (equal-capacity comparison + Re_Δ ablation).
ablation_models(c) = [
    (; arch, tier = c.top, netseed, use_redelta = ur)
        for arch in c.archs for ur in (false, true) for netseed in c.netseeds_grid
]
ablation_families(c) = [
    (; arch, tier = c.top, use_redelta = ur) for arch in c.archs for ur in (false, true)
]

# Curated one-seed subset for the per-curve series plots (+Re top + classical).
series_models(c) = [
    c.classical;
    [(; arch, tier = c.top, netseed = first(c.netseeds_grid), use_redelta = true) for arch in c.archs]
]

# Every distinct model to train (the +Re top is shared between A and C).
all_models(c) = unique([saturation_models(c); ablation_models(c)])

# Build a single closure (classical symbol or learned coordinate) against `setup`.
buildone(case, setup, m) = S.build_models(case, setup, [m])[S.modelname(m)]

# Per-script table file (shares case.plotdir with create-data.jl).
tablefile() = "tables-les.txt"
reset_tables(case; kwargs...) = S.reset_tables(case; filename = tablefile(), kwargs...)
tabulate(args...; kwargs...) = S.tabulate(args...; filename = tablefile(), kwargs...)

"""
Evaluate `models` at eval point (dns, Δf): the reference once, then each model
(lazy build, GPU reclaimed between models). `equi = true` also computes the
a-priori equivariance error (only worthwhile where the equal-capacity figures read
it — the in-distribution point).
"""
function evaluate!(case, config, models, dns, Δf; equi = false)
    setup = S.make_setup(case, dns, Δf)
    :apriori in config.experiments &&
        S.compute_sfs_stats(case, :ref, dns, Δf; force = :apriori in config.force)
    :aposteriori in config.experiments &&
        S.solve_les(case, :ref, dns, Δf; force = :aposteriori in config.force)
    for m in models
        S.clean()
        local built = nothing
        getmodel() = (built === nothing && (built = buildone(case, setup, m)); built)
        :apriori in config.experiments &&
            S.compute_sfs_stats(case, m, dns, Δf, getmodel; force = :apriori in config.force)
        :aposteriori in config.experiments &&
            S.solve_les(case, m, dns, Δf, getmodel; force = :aposteriori in config.force)
        equi && :equivariance in config.experiments && m isa NamedTuple &&
            S.apriori_equivariance_error(case, m, dns, Δf, getmodel; force = :equivariance in config.force)
        built = nothing
        S.clean()
    end
    return
end

function main()
    case = get_case()
    config = get_config()
    reset_tables(case)
    trainpool = S.build_trainpool(case)

    indist = first(S.dns_runs().test)        # in-distribution test run (A/B eval point)
    Δ_ab = first(case.filters_test)          # representative filter for A/B
    gridpoints = [(dns, Δf) for dns in S.dns_runs().test for Δf in case.filters_test]

    #######################
    # Train every distinct model once over the trainpool (explicit list, no product).
    #######################
    if config.train
        for m in all_models(config)
            S.train_model(case, m, trainpool; force = :train in config.force)
            S.clean()
        end
        timings = NamedTuple(
            S.modelkey(m) => (isfile(S.psfile(case, m)) ? load(S.psfile(case, m), "timing") : :missing)
                for m in all_models(config)
        )
        tabulate(case, "Training wall-time (s) per coordinate", timings; digits = 1)
    end
    :plots in config.experiments && S.plot_training(case, all_models(config))

    #######################
    # A + B: every model at the in-distribution point (saturation sizes + top ±Re),
    # with equivariance — the saturation curve and the equal-capacity figures.
    #######################
    @info "===== A/B: in-distribution point (visc=$(indist.visc), Δ=$(Δ_ab)) ====="
    flush(stderr)
    evaluate!(case, config, [config.classical; all_models(config)], indist, Δ_ab; equi = true)

    #######################
    # C: top tier ±Re across the full (ν, Δ) test grid — the Re_Δ generalization trend.
    #######################
    for (dns, Δf) in gridpoints
        @info "===== C: grid point (role=$(dns.role), visc=$(dns.visc), Δ=$(Δf)) ====="
        flush(stderr)
        evaluate!(case, config, [config.classical; ablation_models(config)], dns, Δf)
        if :redelta_binning in config.experiments
            S.compute_redelta_binning(case, dns, Δf; force = :redelta_binning in config.force)
            S.plot_redelta_binning(case, dns, Δf)
        end
    end

    #######################
    # Figures + tables
    #######################
    :saturation in config.experiments &&
        S.plot_saturation(case, indist, Δ_ab, saturation_families(config), config.netseeds_curve)

    :seeds in config.experiments &&
        S.get_seed_statistics(case, ablation_families(config), indist, Δ_ab, config.netseeds_grid; force = :seeds in config.force)

    if :plots in config.experiments
        fams = ablation_families(config)
        S.plot_apriori_bar(case, indist, Δ_ab, fams, config.netseeds_grid; classical = config.classical)
        S.plot_dissipation_bar(case, indist, Δ_ab, fams, config.netseeds_grid; classical = config.classical)
        S.plot_backscatter_bar(case, indist, Δ_ab, fams, config.netseeds_grid; classical = config.classical)
        S.plot_equivariance_bar(case, indist, Δ_ab, fams, config.netseeds_grid)
        series = series_models(config)
        S.plot_densities(case, indist, Δ_ab, [:ref; series])
        S.plot_error_post(case, indist, Δ_ab, series)
        S.plot_budget(case, indist, Δ_ab, [:ref; series])
        S.plot_spectral_transfer(case, indist, Δ_ab, [:ref; series])
        S.plot_spectrum_les(case, indist, Δ_ab, [:ref; series])
    end

    if :trend in config.experiments
        trainpoints = [(dns, Δf) for dns in S.dns_runs().train for Δf in case.filters_train]
        S.plot_trend_vs_redelta(
            case, gridpoints, ablation_families(config);
            netseeds = config.netseeds_grid,
            classical = Tuple(filter(!=(:nomo), config.classical)),  # :nomo diss-ratio = 0 (log axis)
            trainpoints,
        )
    end

    if :tables in config.experiments
        S.write_errors_table(case, indist, Δ_ab, ablation_families(config), config.netseeds_grid; classical = config.classical)
        S.write_timing_table(
            case, indist, Δ_ab,
            unique([saturation_families(config); ablation_families(config)]),
            config.netseeds_curve; classical = config.classical,
        )
    end

    @info "Done."
    flush(stderr)
    return
end

main()
