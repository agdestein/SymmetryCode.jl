# Train the LES closures and run the three-part experiment suite. Coordinate-driven:
# the model coordinates are assembled into a few *purposeful* lists (never one
# Cartesian product), each broad along a single axis:
#
#   A  Saturation   — +Re, every architecture across its size grid, one eval point.
#                     → error vs parameter count: all archs saturate to the same
#                       floor, inductive bias gets there at fewer params. The
#                       symmetrized MLP (:convsym) is overlaid — does post-hoc
#                       symmetry recover the equivariant nets' low-param efficiency?
#   B  Equal-cap.   — top tier ±Re (plus :convsym), the inductive-bias /
#                       equivariance comparison at matched capacity (bars,
#                       equivariance, errors table).
#   C  Re_Δ grid    — top tier ±Re across the full (ν, Δ) test grid → the Re_Δ
#                       generalization trend (no :convsym — skipped on the grid).
#
# A is broad in *size* / narrow in *eval*; C is broad in *eval* / narrow in *size*;
# B reads A/C's artifacts at the in-distribution point. Artifacts self-locate from
# coordinates, so adding a size or a seed later recomputes only the gaps.
#
# SLURM array distribution (mirrors create-data.jl). The per-model work — train the
# closure, then evaluate it at every point it appears in — is the distributable
# unit. With SLURM_ARRAY_TASK_ID=i set this process handles only `worklist[i]`;
# unset, it runs the whole list serially. Partitioning by *model* (not by eval
# point) gives every psfile exactly one writer, so trainings never race. The
# model-independent references, the Phase-0 binning, and every cross-model
# figure/table run only in the serial pass — so after an array run, rerun once
# without the array env to aggregate (the per-model loop is then all cache hits).
# Submit with `--array=1-N`, N = the worklist length logged at startup.

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
    # sizes = (:p120, :p400, :p1200, :p3000, :p8000),
    sizes = (:p120, :p400, :p1200),
    # sizes_extra = (; conv = (:p16000,)),
    sizes_extra = (; conv = (:p3000,)),
    # top = :p8000,                # matched top tier for the B / C comparisons
    top = :p1200,                # matched top tier for the B / C comparisons

    # Seeds: more for the cheap saturation curve (one eval point), fewer for the
    # expensive top-tier grid (full ν × Δ).
    netseeds_curve = 0:3,
    netseeds_grid = 0:1,

    classical = [:nomo, :dynsmag, :clar],
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

# :convsym — the symmetrized MLP: :conv's trained params group-averaged over the
# cube symmetry at inference (build_model loads the matching :conv psfile, so it is
# *not* trained). It joins the saturation curve (A, +Re across the conv sizes) and
# the equal-capacity comparison (B, top tier ±Re), but is excluded from the C grid
# — evaluated at the in-distribution point only, reusing every conv coordinate that
# `all_models` already trains (so `convsym_models` never lacks a psfile).
convsym_curve(c) = [(; arch = :convsym, tier, use_redelta = true) for tier in arch_sizes(c, :conv)]
convsym_top(c) = [(; arch = :convsym, tier = c.top, use_redelta = ur) for ur in (false, true)]
convsym_models(c) = [(; m..., arch = :convsym) for m in all_models(c) if m.arch === :conv]

# B — equal-capacity comparison: the ±Re ablation set plus the symmetrized MLP at
# the top tier (conv's equivariance twin at matched capacity). Drives the
# in-distribution bars/tables only; the C trend stays on `ablation_families`.
comparison_families(c) = [ablation_families(c); convsym_top(c)]

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
Compute the model-independent reference artifacts at eval point (dns, Δf): the
a-priori SFS reference and the no-closure a-posteriori rollout. Shared by every
figure, so it runs in the serial aggregation pass (not per array task).
"""
function eval_ref!(case, config, dns, Δf)
    :apriori in config.experiments &&
        S.compute_sfs_stats(case, :ref, dns, Δf; force = :apriori in config.force)
    :aposteriori in config.experiments &&
        S.solve_les(case, :ref, dns, Δf; force = :aposteriori in config.force)
    return
end

"""
Evaluate a single closure `m` at eval point (dns, Δf): a-priori SFS stats and the
a-posteriori rollout (lazy build, GPU reclaimed around it). `equi = true` also
computes the a-priori equivariance error (only worthwhile at the equal-capacity
point, where the equivariance figure reads it).
"""
function eval_model!(case, config, m, dns, Δf; equi = false)
    setup = S.make_setup(case, dns, Δf)
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
    return
end

function main()
    case = get_case()
    config = get_config()

    slurm_id = get(ENV, "SLURM_ARRAY_TASK_ID", nothing)
    task_id = isnothing(slurm_id) ? nothing : parse(Int, slurm_id)
    serial = isnothing(task_id)

    serial && reset_tables(case)

    # Trainpool load is heavy; build it lazily so eval-only / already-trained tasks
    # and the serial aggregation rerun (all cache hits) never pay for it.
    local trainpool = nothing
    gettrainpool() = (isnothing(trainpool) && (trainpool = S.build_trainpool(case)); trainpool)

    indist = first(S.dns_runs().test)        # in-distribution test run (A/B eval point)
    Δ_ab = first(case.filters_test)          # representative filter for A/B
    gridpoints = [(dns, Δf) for dns in S.dns_runs().test for Δf in case.filters_test]

    # The distributable unit: one closure handled end-to-end. Everyone is evaluated
    # at the A/B in-distribution point; the classical baselines and the ±Re ablation
    # models additionally span the full C grid (the A/B point is itself one of the
    # grid points, hence the dedup below).
    worklist = [config.classical; all_models(config)]
    cgrid_closures = [config.classical; ablation_models(config)]
    @info "LES worklist: $(length(worklist)) units (SLURM array range 1-$(length(worklist)))"
    flush(stderr)

    #######################
    # Per-model work (distributed): train (learned only), then evaluate at every
    # point the closure appears in. A/B is the only point that gets equivariance.
    #######################
    for (i, m) in enumerate(worklist)
        serial || i == task_id || continue
        @info "===== unit $i/$(length(worklist)): $(S.modelname(m)) ====="
        flush(stderr)

        if config.train && m isa NamedTuple && (:train in config.force || !isfile(S.psfile(case, m)))
            S.train_model(case, m, gettrainpool(); force = :train in config.force)
        end
        S.clean()

        points = [(indist, Δ_ab)]
        m in cgrid_closures && append!(points, [p for p in gridpoints if p != (indist, Δ_ab)])
        for (dns, Δf) in points
            eval_model!(case, config, m, dns, Δf; equi = (dns, Δf) == (indist, Δ_ab))
        end
    end

    # An array task is done after its slice; everything below is the serial pass.
    if !serial
        @info "Done (array task $(task_id))."
        flush(stderr)
        return
    end

    #######################
    # Serial aggregation. Model-independent references at every eval point + the
    # Phase-0 Re_Δ binning per grid point, then the training timing/curves and every
    # cross-model figure/table. After an array run, rerun once without the array env:
    # the per-model loop above is all cache hits, so only this section does work.
    #######################
    for (dns, Δf) in unique([(indist, Δ_ab); gridpoints])
        eval_ref!(case, config, dns, Δf)
    end

    # :convsym is eval-only (reuses :conv's psfile) and in-distribution only, so it
    # runs here in the serial pass — once the conv models it wraps are trained —
    # rather than as its own distributed unit (which would race that training).
    for m in convsym_models(config)
        eval_model!(case, config, m, indist, Δ_ab; equi = true)
    end

    if :redelta_binning in config.experiments
        for (dns, Δf) in gridpoints
            S.compute_redelta_binning(case, dns, Δf; force = :redelta_binning in config.force)
            S.plot_redelta_binning(case, dns, Δf)
        end
    end

    if config.train
        timings = NamedTuple(
            S.modelkey(m) => (isfile(S.psfile(case, m)) ? load(S.psfile(case, m), "timing") : :missing)
                for m in all_models(config)
        )
        tabulate(case, "Training wall-time (s) per coordinate", timings; digits = 1)
    end
    :plots in config.experiments && S.plot_training(case, all_models(config))

    :saturation in config.experiments &&
        S.plot_saturation(
        case, indist, Δ_ab, [saturation_families(config); convsym_curve(config)], config.netseeds_curve;
        classical = config.classical,
    )

    :seeds in config.experiments &&
        S.get_seed_statistics(case, comparison_families(config), indist, Δ_ab, config.netseeds_grid; force = :seeds in config.force)

    if :plots in config.experiments
        fams = comparison_families(config)
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
        S.write_errors_table(case, indist, Δ_ab, comparison_families(config), config.netseeds_grid; classical = config.classical)
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
