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
# Phases (`julia run-les.jl <phase>`, default `all` = the whole pipeline serially):
#   models   per-model train+eval, distributed over SLURM_ARRAY_TASK_ID (one unit
#            per `les_worklist` entry). Partitioning by *model* gives every psfile
#            exactly one writer, so concurrent trainings never race.
#   convsym  evaluate the symmetrized MLP (reuses :conv's psfile) — a second array,
#            run *after* `models` so the conv params exist (afterok in submit.sh).
#   reduce   serial: model-independent references, Re_Δ binning, every cross-model
#            figure/table (reads what the two array phases wrote).
#   pending  print `models=<--array spec>` and `convsym=<--array spec>` for the
#            units whose artifacts are incomplete, so `submit.sh` submits only the
#            gaps (a cached unit otherwise costs a full GPU-node Julia load just
#            to quit).
# `submit.sh` chains models → convsym → reduce with SLURM `afterok` dependencies,
# so the sweep is one submission instead of hand-sequenced reruns. Artifacts
# self-locate from coordinates, so adding a size/seed recomputes only the gaps.

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
    # sizes_extra = (; conv = (:p16000,)),
    sizes = (:p120, :p400, :p1200, :p3000),
    sizes_extra = (;),
    top = :p1200,                # matched top tier for the B / C comparisons

    # Seeds: one set shared by the saturation curve and the top-tier grid (network
    # init + batch shuffling only; the data is unaffected).
    netseeds = 0:4,

    classical = [:nomo, :dynsmag, :clar],
    train = true,

    experiments = [
        :apriori,          # compute_sfs_stats (reduce-on-the-fly a-priori)
        :aposteriori,      # solve_les (reduce-on-the-fly)
        :equivariance,     # apriori_equivariance_error (at the equal-capacity point)
        :redelta_binning,  # Phase-0 pointwise-Re_Δ diagnostic (per grid point)
        :condmean,         # direct conditional-mean estimate (the a-priori floor)
        :saturation,       # plot_saturation (A — the headline figure)
        :seeds,            # get_seed_statistics (feeds the B bars + tables)
        :plots,            # training curve, B bars + per-curve series
        :trend,            # plot_trend_vs_redelta (C)
        :tables,           # errors + timing tables
    ],
    force = Set{Symbol}(
        [
            # :seeds,
        ]
    ),
)

# Per-arch size points: the shared grid plus any per-arch extension.
arch_sizes(c, arch) = (c.sizes..., get(c.sizes_extra, arch, ())...)

# A — saturation: +Re, every architecture across its full size range.
saturation_models(c) = [
    (; arch, tier, netseed, use_redelta = true)
        for arch in c.archs for tier in arch_sizes(c, arch) for netseed in c.netseeds
]
saturation_families(c) = [
    (; arch, tier, use_redelta = true) for arch in c.archs for tier in arch_sizes(c, arch)
]

# B / C — top tier, both Re_Δ settings (equal-capacity comparison + Re_Δ ablation).
ablation_models(c) = [
    (; arch, tier = c.top, netseed, use_redelta = ur)
        for arch in c.archs for ur in (false, true) for netseed in c.netseeds
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
    [(; arch, tier = c.top, netseed = first(c.netseeds), use_redelta = true) for arch in c.archs]
]

# Every distinct model to train (the +Re top is shared between A and C).
all_models(c) = unique([saturation_models(c); ablation_models(c)])

# Build a single closure (classical symbol or learned coordinate) against `setup`.
buildone(case, setup, m) = S.build_models(case, setup, [m])[S.modelname(m)]

# Per-script table file (shares case.plotdir with run-dns.jl).
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

# Eval points + the distributed worklist; the phase entry points below read these.
eval_indist(case) = (first(S.dns_runs().test), first(case.filters_test))  # the A/B point
eval_grid(case) = [(dns, Δf) for dns in S.dns_runs().test for Δf in case.filters_test]

# Conditional-mean estimate (ExperimentFollowups.md item 1): the in-distribution
# A/B point (where the shared a-priori floor is quoted) plus the high-Re OOD run
# at the same filter — needs the heavy trainpool + test `fieldsfile`s on disk.
condmean_points(case) = [(dns, first(case.filters_test)) for dns in S.dns_runs().test]
# Floor references: the top tier ±Re (evaluated on the whole C grid, so present
# at both points). The no-Re nets see exactly the estimator's invariant feature
# space — the apples-to-apples floor; the +Re pair shows what the extra Re_Δ
# input buys beyond it. Families without artifacts are skipped by the plot.
condmean_families(c) =
    [(; arch, tier = c.top, use_redelta = ur) for arch in c.archs for ur in (false, true)]
les_worklist(c) = [c.classical; all_models(c)]

"""
Eval points for `models`-phase unit `m`: the in-distribution A/B point for all,
plus the rest of the C grid for the classical baselines and the ±Re ablation set.
Shared by `run_models!` (what to run) and `unit_pending` (what to check), so the
two cannot drift.
"""
function unit_points(case, config, m)
    indist, Δ_ab = eval_indist(case)
    points = [(indist, Δ_ab)]
    m in [config.classical; ablation_models(config)] &&
        append!(points, [p for p in eval_grid(case) if p != (indist, Δ_ab)])
    return points
end

"""
Does `eval_model!` still have work for `m` at (dns, Δf)? Mirrors its cache
guards (the per-experiment artifact files), honoring `config.experiments` and
`config.force` — keep the two in sync.
"""
function eval_pending(case, config, m, dns, Δf; equi = false)
    :apriori in config.experiments &&
        (:apriori in config.force || !isfile(S.sfsstatsfile(case, dns, Δf, m))) &&
        return true
    :aposteriori in config.experiments &&
        (:aposteriori in config.force || !isfile(S.apostfile(case, dns, Δf, m))) &&
        return true
    equi && :equivariance in config.experiments && m isa NamedTuple &&
        (:equivariance in config.force || !isfile(S.equipriorfile(case, dns, Δf, m))) &&
        return true
    return false
end

"Does `models`-phase unit `m` still have work (training or any of its eval points)?"
function unit_pending(case, config, m)
    if config.train && m isa NamedTuple
        (:train in config.force || !isfile(S.psfile(case, m))) && return true
    end
    indist_pt = eval_indist(case)
    return any(
        pt -> eval_pending(case, config, m, pt...; equi = pt == indist_pt),
        unit_points(case, config, m),
    )
end

function print_pending(case, config)
    indist, Δ_ab = eval_indist(case)
    println(
        "models=",
        S.slurm_array_spec(findall(m -> unit_pending(case, config, m), les_worklist(config))),
    )
    println(
        "convsym=",
        S.slurm_array_spec(
            findall(
                m -> eval_pending(case, config, m, indist, Δ_ab; equi = true),
                convsym_models(config),
            ),
        ),
    )
    return
end

"""
Phase `models` (array over `les_worklist`): train each learned closure (cache-
guarded), then evaluate it at every point it appears in — the in-distribution A/B
point for all, plus the full C grid for the classical baselines and the ±Re
ablation set. `task_id === nothing` runs the whole list; an integer runs only that
1-based unit. Equivariance is computed at the A/B point only.
"""
function run_models!(case, config, task_id)
    indist, Δ_ab = eval_indist(case)
    work = les_worklist(config)

    # Trainpool load is heavy; build it lazily so eval-only / already-trained units
    # never pay for it.
    local trainpool = nothing
    gettrainpool() = (isnothing(trainpool) && (trainpool = S.build_trainpool(case)); trainpool)

    for (i, m) in enumerate(work)
        isnothing(task_id) || i == task_id || continue
        @info "===== model $i/$(length(work)): $(S.modelname(m)) ====="
        flush(stderr)
        if config.train && m isa NamedTuple && (:train in config.force || !isfile(S.psfile(case, m)))
            S.train_model(case, m, gettrainpool(); force = :train in config.force)
        end
        S.clean()
        for (dns, Δf) in unit_points(case, config, m)
            eval_model!(case, config, m, dns, Δf; equi = (dns, Δf) == (indist, Δ_ab))
        end
    end
    return
end

"""
Phase `convsym` (array over `convsym_models`): evaluate the symmetrized MLP at the
in-distribution point. It reuses :conv's psfile, so it must run *after* the
`models` phase trained those — the submit DAG enforces the order with an `afterok`
dependency; a serial/local run already has them on disk.
"""
function run_convsym!(case, config, task_id)
    indist, Δ_ab = eval_indist(case)
    models = convsym_models(config)
    for (i, m) in enumerate(models)
        isnothing(task_id) || i == task_id || continue
        @info "===== convsym $i/$(length(models)): $(S.modelname(m)) ====="
        flush(stderr)
        eval_model!(case, config, m, indist, Δ_ab; equi = true)
    end
    return
end

"""
Phase `reduce` (serial): the model-independent references at every eval point, the
Phase-0 Re_Δ binning, and every cross-model figure/table. Reads the per-model
artifacts the `models` / `convsym` phases wrote, so it runs last (`afterok`).
"""
function run_reduce!(case, config)
    indist, Δ_ab = eval_indist(case)
    grid = eval_grid(case)
    reset_tables(case)

    for (dns, Δf) in unique([(indist, Δ_ab); grid])
        eval_ref!(case, config, dns, Δf)
    end

    if :redelta_binning in config.experiments
        for (dns, Δf) in grid
            S.compute_redelta_binning(case, dns, Δf; force = :redelta_binning in config.force)
            S.plot_redelta_binning(case, dns, Δf)
        end
    end

    if :condmean in config.experiments
        for (dns, Δf) in condmean_points(case)
            S.compute_condmean(case, dns, Δf; force = :condmean in config.force)
        end
        S.plot_condmean(case, condmean_points(case), condmean_families(config), config.netseeds)
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
        case, indist, Δ_ab, [saturation_families(config); convsym_curve(config)], config.netseeds;
        classical = config.classical,
    )

    :seeds in config.experiments &&
        S.get_seed_statistics(case, comparison_families(config), indist, Δ_ab, config.netseeds; force = :seeds in config.force)

    if :plots in config.experiments
        fams = comparison_families(config)
        S.plot_apriori_bar(case, indist, Δ_ab, fams, config.netseeds; classical = config.classical)
        S.plot_dissipation_bar(case, indist, Δ_ab, fams, config.netseeds; classical = config.classical)
        S.plot_backscatter_bar(case, indist, Δ_ab, fams, config.netseeds; classical = config.classical)
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
            case, grid, ablation_families(config);
            netseeds = config.netseeds,
            classical = Tuple(filter(!=(:nomo), config.classical)),  # :nomo diss-ratio = 0 (log axis)
            trainpoints,
        )
    end

    if :tables in config.experiments
        S.write_errors_table(case, indist, Δ_ab, comparison_families(config), config.netseeds; classical = config.classical, include_tier = false, include_crosscor = false)
        S.write_timing_table(
            case, indist, Δ_ab,
            unique([saturation_families(config); ablation_families(config)]),
            config.netseeds; classical = config.classical,
        )
    end

    @info "Done (reduce)."
    flush(stderr)
    return
end

"""
Entry point. `julia run-les.jl [phase]`, `phase ∈ all|models|convsym|reduce|pending`
(default `all` = the full pipeline, serial). The cluster submits `models` and
`convsym` as SLURM arrays (one unit per `SLURM_ARRAY_TASK_ID`) then `reduce`
serially, chained by `afterok` (see `submit.sh`); `pending` prints the two
`--array` specs of the units with missing artifacts (GPU-free — `submit.sh` runs
it on the login node). `all` ignores the array env so a stray
`SLURM_ARRAY_TASK_ID` can't shard it.
"""
function (@main)(args)
    case = get_case()
    config = get_config()
    phase = isempty(args) ? "all" : first(args)
    task_id = let s = get(ENV, "SLURM_ARRAY_TASK_ID", nothing)
        isnothing(s) ? nothing : parse(Int, s)
    end

    phase in ("all", "models", "convsym", "reduce", "pending") ||
        error("unknown phase '$(phase)'; expected all|models|convsym|reduce|pending")

    if phase == "pending"
        print_pending(case, config)
    elseif phase == "all"
        run_models!(case, config, nothing)
        run_convsym!(case, config, nothing)
        run_reduce!(case, config)
    elseif phase == "models"
        run_models!(case, config, task_id)
    elseif phase == "convsym"
        run_convsym!(case, config, task_id)
    elseif phase == "reduce"
        run_reduce!(case, config)
    end
    return 0
end
