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
#
# Phases (`julia run-tgv.jl <phase>`, default `all` = serial end-to-end):
#   data     generate one TGV run's data per SLURM_ARRAY_TASK_ID (a high-res DNS
#            time march). Each run is independent, so the array splits over runs.
#   models   evaluate one closure at every (tgv, Δf) point — a second array over
#            `eval_models`. Reuses run-les.jl's psfiles; assumes `data` has run.
#   reduce   serial: model-independent references + the per-eval-point figures.
#   count    print the `data` and `models` array sizes for submit.sh.
# `submit.sh` chains data → models → reduce with SLURM `afterok` dependencies.

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
    # Must match the trained set in run-les.jl (these reuse those ps-*.jld2): the
    # top tier ±Re — sweep D applies sweep C's models to the decaying TGV.
    archs = (:conv, :equi, :tbnn),
    top = :p1200,
    netseeds = 0:1,
    classical = [:nomo, :dynsmag, :clar],

    experiments = [
        :data,            # create_data_tgv -> dnsmetafile + fieldsfile/lesmeta per Δ
        :apriori,         # compute_sfs_stats (reduce-on-the-fly a-priori)
        :aposteriori,     # solve_les (decaying rollout, reduce-on-the-fly)
        :seeds,           # get_seed_statistics (netseed aggregate -> seedstatsfile; feeds the bars)
        :plots,           # per-eval-point figures
        :dissipation,     # plot_dissipation_tgv (the benchmark)
        :vorticity,       # plot_vorticity_tgv montage (full-DNS z-vorticity, Δ-independent)
    ],
    force = Set{Symbol}(
        [
            # :data,            # create_data_tgv -> dnsmetafile + fieldsfile/lesmeta per Δ
            # :apriori,         # compute_sfs_stats (reduce-on-the-fly a-priori)
            # :aposteriori,     # solve_les (decaying rollout, reduce-on-the-fly)
            # :seeds,           # get_seed_statistics (netseed aggregate -> seedstatsfile)
            # :plots,           # per-eval-point figures
            # :dissipation,     # plot_dissipation_tgv (the benchmark)
            # :vorticity,       # plot_vorticity_tgv montage (full-DNS z-vorticity, Δ-independent)
        ]
    ),
)

# Top tier ±Re (mirrors run-les.jl's ablation set, which trained these ps-*.jld2).
eval_models(c) = [
    c.classical;
    [
        (; arch, tier = c.top, netseed, use_redelta = ur)
            for arch in c.archs for ur in (false, true) for netseed in c.netseeds
    ]
]
families(c) = [
    (; arch, tier = c.top, use_redelta = ur) for arch in c.archs for ur in (false, true)
]
series_models(c) = [
    c.classical;
    [
        (; arch, tier = c.top, netseed = first(c.netseeds), use_redelta = ur)
            for arch in c.archs for ur in (false, true)
    ]
]
buildone(case, setup, m) = S.build_models(case, setup, [m])[S.modelname(m)]

"Model-independent reference artifacts at eval point (tgv, Δf): a-priori stats + the no-closure rollout."
function eval_ref!(case, config, dns, Δf)
    :apriori in config.experiments &&
        S.compute_sfs_stats(case, :ref, dns, Δf; force = :apriori in config.force)
    :aposteriori in config.experiments &&
        S.solve_les(case, :ref, dns, Δf; force = :aposteriori in config.force)
    return
end

"Evaluate a single closure `m` at (tgv, Δf): a-priori stats + the decaying rollout (lazy build, GPU reclaimed)."
function eval_model!(case, config, m, dns, Δf)
    setup = S.make_setup(case, dns, Δf)
    S.clean()
    local built = nothing
    getmodel() = (built === nothing && (built = buildone(case, setup, m)); built)
    :apriori in config.experiments &&
        S.compute_sfs_stats(case, m, dns, Δf, getmodel; force = :apriori in config.force)
    :aposteriori in config.experiments &&
        S.solve_les(case, m, dns, Δf, getmodel; force = :aposteriori in config.force)
    built = nothing
    S.clean()
    return
end

tgv_points(case) = [(tgv, Δf) for tgv in S.tgv_runs() for Δf in case.filters_test]
print_counts(config) = println(length(S.tgv_runs()), " ", length(eval_models(config)))

"""
Phase `data` (array over `tgv_runs()`): generate one TGV run's (ūbar, τ) data
(`create_data_tgv`) — a single high-res DNS time march. Each run is independent.
`task_id === nothing` does every run; an integer does only that 1-based run.
"""
function run_data!(case, config, task_id)
    :data in config.experiments || return
    for (i, tgv) in enumerate(S.tgv_runs())
        isnothing(task_id) || i == task_id || continue
        @info "===== TGV data $i: Re=$(tgv.Re_target), visc=$(tgv.visc) ====="
        flush(stderr)
        S.create_data_tgv(case, tgv; force = :data in config.force)
    end
    return
end

"""
Phase `models` (array over `eval_models`): evaluate one closure at every (tgv, Δf)
point — a-priori stats + the decaying rollout. Closures are reused verbatim from
run-les.jl's psfiles (no training); assumes the `data` phase has run.
"""
function run_models!(case, config, task_id)
    models = eval_models(config)
    points = tgv_points(case)
    for (i, m) in enumerate(models)
        isnothing(task_id) || i == task_id || continue
        @info "===== model $i/$(length(models)): $(S.modelname(m)) ====="
        flush(stderr)
        for (tgv, Δf) in points
            eval_model!(case, config, m, tgv, Δf)
        end
    end
    return
end

"""
Phase `reduce` (serial): the model-independent references, the netseed aggregate
(`seedstatsfile`, which the bars consume — overwrite it with `:seeds` in
`config.force`), then the per-eval-point cross-model figures (bars, error, spectra,
the dissipation benchmark, the field montage). Reads what the `models` phase wrote.
"""
function run_reduce!(case, config)
    fams = families(config)
    series = series_models(config)
    for (tgv, Δf) in tgv_points(case)
        eval_ref!(case, config, tgv, Δf)
        :seeds in config.experiments &&
            S.get_seed_statistics(case, fams, tgv, Δf, config.netseeds; force = :seeds in config.force)
        if :plots in config.experiments
            S.plot_apriori_bar(case, tgv, Δf, fams, config.netseeds; classical = config.classical)
            S.plot_dissipation_bar(case, tgv, Δf, fams, config.netseeds; classical = config.classical)
            S.plot_error_post(case, tgv, Δf, series)
            S.plot_spectrum_les(case, tgv, Δf, [:ref; series])
            # Per-filter TGV error table. Distinct filename so it never clobbers the
            # forced-LES `errors.tex` written by run-les.jl into the same plotdir.
            S.write_errors_table(
                case, tgv, Δf, fams, config.netseeds;
                classical = config.classical, include_equi = false,
                filename = "errors-tgv-delta=$(Δf).tex",
            )
        end
        :dissipation in config.experiments &&
            S.plot_dissipation_tgv(case, tgv, Δf, [:ref; series])
    end
    # Δ-independent (one per TGV run): the full-DNS z-vorticity montage + animation.
    if :vorticity in config.experiments
        for tgv in S.tgv_runs()
            S.plot_vorticity_tgv(case, tgv)
            S.animate_vorticity_tgv(case, tgv)
        end
    end
    @info "Done (reduce)."
    flush(stderr)
    return
end

"""
Entry point. `julia run-tgv.jl [phase]`, `phase ∈ all|data|models|reduce|count`
(default `all` = serial end-to-end). The cluster submits `data` and `models` as
SLURM arrays then `reduce` serially, chained by `afterok` (see `submit.sh`);
`count` prints the two array sizes.
"""
function (@main)(args)
    case = get_case()
    config = get_config()
    phase = isempty(args) ? "all" : first(args)
    task_id = let s = get(ENV, "SLURM_ARRAY_TASK_ID", nothing)
        isnothing(s) ? nothing : parse(Int, s)
    end

    phase in ("all", "data", "models", "reduce", "count") ||
        error("unknown phase '$(phase)'; expected all|data|models|reduce|count")

    if phase == "count"
        print_counts(config)
    elseif phase == "all"
        run_data!(case, config, nothing)
        run_models!(case, config, nothing)
        run_reduce!(case, config)
    elseif phase == "data"
        run_data!(case, config, task_id)
    elseif phase == "models"
        run_models!(case, config, task_id)
    elseif phase == "reduce"
        run_reduce!(case, config)
    end
    return 0
end
