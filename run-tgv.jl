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
# SLURM array distribution (mirrors run-les.jl). The distributable unit is one
# closure evaluated at every (tgv, Δf) point; with SLURM_ARRAY_TASK_ID=i set this
# process handles only `models[i]`, unset it runs them all serially. The TGV data
# generation is one inherently-serial high-res DNS per run (a single time march, not
# array-splittable), and the model-independent references and figures are
# cross-model — all run in the serial pass only. So the parallel workflow is:
# generate the data with a serial run (`experiments = [:data]`), submit the eval
# array (`--array=1-N`, N = the logged worklist length), then a final serial run for
# the references + figures. A plain serial run does the whole thing end-to-end.

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
    top = :p8000,
    netseeds = 0:1,
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

function main()
    case = get_case()
    config = get_config()

    slurm_id = get(ENV, "SLURM_ARRAY_TASK_ID", nothing)
    task_id = isnothing(slurm_id) ? nothing : parse(Int, slurm_id)
    serial = isnothing(task_id)

    models = eval_models(config)
    fams = families(config)
    tgvs = S.tgv_runs()
    points = [(tgv, Δf) for tgv in tgvs for Δf in case.filters_test]

    # TGV data generation: one inherently-serial high-res DNS per run, so it runs
    # only in the serial pass (before the eval loop). An eval array therefore assumes
    # the data already exists — generate it first with a serial `:data` run.
    if serial && :data in config.experiments
        for tgv in tgvs
            @info "===== TGV data: Re=$(tgv.Re_target), visc=$(tgv.visc) ====="
            flush(stderr)
            S.create_data_tgv(case, tgv; force = :data in config.force)
        end
    end

    #######################
    # Per-model work (distributed): evaluate the closure at every (tgv, Δf) point.
    #######################
    @info "TGV worklist: $(length(models)) units (SLURM array range 1-$(length(models)))"
    flush(stderr)
    for (i, m) in enumerate(models)
        serial || i == task_id || continue
        @info "===== unit $i/$(length(models)): $(S.modelname(m)) ====="
        flush(stderr)
        for (tgv, Δf) in points
            eval_model!(case, config, m, tgv, Δf)
        end
    end

    # An array task is done after its slice; everything below is the serial pass.
    if !serial
        @info "Done (array task $(task_id))."
        flush(stderr)
        return
    end

    #######################
    # Serial aggregation: model-independent references, then the per-eval-point
    # figures (cross-model). After an array run, rerun without the array env — the
    # per-model loop above is then all cache hits and only this section runs.
    #######################
    for (tgv, Δf) in points
        eval_ref!(case, config, tgv, Δf)
    end

    series = series_models(config)
    for (tgv, Δf) in points
        if :plots in config.experiments
            S.plot_apriori_bar(case, tgv, Δf, fams, config.netseeds; classical = config.classical)
            S.plot_dissipation_bar(case, tgv, Δf, fams, config.netseeds; classical = config.classical)
            S.plot_error_post(case, tgv, Δf, series)
            S.plot_spectrum_les(case, tgv, Δf, [:ref; series])
        end
        :dissipation in config.experiments &&
            S.plot_dissipation_tgv(case, tgv, Δf, [:ref; series])
        :field_evolution in config.experiments &&
            S.plot_field_evolution_tgv(case, tgv, Δf)
    end

    @info "Done."
    flush(stderr)
    return
end

main()
