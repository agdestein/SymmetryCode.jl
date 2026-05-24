@info "Loading packages"
flush(stderr)

using Adapt
using CairoMakie
using CUDA, cuDNN
using JLD2
using Statistics: mean
using WGLMakie

import SymmetryCode as S

# Warmup plot
lines([1, 2, 3])

#######################
# Setup + experiment config
#######################

# Pick one. `getsetup` derives output paths automatically.
# setup = S.setup_laptop()
setup = S.setup_turbulator_small()
# setup = S.setup_turbulator_medium()
# setup = S.setup_turbulator_large()
# setup = S.setup_snellius()

S.tabulate("Problem setup", setup)

let r = S.data_ranges(setup), dg = setup.datagen
    S.tabulate(
        "Train/eval split (data.jld2 snapshot indices)",
        (;
            nstep = dg.nstep,
            tstop = dg.tstop,
            n_train = dg.n_train,
            t_train = dg.tstop * dg.n_train / dg.nstep,
            n_eval = length(r.eval),
            eval_first = r.eval[1],
            eval_last = r.eval[end],
        ),
    )
end

config = (;
    # Closures included in every multi-model step. Order propagates to plots.
    # Available: :nomo, :dynsmag, :clar, :smag, :vers, :bard, :tbnn, :equi, :conv.
    models = [
        :nomo,
        # :smag,
        # :vers,
        :dynsmag,
        :clar,
        :tbnn,
        # :equi,
        :conv,
    ],

    # How to load trainable closures (:skip loads ps-<key>.jld2 without
    # retraining; :resume continues from a checkpoint; :scratch retrains).
    train_mode = :scratch,

    # Pipeline stages to execute. Cached stages skip per-key when their
    # artifact already exists under setup.outdir — see :force for invalidation.
    experiments = [
        :training_summary,   # print training wall-time per learned model
        :rollouts,           # solve_les -> u-post-<key>.jld2
        :les_stats,          # get_les_statistics -> les_stat.jld2 + error-vs-t plot
        :spectrum_les,       # LES energy-spectrum plot
        :sfs,                # predict_sfs -> sfs_<key>.jld2
        :densities,          # compute_densities + plot_densities
        :apriori,            # apriori_error (prints table; not persisted)
        :equi_prior,         # apriori_equivariance_error + plot
        :equi_post,          # apost_equivariance_error + plot
        :velocities,         # plot_velocities slice grid
        :sfs_plot,           # plot_sfs tensor-field snapshots
        :qr,                 # compute_qr + plot_qr
        :dissipation,        # get_dissipation_errors (loads DNS field)
    ],

    # Stage labels here force a re-compute regardless of cache. Uncomment
    # a line below (or `push!(config.force, :qr)` at the REPL) to invalidate.
    # Only the cached stages are listed; the others have nothing to invalidate.
    force = Set{Symbol}(
        [
            :rollouts,
            :les_stats,
            :sfs,
            :densities,
            :equi_prior,
            :equi_post,
            :qr,
        ]
    ),
)

#######################
# Build closure models
#######################

models = S.build_models(setup, config.models; config.train_mode)

#######################
# Pipeline
#######################

if :training_summary in config.experiments
    learned = filter(in([:tbnn, :equi, :conv]), config.models)
    timings = NamedTuple(
        map(learned) do k
            file = joinpath(setup.outdir, "ps-$(k).jld2")
            k => isfile(file) ? load_object(file).timing : :missing
        end
    )
    S.tabulate("Training wall-time (seconds) per learned model", timings; digits = 1)
end

if :rollouts in config.experiments
    S.solve_les(setup, models; force = :rollouts in config.force)
    upostfiles = S.get_upostfiles(setup)
    timings = NamedTuple(k => load_object(upostfiles[k]).timing for k in keys(models))
    S.tabulate("LES rollout wall-time (seconds) per model", timings; digits = 1)
end

if :les_stats in config.experiments
    les_stat =
        S.get_les_statistics_cached(setup, keys(models); force = :les_stats in config.force)
    S.tabulate(
        "Time-mean relative LES error vs filtered DNS, per model",
        map(s -> mean(s.e_post), les_stat),
    )
    S.plot_error_post(setup, les_stat)
end

if :spectrum_les in config.experiments
    les_stat = S.get_les_statistics_cached(setup, keys(models))
    S.plot_spectrum_les(setup, les_stat)
end

if :sfs in config.experiments
    sfs_models = NamedTuple(k => models[k] for k in keys(models) if k != :nomo)
    S.predict_sfs(setup, sfs_models; force = :sfs in config.force)
end

if :densities in config.experiments
    sfs_keys = filter(!=(:nomo), config.models)
    S.compute_densities(setup, sfs_keys; force = :densities in config.force)
    S.plot_densities(setup, [:ref; sfs_keys]; dolog = true)
end

if :apriori in config.experiments
    err = S.apriori_error(setup, config.models)
    S.tabulate("A-priori relative SFS error per model", map(x -> x.relerr, err))
    S.tabulate("A-priori SFS cross-correlation per model", map(x -> x.crosscor, err))
end

if :equi_prior in config.experiments
    equi_models = NamedTuple(k => models[k] for k in keys(models) if k != :nomo)
    errs =
        S.apriori_equivariance_error(setup, equi_models; force = :equi_prior in config.force)
    S.tabulate("Mean a-priori equivariance error (over group elements) per model", map(mean, errs))
    S.plot_equivariance_errors(setup, errs; tag = :prior)
end

if :equi_post in config.experiments
    errs = S.apost_equivariance_error(setup, models; force = :equi_post in config.force)
    S.tabulate("Mean a-posteriori equivariance error (over group elements) per model", map(mean, errs))
    S.plot_equivariance_errors(setup, errs; tag = :post)
end

if :velocities in config.experiments
    S.plot_velocities(setup, :x, [:ref; config.models])
    S.plot_velocities(setup, :z, [:ref; config.models])
end

if :sfs_plot in config.experiments
    S.plot_sfs(setup, filter(!=(:nomo), config.models))
end

if :qr in config.experiments
    qr_keys = [:ref; config.models]
    S.compute_qr(setup, qr_keys; force = :qr in config.force)
    S.plot_qr(setup, qr_keys)
end

if :dissipation in config.experiments
    u_dns = load("$(setup.outdir)/dns.jld2", "u") |> adapt(setup.backend)
    diss = S.get_dissipation_errors(; setup, u_dns, models)
    S.tabulate("Median SFS dissipation per model (including :ref baseline)", diss)
end

@info "Done."
flush(stderr)
