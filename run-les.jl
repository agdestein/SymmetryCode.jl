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
setup |> pairs

config = (;
    # Closures included in every multi-model step. Order propagates to plots.
    # Available: :nomo, :dynsmag, :clar, :smag, :vers, :tbnn, :equi, :conv.
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
    train_mode = :skip,

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
        # :dissipation,      # get_dissipation_errors (loads DNS field)
    ],

    # Stage labels here force a re-compute regardless of cache. Uncomment
    # a line below (or `push!(config.force, :qr)` at the REPL) to invalidate.
    # Only the cached stages are listed; the others have nothing to invalidate.
    force = Set{Symbol}(
        [
            # :rollouts,
            # :les_stats,
            # :sfs,
            # :densities,
            # :equi_prior,
            # :equi_post,
            # :qr,
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
    @info "Training wall-time (seconds) per learned model"
    learned = filter(in([:tbnn, :equi, :conv]), config.models)
    map(learned) do k
        file = joinpath(setup.outdir, "ps-$(k).jld2")
        k => isfile(file) ? round(load(file)["timing"]; digits = 1) : :missing
    end |> NamedTuple |> pairs |> display
    flush(stdout)
end

if :rollouts in config.experiments
    S.solve_les(setup, models; force = :rollouts in config.force)
    upostfiles = S.get_upostfiles(setup)
    @info "LES rollout wall-time (seconds) per model"
    map(keys(models)) do k
        k => round(load_object(upostfiles[k]).timing; digits = 1)
    end |> NamedTuple |> pairs |> display
    flush(stdout)
end

if :les_stats in config.experiments
    les_stat =
        S.get_les_statistics_cached(setup, keys(models); force = :les_stats in config.force)
    @info "Time-mean relative LES error vs filtered DNS, per model"
    map(s -> round(mean(s.e_post); sigdigits = 4), les_stat) |> pairs |> display
    flush(stdout)
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
    @info "A-priori relative SFS error per model"
    map(x -> round(x.relerr; sigdigits = 4), err) |> pairs |> display
    @info "A-priori SFS cross-correlation per model"
    map(x -> round(x.crosscor; sigdigits = 4), err) |> pairs |> display
    flush(stdout)
end

if :equi_prior in config.experiments
    equi_models = NamedTuple(k => models[k] for k in keys(models) if k != :nomo)
    errs =
        S.apriori_equivariance_error(setup, equi_models; force = :equi_prior in config.force)
    @info "Mean a-priori equivariance error (over group elements) per model"
    map(x -> round(mean(x); sigdigits = 4), errs) |> pairs |> display
    flush(stdout)
    fig = S.plot_equivariance_errors(errs)
    save("$(setup.plotdir)/equi-errors-prior.pdf", fig; backend = CairoMakie)
end

if :equi_post in config.experiments
    errs = S.apost_equivariance_error(setup, models; force = :equi_post in config.force)
    @info "Mean a-posteriori equivariance error (over group elements) per model"
    map(x -> round(mean(x); sigdigits = 4), errs) |> pairs |> display
    flush(stdout)
    fig = S.plot_equivariance_errors(errs)
    save("$(setup.plotdir)/equi-errors-post.pdf", fig; backend = CairoMakie)
end

if :velocities in config.experiments
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
    @info "Median SFS dissipation per model (including :ref baseline)"
    map(x -> round(x; sigdigits = 4), diss) |> pairs |> display
    flush(stdout)
end

@info "Done."
flush(stderr)
