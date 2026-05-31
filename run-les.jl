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

# Script-specific table file. create-data.jl and run-les.jl share a setup (hence
# a plotdir), so each writes its summary tables to its own file to avoid one
# script truncating the other's. Defined as a function (not a `const`) so both
# scripts can be `include`d into the same REPL session — a `const` in one would
# clash with the `const` of the same name in the other, whereas a function method
# is freely redefinable. These thin wrappers thread that filename through
# `S.tabulate` / `S.reset_tables` so the call sites below stay uncluttered.
tablefile() = "tables-les.txt"
reset_tables(setup; kwargs...) = S.reset_tables(setup; filename = tablefile(), kwargs...)
tabulate(args...; kwargs...) = S.tabulate(args...; filename = tablefile(), kwargs...)

function main()

    #######################
    # Setup + experiment config
    #######################

    # Pick one. `getsetup` derives output paths automatically.
    # setup = S.setup_laptop()
    setup = S.setup_turbulator_small()
    # setup = S.setup_turbulator_medium()
    # setup = S.setup_turbulator_large()
    # setup = S.setup_snellius()

    reset_tables(setup)
    tabulate(setup, "Problem setup", setup)

    let
        r = S.data_ranges(setup)
        dg = setup.datagen
        tabulate(
            setup,
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
            # :bard,
            :dynsmag,
            :clar,
            :conv,
            :equi,
            :tbnn,
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
            :stats,              # compute_sfs_stats -> sfs_stats_<k>.jld2; KDE + bar plots + tables
            :budget,             # compute_budget -> budget_<k>.jld2; KE(t) + eps_sfs(t) plot
            :spectral_transfer,  # compute_spectral_transfer -> transfer_<k>.jld2; eps_sfs(k) plot
            :equi_prior,         # apriori_equivariance_error + plot
            :equi_post,          # apost_equivariance_error + plot
            :velocities,         # plot_velocities slice grid
            :sfs_plot,           # plot_sfs tensor-field snapshots
            :qr,                 # compute_qr + plot_qr
        ],

        # Stage labels here force a re-compute regardless of cache. Uncomment
        # a line below (or `push!(config.force, :qr)` at the REPL) to invalidate.
        # Only the cached stages are listed; the others have nothing to invalidate.
        force = Set{Symbol}(
            [
                # :rollouts,
                # :les_stats,
                # :sfs,
                # :stats,
                # :budget,
                # :spectral_transfer,
                # :equi_prior,
                # :equi_post,
                # :qr,
            ]
        ),
    )

    #######################
    # Train learned closures
    #######################

    # train_models is a no-op when train_mode = :skip; otherwise each
    # learned closure is trained in isolation with clean() between them.
    S.train_models(setup, config.models; config.train_mode)

    #######################
    # Phase A — per-model compute loop
    #######################
    #
    # Build exactly one closure per iteration, run every per-model
    # artifact-producing experiment for it, then discard so the GPU
    # footprint at any moment is bounded by a single closure's working set.
    # Cached stages short-circuit per-key without instantiating the model.

    for key in config.models
        @info "===== compute phase: $(key) ====="
        flush(stderr)
        S.clean()
        model = S.build_models(setup, [key])[key]

        if :rollouts in config.experiments
            S.solve_les(setup, key, model; force = :rollouts in config.force)
        end
        if key != :nomo
            if :sfs in config.experiments
                S.predict_sfs(setup, key, model; force = :sfs in config.force)
            end
            if :equi_prior in config.experiments
                S.apriori_equivariance_error(
                    setup, key, model;
                    force = :equi_prior in config.force,
                )
            end
        end
        if :equi_post in config.experiments
            S.apost_equivariance_error(
                setup, key, model;
                force = :equi_post in config.force,
            )
        end
        if :budget in config.experiments
            S.compute_budget(setup, key, model; force = :budget in config.force)
        end
        if :spectral_transfer in config.experiments
            S.compute_spectral_transfer(
                setup, key, model;
                force = :spectral_transfer in config.force,
            )
        end

        model = nothing
        S.clean()
    end

    #######################
    # Phase B — aggregation and plotting (reads on-disk artifacts only)
    #######################

    if :training_summary in config.experiments
        learned = filter(in([:tbnn, :equi, :conv]), config.models)
        timings = NamedTuple(
            map(learned) do k
                file = joinpath(setup.outdir, "ps-$(k).jld2")
                k => isfile(file) ? load_object(file).timing : :missing
            end
        )
        tabulate(setup, "Training wall-time (seconds) per learned model", timings; digits = 1)
        S.plot_training(setup, config.models)
    end

    if :rollouts in config.experiments
        upostfiles = S.get_upostfiles(setup)
        timings = NamedTuple(k => load_object(upostfiles[k]).timing for k in config.models)
        tabulate(setup, "LES rollout wall-time (seconds) per model", timings; digits = 1)
    end

    if :les_stats in config.experiments
        les_stat = S.get_les_statistics_cached(
            setup, Tuple(config.models); force = :les_stats in config.force,
        )
        tabulate(
            setup,
            "Time-mean relative LES error vs filtered DNS, per model",
            map(s -> mean(s.e_post), les_stat),
        )
        S.plot_error_post(setup, les_stat)
    end

    if :spectrum_les in config.experiments
        les_stat = S.get_les_statistics_cached(setup, Tuple(config.models))
        S.plot_spectrum_les(setup, les_stat)
    end

    if :stats in config.experiments
        sfs_keys = filter(!=(:nomo), config.models)
        all_keys = [:ref; config.models]
        S.compute_sfs_stats(setup, all_keys; force = :stats in config.force)

        stats = NamedTuple(
            k => load_object("$(setup.outdir)/sfs_stats_$(k).jld2") for k in all_keys
        )
        tabulate(
            setup,
            "A-priori relative SFS error per model",
            map(s -> s.apriori.relerr, stats),
        )
        tabulate(
            setup,
            "A-priori SFS cross-correlation per model",
            map(s -> s.apriori.crosscor, stats),
        )
        tabulate(
            setup,
            "Median pointwise SFS dissipation per model (incl :ref baseline)",
            map(s -> s.diss.median, stats),
        )
        tabulate(
            setup,
            "Dissipation skewness per model (negative = backscatter tail)",
            map(s -> s.diss.skewness, stats),
        )
        tabulate(
            setup,
            "Backscatter fraction per model (τ:S > 0; 0 for Smag by construction)",
            map(s -> s.diss.backscatter, stats),
        )

        S.plot_densities(setup, [:ref; sfs_keys]; dolog = true)
        S.plot_dissipation_bar(setup, all_keys)
        S.plot_backscatter_bar(setup, all_keys)
        S.plot_apriori_bar(setup, config.models)
    end

    if :budget in config.experiments
        S.compute_budget(setup, :ref; force = :budget in config.force)
        S.plot_budget(setup, [:ref; config.models])
    end

    if :spectral_transfer in config.experiments
        S.compute_spectral_transfer(setup, :ref; force = :spectral_transfer in config.force)
        S.plot_spectral_transfer(setup, [:ref; config.models])
    end

    if :equi_prior in config.experiments
        equi_keys = filter(!=(:nomo), config.models)
        errs = S.load_equivariance_errors(setup, equi_keys, :prior)
        tabulate(
            setup,
            "Mean a-priori equivariance error (over group elements) per model",
            map(mean, errs),
        )
        S.plot_equivariance_errors(setup, errs; tag = :prior)
    end

    if :equi_post in config.experiments
        errs = S.load_equivariance_errors(setup, config.models, :post)
        tabulate(
            setup,
            "Mean a-posteriori equivariance error (over group elements) per model",
            map(mean, errs),
        )
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

    @info "Done."
    flush(stderr)

    return
end

main()
