# Train LES closures and evaluate them a-priori and a-posteriori.
# Adjust the parameters in `get_setup` and `get_config`.
# Results are written to `plotdir` and arrays are cached in `outdir`.

@info "Loading packages"
flush(stderr)

using Adapt
using CairoMakie
using CUDA, cuDNN
using JLD2
using Statistics: mean

import SymmetryCode as S

#######################
# Setup + experiment config
#######################

# Pick one. `getsetup` derives output paths automatically.

# get_setup() = S.setup_laptop()
get_setup() = S.setup_turbulator_small()
# get_setup() = S.setup_turbulator_medium()
# get_setup() = S.setup_turbulator_large()
# get_setup() = S.setup_snellius()

get_config() = (;
    # Closures included in every multi-model step. Order propagates to plots.
    # Available: :nomo, :dynsmag, :clar, :smag, :vers, :bard, :tbnn, :equi,
    # :conv, :convsym (= :conv with group-averaged, exactly equivariant
    # inference; reuses :conv's trained parameters, costs |G| = 48 forward
    # passes per closure call).
    models = [
        :nomo,
        # :smag,
        # :vers,
        # :bard,
        :dynsmag,
        :clar,
        :conv,
        :convsym,
        :equi,
        :tbnn,
    ],

    # How to load trainable closures (:skip loads ps-<key>.jld2 without
    # retraining; :resume continues from a checkpoint, skipping seeds whose
    # training already completed; :scratch retrains).
    train_mode = :skip,

    # Training seeds for the :seeds robustness sweep. The first (canonical)
    # seed backs all single-model figures and reuses the plain-key artifacts;
    # the others train the learned closures again (network init + batch
    # shuffling) so scalar metrics get a mean ± std over seeds.
    seeds = 0:4,

    # Smoothing parameter for q-r plot
    # smooth_σ = nothing,
    smooth_σ = 2,

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
        :seeds,              # per-seed rollout + a-priori stats -> seed_stats.jld2
        :paper_tables,       # write_errors_table -> plotdir/errors.tex
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
            # :seeds,
        ]
    ),
)


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

"Just a little helper for designing architectures."
function printnetsizes()
    setup = get_setup()
    config = get_config()
    reset_tables(setup)
    learned = filter(in([:tbnn, :equi, :conv]), config.models)
    tabulate(
        setup,
        "Trainable parameters per learned model (equi counted pre-synthesis)",
        NamedTuple(k => S.paramcount(setup, k) for k in learned),
    )
    error()
end

# # Print sizes and raise error
# printnetsizes()

function main()
    # Load setup/config from top of file
    setup = get_setup()
    config = get_config()
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

    #######################
    # Train learned closures
    #######################

    # train_models is a no-op when train_mode = :skip; otherwise each
    # (learned closure, seed) pair is trained in isolation with clean()
    # between them. Only the canonical (first) seed is trained unless the
    # :seeds stage is active.
    S.train_models(
        setup, config.models;
        config.train_mode,
        seeds = :seeds in config.experiments ? config.seeds : (setup.train_setup.seed,),
    )

    #######################
    # Phase A — per-model compute loop
    #######################
    #
    # Each closure is built lazily, at most once per key via the memoized
    # `getmodel` thunk: a stage only invokes it on a cache miss, so cached or
    # plot-only runs never instantiate the model (no ps-*.jld2 / GPU needed).
    # The closure goes out of scope at the per-key `clean()`, bounding the GPU
    # footprint to a single closure's working set.

    for key in config.models
        @info "===== compute phase: $(key) ====="
        flush(stderr)
        S.clean()

        local built = nothing
        getmodel() = (built === nothing && (built = S.build_models(setup, [key])[key]); built)

        if :rollouts in config.experiments
            S.solve_les(setup, key, getmodel; force = :rollouts in config.force)
        end
        if key != :nomo
            if :sfs in config.experiments
                S.predict_sfs(setup, key, getmodel; force = :sfs in config.force)
            end
            if :equi_prior in config.experiments
                S.apriori_equivariance_error(
                    setup, key, getmodel;
                    force = :equi_prior in config.force,
                )
            end
        end
        if :equi_post in config.experiments && key != :convsym
            # :convsym is skipped: it is equivariant by construction (the
            # a-priori stage confirms machine precision) and the a-posteriori
            # test costs 48 rollouts at 48 forward passes per closure call.
            S.apost_equivariance_error(
                setup, key, getmodel;
                force = :equi_post in config.force,
            )
        end
        if :budget in config.experiments
            S.compute_budget(setup, key, getmodel; force = :budget in config.force)
        end
        if :spectral_transfer in config.experiments
            S.compute_spectral_transfer(
                setup, key, getmodel;
                force = :spectral_transfer in config.force,
            )
        end

        built = nothing
        S.clean()
    end

    #######################
    # Phase A2 — seed sweep for the learned closures
    #######################
    #
    # Repeats rollout + a-priori stats for every (learned model, seed) pair so
    # the scalar metrics get a spread over training seeds. The canonical seed
    # maps to the plain key (seed_key) and therefore reuses the Phase A
    # artifacts. :convsym rides on :conv's seeds and only the a-priori part is
    # swept — its rollout costs |G|× the MLP's, and the canonical-seed rollout
    # from Phase A already provides its a-posteriori error.
    if :seeds in config.experiments
        learned = filter(in([:tbnn, :equi, :conv, :convsym]), config.models)
        for key in learned, seed in config.seeds
            skey = S.seed_key(setup, key, seed)
            @info "===== seed sweep: $(skey) ====="
            flush(stderr)
            S.clean()
            local built = nothing
            getmodel() =
                (built === nothing && (built = S.build_seed_model(setup, key, seed)); built)
            force = :seeds in config.force
            key == :convsym || S.solve_les(setup, skey, getmodel; force)
            S.predict_sfs(setup, skey, getmodel; force)
            key == :convsym || S.apriori_equivariance_error(setup, skey, getmodel; force)
            built = nothing
            S.clean()
        end
        skeys = [S.seed_key(setup, k, s) for k in learned for s in config.seeds]
        S.compute_sfs_stats(setup, skeys; force = :seeds in config.force)
    end

    #######################
    # Phase B — aggregation and plotting (reads on-disk artifacts only)
    #######################

    # Seed aggregation first: the bar plots, error-vs-time plot, and LaTeX
    # tables below all accept the spread when it is available.
    seed_stat = nothing
    if :seeds in config.experiments
        learned = filter(in([:tbnn, :equi, :conv, :convsym]), config.models)
        seed_stat = S.get_seed_statistics_cached(
            setup, Tuple(learned), config.seeds;
            force = :seeds in config.force,
        )
        for (metric, lbl) in [
                (:relerr, "A-priori relative SFS error"),
                (:crosscor, "A-priori SFS cross-correlation"),
                (:e_post, "Time-mean relative LES error"),
                (:equi, "Mean a-priori equivariance error"),
                (:diss_median, "Median SFS dissipation / reference"),
                (:backscatter, "Backscatter fraction"),
            ]
            tabulate(
                setup,
                "$(lbl), mean ± std over seeds $(config.seeds)",
                map(s -> S.pm_string(s[metric]), seed_stat),
            )
        end
    end

    if :training_summary in config.experiments
        learned = filter(in([:tbnn, :equi, :conv]), config.models)
        timings = NamedTuple(
            map(learned) do k
                file = joinpath(setup.outdir, "ps-$(k).jld2")
                k => isfile(file) ? load_object(file).timing : :missing
            end
        )
        tabulate(setup, "Training wall-time (seconds) per learned model", timings; digits = 1)
        tabulate(
            setup,
            "Trainable parameters per learned model (equi counted pre-synthesis)",
            NamedTuple(k => S.paramcount(setup, k) for k in learned),
        )
        S.plot_training(setup, config.models)
    end

    if :rollouts in config.experiments
        timings = NamedTuple(
            k => load_object(S.upostfile(setup, k)).timing for k in config.models
        )
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
        S.plot_error_post(setup, les_stat; seed_stat)
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
        S.plot_dissipation_bar(setup, all_keys; seed_stat)
        S.plot_backscatter_bar(setup, all_keys; seed_stat)
        S.plot_apriori_bar(setup, config.models; seed_stat)
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
        S.plot_velocities(setup, :vortz, [:ref; config.models])
    end

    if :sfs_plot in config.experiments
        S.plot_sfs(setup, filter(!=(:nomo), config.models))
    end

    if :qr in config.experiments
        qr_keys = [:ref; config.models]
        S.compute_qr(setup, qr_keys; force = :qr in config.force)
        S.plot_qr(setup, qr_keys; config.smooth_σ)
    end

    if :paper_tables in config.experiments
        # Paper-ready errors table (mean ± std over seeds where available);
        # copy plotdir/errors.tex over the paper repo's tables/errors.tex.
        S.write_errors_table(setup, config.models; seed_stat)
    end

    @info "Done."
    flush(stderr)

    return
end

main()
