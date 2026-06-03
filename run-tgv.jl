# Second experiment: apply the closures trained on forced turbulence (by
# `run-les.jl`) — *unchanged* — to a decaying Taylor-Green vortex, swept over a
# range of integral Reynolds numbers. The TGV test probes generalization to
# (1) laminar/transitional/turbulent regimes, (2) an unforced decaying flow, and
# (3) the canonical dissipation benchmark; the Reynolds sweep additionally turns
# the closures' regime mis-calibration into a *trend* vs Re (see `main`).
#
# The models are reused verbatim: the learned closures are amplitude- and
# Reynolds-invariant by construction (they regress the normalized target
# τ/(Δ²‖∇u‖²) from the normalized gradient ∇u/‖∇u‖), so only ν, n_les and Δ must
# stay fixed. `setup_taylorgreen` copies those from the training setup and only
# changes the flow (TGV initial condition, no forcing).

@info "Loading packages"
flush(stderr)

using Adapt
using CairoMakie
using CUDA, cuDNN
using JLD2
using Statistics: mean

import SymmetryCode as S

#######################
# Setup + experiment config (shared across the Reynolds sweep)
#######################

# Pick the *training* setup whose trained closures (ps-*.jld2) we reuse.
# The paper uses setup_snellius; the others work identically.

get_setup() = S.setup_turbulator_small()
# get_setup() = S.setup_turbulator_medium()
# get_setup() = S.setup_turbulator_large()
# get_setup() = S.setup_snellius()

get_config() = (;
    # Closures included in every multi-model step. Order propagates to plots.
    models = [
        :nomo,
        :dynsmag,
        :clar,
        :conv,
        :equi,
        :tbnn,
    ],

    # Nominal (V₀L/ν) integral Reynolds numbers to sweep. The forced-training
    # anchor's *measured* Re_int is computed from its data, not hard-coded; the
    # plot likewise uses each TGV's measured Re_int at peak dissipation.
    # Re_targets = [1600, 2300, 4000],
    Re_targets = [3000, 6000, 9000],

    # Focused stage set: the dissipation/transition benchmark plus a-priori
    # generalization metrics. (Q-R, equivariance and field snapshots from
    # run-les.jl are omitted here but would work the same way against `tgv`.)
    experiments = [
        :rollouts,           # solve_les -> u-post-<key>.jld2
        :budget,             # compute_budget -> budget_<k>.jld2; dissipation(t) benchmark
        :les_stats,          # get_les_statistics -> error-vs-t plot
        :spectrum_les,       # LES energy-spectrum plot
        :sfs,                # predict_sfs -> sfs_<key>.jld2
        :stats,              # compute_sfs_stats -> KDE + bar plots + tables
        :spectral_transfer,  # compute_spectral_transfer -> eps_sfs(k) plot
    ],

    # Stage labels here force a re-compute regardless of cache.
    force = Set{Symbol}(
        [
            # :rollouts,
            # :budget,
            # :les_stats,
            # :sfs,
            # :stats,
            # :spectral_transfer,
        ]
    ),
)

"""
Run the full per-Reynolds Taylor-Green test for one derived setup `tgv`:
generate the DNS + filtered (ubar, τ) data, run every per-model experiment,
and produce the per-Re plots/tables. `train` owns the trained closures and
`config` selects the model/experiment set (shared across the sweep).
"""
function run_tgv(train, tgv, config)

    S.reset_tables(tgv)
    S.tabulate(tgv, "Problem setup (Taylor-Green test)", tgv)

    # Report that the closures are reused unchanged from training.
    S.tabulate(
        tgv,
        "Closures reused unchanged from training",
        (;
            train = train.name,
            ps_dir = train.outdir,
            visc = tgv.visc,
            n_les = tgv.n_les,
            Δ = tgv.Δ,
            V0 = tgv.V0,
            Re_TGV = tgv.V0 / tgv.visc,
            forced = tgv.forced,
        ),
    )

    #######################
    # Generate Taylor-Green data (DNS + filtered (ubar, τ) pairs)
    #######################

    S.create_data_tgv(tgv)

    # Report DNS resolution at peak dissipation (kmax·η ≳ 1 is well-resolved).
    let
        data = joinpath(tgv.outdir, "data.jld2") |> load_object
        diss = [s.diss for s in data.statistics_dns]
        ipk = argmax(diss)
        stat = data.statistics_dns[ipk]
        S.tabulate(
            tgv,
            "DNS statistics at peak dissipation",
            (;
                t_peak = data.times[ipk],
                t_star_peak = data.times[ipk] * tgv.V0,
                stat...,
            ),
        )
    end

    #######################
    # Phase A — per-model compute loop
    #######################
    #
    # Build exactly one closure per iteration from the *training* setup (so the
    # learned ones load train's ps-*.jld2), run every per-model artifact-
    # producing experiment for it against the TGV setup, then discard so the GPU
    # footprint at any moment is bounded by a single closure's working set.

    for key in config.models
        @info "===== compute phase: $(key) ====="
        flush(stderr)
        S.clean()
        model = S.build_models(train, [key])[key]

        if :rollouts in config.experiments
            S.solve_les(tgv, key, model; force = :rollouts in config.force)
        end
        if key != :nomo
            if :sfs in config.experiments
                S.predict_sfs(tgv, key, model; force = :sfs in config.force)
            end
        end
        if :budget in config.experiments
            S.compute_budget(tgv, key, model; force = :budget in config.force)
        end
        if :spectral_transfer in config.experiments
            S.compute_spectral_transfer(
                tgv, key, model;
                force = :spectral_transfer in config.force,
            )
        end

        model = nothing
        S.clean()
    end

    #######################
    # Phase B — aggregation and plotting (reads on-disk artifacts only)
    #######################

    if :rollouts in config.experiments
        upostfiles = S.get_upostfiles(tgv)
        timings = NamedTuple(k => load_object(upostfiles[k]).timing for k in config.models)
        S.tabulate(tgv, "LES rollout wall-time (seconds) per model", timings; digits = 1)
    end

    if :budget in config.experiments
        S.compute_budget(tgv, :ref; force = :budget in config.force)
        S.plot_budget(tgv, [:ref; config.models])
        # Headline Taylor-Green benchmark: ε*(t*) and E*(t*) vs DNS + published Re=1600.
        S.plot_dissipation_tgv(tgv, [:ref; config.models])
    end

    if :les_stats in config.experiments
        les_stat = S.get_les_statistics_cached(
            tgv, Tuple(config.models); force = :les_stats in config.force,
        )
        S.tabulate(
            tgv,
            "Time-mean relative LES error vs filtered DNS, per model",
            map(s -> mean(s.e_post), les_stat),
        )
        S.plot_error_post(tgv, les_stat)
    end

    if :spectrum_les in config.experiments
        les_stat = S.get_les_statistics_cached(tgv, Tuple(config.models))
        S.plot_spectrum_les(tgv, les_stat)
    end

    if :stats in config.experiments
        sfs_keys = filter(!=(:nomo), config.models)
        all_keys = [:ref; config.models]
        S.compute_sfs_stats(tgv, all_keys; force = :stats in config.force)

        stats = NamedTuple(
            k => load_object("$(tgv.outdir)/sfs_stats_$(k).jld2") for k in all_keys
        )
        S.tabulate(
            tgv,
            "A-priori relative SFS error per model",
            map(s -> s.apriori.relerr, stats),
        )
        S.tabulate(
            tgv,
            "A-priori SFS cross-correlation per model",
            map(s -> s.apriori.crosscor, stats),
        )
        S.tabulate(
            tgv,
            "Median pointwise SFS dissipation per model (incl :ref baseline)",
            map(s -> s.diss.median, stats),
        )
        S.tabulate(
            tgv,
            "Backscatter fraction per model (τ:S > 0; 0 for Smag by construction)",
            map(s -> s.diss.backscatter, stats),
        )

        S.plot_densities(tgv, [:ref; sfs_keys]; dolog = true)
        S.plot_dissipation_bar(tgv, all_keys)
        S.plot_backscatter_bar(tgv, all_keys)
        S.plot_apriori_bar(tgv, config.models)
    end

    if :spectral_transfer in config.experiments
        S.compute_spectral_transfer(tgv, :ref; force = :spectral_transfer in config.force)
        S.plot_spectral_transfer(tgv, [:ref; config.models])
    end

    @info "Done with Taylor-Green test at Re = $(round(Int, tgv.Re_target))."
    flush(stderr)

    return
end

function main()
    # Setup and config (from top of file)
    train = get_setup()
    config = get_config()

    #######################
    # Reynolds sweep
    #######################
    #
    # Apply the *same* trained closures to a decaying Taylor-Green vortex at
    # several integral Reynolds numbers (Re = V₀L/ν, set via V₀ = Re·ν with ν
    # fixed to match training). This turns the over-dissipation of the learned
    # closures — calibrated on the forced training regime — into a *trend* vs Re
    # rather than a single anecdote. Each Re mirrors train's ν, n_les, Δ exactly;
    # only the flow differs. setup_taylorgreen gives each Re its own outdir
    # (`tgv_<train>_Re=<Re>_n=<n>`), so artifacts and per-Re plots never collide.
    tgvs = map(config.Re_targets) do Re_target
        # For local setups plotdir lives inside outdir; for Snellius outdir is on
        # the cluster so we root plotdir under train.plotdir's parent instead.
        tgv_name = "tgv_$(train.name)_Re=$(Re_target)_n=$(train.n_dns)"
        plotroot = dirname(train.plotdir) == train.outdir ?
            dirname(train.outdir) : dirname(train.plotdir)
        plotdir = joinpath(plotroot, tgv_name, "plots") |> mkpath
        outdir = joinpath(dirname(train.outdir), tgv_name) |> mkpath
        tgv = S.setup_taylorgreen(train; Re_target, outdir, plotdir)

        # Run TGV for current Re_target
        run_tgv(train, tgv, config)

        # Return setup (we need the paths for each Re_target)
        return tgv
    end

    #######################
    # Cross-Reynolds aggregation — the generalization trend
    #######################
    #
    # Anchor the trend at the forced training regime: pass the training setup
    # itself, so the plot reads its measured eval-window-mean Re_int (and its
    # sfs_stats under train.outdir) directly — best-effort, skipped if those
    # artifacts are not present (e.g. when the forced data is only on the cluster).
    S.plot_dissipation_vs_re(
        tgvs, [:ref; config.models];
        train_anchor = train,
        Re_key = :Re_int,
    )
    S.plot_dissipation_vs_re(
        tgvs, [:ref; config.models];
        train_anchor = train,
        Re_key = :Re_tay,
    )

    @info "Done with Reynolds sweep."
    flush(stderr)
    return
end

main()
