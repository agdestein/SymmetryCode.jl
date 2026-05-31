@info "Loading packages"
flush(stderr)

using Adapt
using CairoMakie
using CUDA
using JLD2
using Statistics: mean
using WGLMakie

import SymmetryCode as S

# Warmup plot
lines([1, 2, 3])

# Script-specific table file. create-data.jl and run-les.jl share a setup (hence
# a plotdir), so each writes its summary tables to its own file to avoid one
# script truncating the other's. These thin wrappers thread that filename through
# `S.tabulate` / `S.reset_tables` so the call sites below stay uncluttered.
const tablefile = "tables-data.txt"
reset_tables(setup; kwargs...) = S.reset_tables(setup; filename = tablefile, kwargs...)
tabulate(args...; kwargs...) = S.tabulate(args...; filename = tablefile, kwargs...)

# Stage 1 of the pipeline: DNS warm-up (`create_dns`) followed by (ubar, τ) data
# generation (`create_data`), plus the diagnostic plots for both. Produces
# `dns.jld2` and `data.jld2`, which `run-les.jl` consumes. The `config.experiments`
# list below toggles which stages run, mirroring `run-les.jl` / `run-tgv.jl`.
function main()

    #######################
    # Setup + experiment config
    #######################

    # Pick one. Output paths are derived automatically.
    # setup = S.setup_laptop()
    setup = S.setup_turbulator_small()
    # setup = S.setup_turbulator_medium()
    # setup = S.setup_turbulator_large()
    # setup = S.setup_snellius()

    reset_tables(setup)
    tabulate(setup, "Problem setup", setup)

    config = (;
        # Pipeline stages to execute, in order. `create_dns` (:dns) writes
        # dns.jld2; `create_data` (:data) reads it and writes data.jld2; the
        # plot stages read whichever artifact already exists on disk.
        experiments = [
            :dns,              # create_dns -> dns.jld2 (DNS warm-up) + tabulate warm-up stats
            :spectrum_dns,     # plot_spectrum_dns -> DNS energy spectrum
            :evolution_dns,    # plot_evolution_dns -> DNS time series
            # :dissipation_fd, # plot_dissipation_finite_difference -> ε vs dE/dt check
            :data,             # create_data -> data.jld2 ((ubar, τ) pairs) + tabulate time-avg stats
            :dnsfield,         # heatmap of a DNS component + its filtered counterpart
            :evolution_data,   # plot_evolution_data -> (ubar, τ) data time series
            :spectrum_data,    # plot_spectrum_data -> DNS vs filtered-DNS spectra
        ],

        # Stage labels here force a re-compute regardless of cache. By default
        # :dns/:data short-circuit when dns.jld2/data.jld2 already exist; uncomment
        # a line below (or `push!(config.force, :data)` at the REPL) to invalidate.
        # Only these two stages are cached; the plot stages always regenerate.
        force = Set{Symbol}(
            [
                # :dns,
                # :data,
            ]
        ),
    )

    #######################
    # DNS warm-up
    #######################

    if :dns in config.experiments
        S.create_dns(setup; force = :dns in config.force)

        # Tabulate statistics at the end of warm-up, to sanity-check the
        # turbulent state (Reynolds numbers, resolution) before data generation.
        let
            statistics = load("$(setup.outdir)/dns.jld2", "statistics")
            tabulate(setup, "DNS statistics after warm-up", statistics[end])
        end
    end

    if :spectrum_dns in config.experiments
        S.plot_spectrum_dns(setup)
    end

    if :evolution_dns in config.experiments
        S.plot_evolution_dns(setup)
    end

    if :dissipation_fd in config.experiments
        S.plot_dissipation_finite_difference(setup)
    end

    #######################
    # (ubar, τ) data generation
    #######################

    if :data in config.experiments
        S.create_data(setup; force = :data in config.force)

        # Tabulate the statistics averaged over the data-generation window, for
        # reporting the characteristic Reynolds numbers, length/time scales and
        # resolution of the (ubar, τ) dataset. Done for both the DNS and the
        # filtered (LES-grid) fields.
        let
            data = joinpath(setup.outdir, "data.jld2") |> load_object
            timemean(stats) = let ks = keys(first(stats))
                NamedTuple{ks}(map(k -> mean(getindex.(stats, k)), ks))
            end
            tabulate(
                setup, "Time-averaged DNS statistics (data window)",
                timemean(data.statistics_dns),
            )
            tabulate(
                setup, "Time-averaged filtered-DNS statistics (data window)",
                timemean(data.statistics_les),
            )
        end
    end

    if :dnsfield in config.experiments
        # Plot DNS component
        let
            (; D) = setup
            u = load("$(setup.outdir)/dns.jld2", "u") |> u -> map(copy, u) |> adapt(setup.backend)
            g = S.Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
            v = S.spacescalarfield(g)
            p = S.getplan(g)
            if D == 2
                S.to_phys!(v, u.x, p, g)
                field = v |> Array
            else
                S.to_phys!(v, u.z, p, g)
                field = v[:, :, end] |> Array
            end
            fig, _ = heatmap(field; colormap = :RdBu)
            save("$(setup.plotdir)/dnsfield.png", fig)
            fig
        end

        # Plot filtered DNS component
        let
            (; D) = setup
            data = joinpath(setup.outdir, "data.jld2") |> load_object
            u = map(copy, data.inputs[1]) |> adapt(setup.backend)
            g = S.Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
            v = S.spacescalarfield(g)
            p = S.getplan(g)
            if D == 2
                S.to_phys!(v, u.x, p, g)
                field = v |> Array
            else
                S.to_phys!(v, u.z, p, g)
                field = v[:, :, end] |> Array
            end
            fig, _ = heatmap(field; colormap = :RdBu)
            save("$(setup.plotdir)/dnsfield_filtered.png", fig)
            fig
        end
    end

    if :evolution_data in config.experiments
        S.plot_evolution_data(setup)
    end

    if :spectrum_data in config.experiments
        S.plot_spectrum_data(setup)
    end

    @info "Done."
    flush(stderr)

    return
end

main()
