@info "Loading packages"
flush(stderr)

using Adapt
using CairoMakie
using CUDA
using JLD2
using WGLMakie

import SymmetryCode as S

# Warmup plot
lines([1, 2, 3])

# Choose setup
setup = S.setup_laptop()
setup = S.setup_turbulator_small()
setup = S.setup_turbulator_medium()
setup = S.setup_turbulator_large()
setup = S.setup_snellius()
setup |> pairs

# Warmup simulation
S.create_dns(setup)

# Show statistics after warm-up
let
    statistics = load("$(setup.outdir)/dns.jld2", "statistics")
    s = statistics[end]
    s |> pairs |> display
    flush(stdout)
end

# Plot DNS spectrum
S.plot_spectrum_dns(setup)

# Plot time series
S.plot_evolution_dns(setup)

# # Plot dissipation vs finite difference of energy
# S.plot_dissipation_finite_difference(setup)

S.create_data(setup);

# data = joinpath(setup.outdir, "data.jld2") |> load_object;

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

# Base.summarysize(data) * 1.0e-9
#
# data |> pairs
#
# getindex.(data.statistics_dns, :diss)
# getindex.(data.statistics_dns, :uavg) .^ 2 / 2 * 3
# getindex.(data.statistics_dns, :Re_tay)
# getindex.(data.statistics_dns, :t_int)
# getindex.(data.statistics_dns, :l_int)
# getindex.(data.statistics_dns, :l_kol)

S.plot_evolution_data(setup)

# Plot normalized evolution of some statistics
let
    data = joinpath(setup.outdir, "data.jld2") |> load_object
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Quantity")
    s = data.statistics_dns
    t = data.times
    for (key, label) in [
            (:diss, "Dissipation"),
            (:uavg, "Kinetic Energy"),
            (:Re_tay, "Taylor Reynolds"),
            (:t_int, "Integral time"),
        ]
        y = getindex.(s, key)
        lines!(ax, t, y ./ maximum(y); label)
    end
    eps = 0.1
    ylims!(ax, -eps, 1 + eps)
    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        tellheight = true,
        framevisible = false,
        horizontal = true,
        nbanks = 4,
    )
    save(joinpath(setup.plotdir, "evolution_data.pdf"), fig; backend = CairoMakie)
    fig
end

S.plot_spectrum_data(setup)
