using JLD2
using LinearAlgebra
using Random
using SymmetryCode: SymmetryCode as S
using GLMakie
using CUDA
lines([1, 2, 3])

outdir = joinpath(@__DIR__, "output") |> mkpath

let
    t = 0.0
    cfl = 0.85
    tstop = 2.0e0
    Δt = 0.0
    # l = 5.0
    l = 1.0
    # l = 2π
    g = S.Grid{2}(; l, n = 4, backend = CUDABackend())
    visc = 1.0e-3
    cache = S.getcache(g)
    u = S.taylorgreen(g, cache.plan)
    e = round(S.energy(u); sigdigits = 4)
    i = 0
    while t < tstop
        if i > 0 # Skip first step to get initial statistics
            Δt = cfl * S.propose_timestep(u, g, visc, cache)
            Δt = min(Δt, tstop - t)
            S.wray3!(S.convectiondiffusion!, u, Δt, g, cache; visc)
            t += Δt
        end
        if i % 1 == 0
            foreach(u -> S.apply!(S.twothirds!, g, (u, g)), u) # Remove polluted components
            @info join(
                [
                    "t = $(round(t; sigdigits = 4))",
                    "Δt = $(round(Δt; sigdigits = 4))",
                ],
                ",\t",
            )
        end
        i += 1
    end
    uref0 = S.taylorgreen(g, cache.plan; doproject = false)
    decay = exp(-visc * 2 * (2π / g.l)^2 * tstop)
    @show decay
    uref = map(u -> decay * u, uref0)
    map(u, uref) do u, uref
        norm(u - uref) / norm(uref)
    end
end

jldsave("$outdir/taylorgreen-dissipation.jld2"; times, dissipation)
times, dissipation = load("$outdir/taylorgreen-dissipation.jld2", "times", "dissipation")

let
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "time", ylabel = "dissipation")
    lines!(ax, times, dissipation)
    fig
end

let
    D = S.dim(g)
    stat = S.turbulence_statistics(u, visc, g)
    s = S.spectrum(u, g)
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10)
    kmax = div(g.n, 2)
    kcut = div(2 * kmax, 3)
    # k = [2, 500]
    k = [2, g.n / 8]
    if D == 2
        kolmo = @. 2.0e0 * stat.diss^(1 / 3) * k^(-3)
        escale = stat.diss^(-2 / 3) * stat.l_kol^(-3)
    elseif D == 3
        kolmo = @. 5.0e-1 * stat.diss^(2 / 3) * k^(-5 / 3)
        escale = stat.diss^(-2 / 3) * stat.l_kol^(-5 / 3)
    end
    kscale = stat.l_kol
    lines!(ax, kscale * s.k, escale * s.s)
    lines!(kscale * k, escale * kolmo)
    # vlines!(kscale * kcut)
    # ylims!(1e-7, 1)
    fig
end

let
    s = S.turbulence_statistics(u, visc, g)
    s |> pairs
end

let
    a = S.scalarfield(g)
    b = S.scalarfield(g)
    randn!(b)
    S.apply!(S.twothirds!, g, a, b, g)
    a[:, :, 1] .|> abs |> Array |> heatmap
end
