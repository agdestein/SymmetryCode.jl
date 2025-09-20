if false
    include("src/SymmetryCode.jl")
    using .SymmetryCode
    using .SymmetryCode.Spectral
end

using Adapt
using CUDA, cuDNN
using FFTW
using JLD2
using KernelAbstractions
using KernelDensity
using LinearAlgebra
using Lux
using MLUtils
using Random
using Seneca
using StaticArrays
using Statistics
using SymmetryCode
using SymmetryCode.Spectral
using WGLMakie
using Zygote
lines([1, 2, 3])

outdir = joinpath(@__DIR__, "output") |> mkpath

dns_aid()

setup = let
    l = 1.0
    n_les = 256
    Δ = 4 * l / n_les
    (; visc = 5e-5, D = 2, l = 1.0, n_dns = 1024, n_les, Δ, backend = CUDABackend())
end

data = let
    filename = joinpath(outdir, "data-$(setup.n_les).jld2")
    if false
        d = create_data(setup; nstep = 1000, nsubstep = 100, rng = Xoshiro(0), setup.Δ)
        jldsave(filename; data = d)
    end
    load(filename, "data")
end;

m_nomo = let
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    nx = space_ndrange(g)
    nt = tensordim(g)
    m_nomo(G) = fill!(similar(G.xx, nx..., nt), 0)
end

m_tbnn = let
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    net = Chain(
        Conv((1, 1), Spectral.ninvariant(g) => 64, gelu),
        Conv((1, 1), 64 => 64, gelu),
        Conv((1, 1), 64 => 128, gelu),
        Conv((1, 1), 128 => Spectral.nbasis(g); use_bias = false),
    ) # 13_056 parameters
    net |> display
    ps, st = Lux.setup(Xoshiro(0), net) |> f64 |> adapt(setup.backend)
    for l in ps
        l.weight .*= 0.1
    end
    file = joinpath(outdir, "ps-tbnn-$(setup.n_les).jld2")
    if false
        ps, st = train(;
            loss = create_loss_tbnn(g),
            setup,
            dataloader = create_dataloader_tbnn(
                setup,
                data;
                batchsize = 5,
                rng = Xoshiro(0),
            ),
            nepoch = 10,
            learning_rate = 1e-3,
            net_stuff = (; net, ps, st),
        )
        jldsave(file; ps = ps |> cpu_device())
    end
    ps = load(file, "ps") |> adapt(setup.backend)
    m_tbnn = tbnn(net, ps, st, g)
end;

m_equi = let
    net_stuff = equivariant_net(
        setup,
        [16, 24, 24, 32], # 14112 actual params
    )
    st = net_stuff.st
    file = joinpath(outdir, "ps_equi.jld2")
    if false
        ps, st = train(;
            loss = create_loss(net_stuff.project),
            setup,
            dataloader = create_dataloader(setup, data; batchsize = 5),
            nepoch = 10,
            learning_rate = 1e-3,
            net_stuff,
        )
        jldsave(file; ps = ps |> cpu_device())
    end
    ps = load(file, "ps") |> adapt(setup.backend)
    # ps |> cpu_device() |> ComponentArray |> length |> display
    (; net, project) = net_stuff
    fullchain(setup, net, project, ps, st)
end;

m_conv = let
    net_stuff = cnn(
        setup,
        [24, 64, 64, 128]; # 14_587 parameters
        same_as_equi = false,
    )
    for ps in net_stuff.ps
        # Initialize weights are too large
        hasfield(typeof(ps), :weight) && (ps.weight .*= 0.1)
    end
    st = net_stuff.st
    file = joinpath(outdir, "ps_conv.jld2")
    if false
        ps, st = train(;
            loss = create_loss(net_stuff.project),
            setup,
            dataloader = create_dataloader(setup, data; batchsize = 5),
            nepoch = 10,
            learning_rate = 1e-3,
            net_stuff,
        )
        jldsave(file; ps = ps |> cpu_device())
    end
    ps = load(file, "ps") |> adapt(setup.backend)
    (; net, project) = net_stuff
    fullchain(setup, net, project, ps, st)
end;

equi_errors_post = let
    grid = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    models = (; nomo = m_nomo, tbnn = m_tbnn, conv = m_conv, equi = m_equi)
    ustart = data[1][end] |> adapt(setup.backend)
    map(keys(models)) do key
        model = models[key]
        @info "Computing equivariance error for $(key)"
        e = test_equivariance_post(;
            ustart,
            setup,
            grid,
            model,
            groupindex = 6,
            rng = Xoshiro(123),
            tstop = 1e-1,
        )
        key => e
    end |> NamedTuple
end

equi_errors_post |> e -> map(x -> round(x; sigdigits = 4), e) |> pairs

u_dns, u_les = let
    g_dns = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    ustart = randomfield(g_dns; rng = Xoshiro(123), kpeak = 5)
    inference_post(;
        ustart,
        setup,
        models = (; nomo = m_nomo, tbnn = m_tbnn, conv = m_conv, equi = m_equi),
        setup.Δ,
        tstop = 1e-1,
    )
end;

get_errors(setup, u_les);

let
    g_dns = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    g_les = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    D = dim(g_dns)
    stat = turbulence_statistics(u_dns, setup.visc, g_dns)
    s = spectrum(u_dns, g_dns)
    s_les = map(u -> spectrum(u, g_les), u_les)
    fig = Figure()
    ax = Makie.Axis(fig[1, 1]; xscale = log10, yscale = log10)
    k = [2, g_dns.n / 8]
    if D == 2
        kolmo = @. 2e0 * stat.diss^(1 / 3) * k^(-3)
        escale = stat.diss^(-2 / 3) * stat.l_kol^(-3)
    elseif D == 3
        kolmo = @. 5e-1 * stat.diss^(2 / 3) * k^(-5 / 3)
        escale = stat.diss^(-2 / 3) * stat.l_kol^(-5 / 3)
    end
    kscale = stat.l_kol
    # lines!(ax, kscale * s.k, escale * s.s)
    # lines!(kscale * k, escale * kolmo)
    for (key, val) in pairs(s_les)
        lines!(ax, kscale * val.k, escale * val.s; label = string(key))
    end
    axislegend(ax)
    # ylims!(1e-7, 1)
    fig
end

let
    g_dns = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    g_les = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    D = dim(g_dns)
    u_ref = u_les.ref
    τhat = sfs(u_dns, g_dns, g_les, setup.Δ)
    τ = spacetensorfield(g_les)
    plan = Seneca.getplan(g_les)
    for (τ, τhat) in zip(τ, τhat)
        apply!(twothirds!, g_les, (τhat, g_les))
        ldiv!(τ, plan, τhat)
        τ .*= g_les.n^dim(g_les)
    end
    G = getgradient(u_ref, g_les)
    S = (; G.xx, G.yy, xy = (G.xy .+ G.yx) ./ 2)
    models = (; tbnn = m_tbnn, conv = m_conv, equi = m_equi)
    labels = (; ref = "Reference", tbnn = "TBNN", conv = "Conv", equi = "G-Conv")
    τ_les = map(models) do m
        y = m(G)
        if D == 2
            xx, yy, xy = 1, 2, 3
            (; xx = view(y, :, :, xx), yy = view(y, :, :, yy), xy = view(y, :, :, xy))
        elseif D == 3
            error()
        end
    end
    τ_all = (; ref = τ, τ_les...)
    τxx = map(τ -> τ.xx, τ_all)
    τyy = map(τ -> τ.yy, τ_all)
    τxy = map(τ -> τ.xy, τ_all)
    diss = map(τ_all) do τ
        @. τ.xx * S.xx + τ.yy * S.yy + 2 * τ.xy * S.xy
    end
    # τ.xx |> x -> plan \ x |> display; error()
    # τ.xx |> display; error()
    # map(mean, τxx) |> pairs |> display; error()
    # τ.xx |> Array |> heatmap |> display; error()
    # τ.xx |> Array |> display
    # error()
    fig = Figure(; size = (700, 300))
    #
    ax_xx =
        Makie.Axis(fig[1, 1]; xlabel = "xx-component", ylabel = "Density", yscale = log10)
    τxx = map(d -> kde(d |> vec |> Array) |> x -> (; x.x, x.density), τxx)
    for (key, val) in pairs(τxx)
        lines!(ax_xx, val.x / setup.Δ, val.density; label = labels[key])
    end
    ylims!(ax_xx, 3e-2, 2e2)
    # xlims!(ax_xx, -0.1, 0.4)
    #
    ax_xy = Makie.Axis(fig[1, 2]; xlabel = "xy-component", yscale = log10)
    τxy = map(d -> kde(d |> vec |> Array) |> x -> (; x.x, x.density), τxy)
    for (key, val) in pairs(τxy)
        lines!(ax_xy, val.x / setup.Δ, val.density)
    end
    ylims!(ax_xy, 5e-2, 1e2)
    # xlims!(ax_xy, -0.2, 0.12)
    #
    ax_diss = Makie.Axis(fig[1, 3]; xlabel = "Dissipation", yscale = log10)
    diss = map(d -> kde(d |> vec |> Array) |> x -> (; x.x, x.density), diss)
    for (key, val) in pairs(diss)
        lines!(
            ax_diss,
            # val.x,
            val.x / setup.Δ,
            val.density,
        )
    end
    ylims!(ax_diss, 1e-1, 7e1)
    xlims!(ax_diss, -20, 20)
    Legend(fig[0, :], ax_xx; orientation = :horizontal)
    fig
end

let
    g = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    u = u_dns
    s = turbulence_statistics(u, setup.visc, g)
    s |> pairs
end
