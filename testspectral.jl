if false
    include("src/SymmetryCode.jl")
    using .SymmetryCode
    using .SymmetryCode.Spectral
end

using Adapt
using CairoMakie
using CUDA, cuDNN
using FFTW
using JLD2
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
lines([1, 2, 3])

outdir = joinpath(@__DIR__, "output") |> mkpath
# plotdir = "~/Projects/SymmetryPaper/figures" |> expanduser |> mkpath
plotdir = outdir

dns_aid()

setup = let
    l = 1.0
    n_les = 256
    Δ = 4 * l / n_les
    (; visc = 5e-5, D = 2, l = 1.0, n_dns = 1024, n_les, kpeak = 5, Δ, backend = CUDABackend())
end

data = let
    filename = joinpath(outdir, "data-$(setup.n_les).jld2")
    if false
        d = create_data(setup; nstep = 1000, nsubstep = 100, setup.kpeak, rng = Xoshiro(0), setup.Δ)
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
    groupindex = 6
    (; mats, dets) = group_stuff(setup.D)
    m = mats[groupindex]
    d = dets[groupindex]
    @info "Roto-reflection matrix $(m) (with determinant $(d))"
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
            cfl = 0.35,
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
    labels = (; ref = "Filtered DNS", nomo = "No-model", tbnn = "TBNN", conv = "Conv", equi = "G-Conv")
    D = dim(g_dns)
    stat = turbulence_statistics(u_dns, setup.visc, g_dns)
    s = spectrum(u_dns, g_dns)
    s_les = map(u -> spectrum(u, g_les), u_les)
    fig = Figure(; size = (400, 300))
    ax = Axis(
        fig[1, 1]; xscale = log10, yscale = log10,
        xlabel = "Normalized wavenumber",
        ylabel = "Normalized spectrum",
    )
    k = [2, g_dns.n / 8]
    if D == 2
        kolmo = @. 2e0 * stat.diss^(1 / 3) * k^(-3)
        escale = stat.diss^(-2 / 3) * stat.l_kol^(-3)
    elseif D == 3
        kolmo = @. 5e-1 * stat.diss^(2 / 3) * k^(-5 / 3)
        escale = stat.diss^(-2 / 3) * stat.l_kol^(-5 / 3)
    end
    kscale = stat.l_kol
    # kscale = 1
    lines!(ax, kscale * s.k, escale * s.s; label = "DNS")
    # lines!(kscale * k, escale * kolmo)
    for (key, val) in pairs(s_les)
        lines!(ax, kscale * val.k, escale * val.s; label = labels[key])
    end
    axislegend(ax; position = :lb)
    # ylims!(1e-7, 1)
    save("$(plotdir)/spectrum-$(setup.n_les).pdf", fig; backend = CairoMakie)
    fig
end

let
    models = (; tbnn = m_tbnn, conv = m_conv, equi = m_equi)
    labels = (; ref = "Reference", tbnn = "TBNN", conv = "Conv", equi = "G-Conv")
    plot_densities(setup, u_dns, u_les, models, labels, plotdir)
end

let
    g = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    u = u_dns
    s = turbulence_statistics(u, setup.visc, g)
    s |> pairs
end
