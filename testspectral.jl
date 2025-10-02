if false
    include("src/SymmetryCode.jl")
    using .SymmetryCode
    using .SymmetryCode.Spectral
end

using Adapt
using CairoMakie
using ComponentArrays: ComponentArray
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

let
    l = 10.0
    Δ = l / 20
    n = 512
    x = range(-l / 2, l / 2, n + 1)[1:end-1]
    y = @. sqrt(6 / pi / Δ^2) * exp(-6 * x^2 / Δ^2)
    yhat = rfft(y) * l / n
    kmax = div(n, 2)
    k = 0:kmax
    yref = @. exp(-Δ^2 * (2 * pi * k / l)^2 / 24)
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10)
    lines!(ax, 2 * pi / l * k[2:end], abs.(yhat[2:end]); label = "FFT")
    lines!(ax, 2 * pi / l * k[2:end], abs.(yref[2:end]); label = "Theory")
    vlines!(ax, 2 * pi / Δ)
    vlines!(ax, 2 * pi / l)
    ylims!(1e-10, 1e2)
    fig
end

outdir = joinpath(@__DIR__, "output") |> mkpath
# plotdir = "~/Projects/SymmetryPaper/figures" |> expanduser |> mkpath
plotdir = outdir

dns_aid()

# setup = let
#     l = 1.0
#     n_les = 128
#     Δ = 2 * l / n_les
#     (; visc = 5e-5, D = 2, l = 1.0, n_dns = 1024, n_les, kpeak = 5, Δ, backend = CUDABackend())
# end

let
    l = 1.0
    n_les = 64
    ncut = 2 * div(n_les, 3)
    Δ = 2 * l / n_les
    k = 2 * pi / l * (1:32)
    w = @. exp(-Δ^2 * k^2 / 24)
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10)
    kol = k .^ (-5 / 3)
    lines!(ax, k, kol)
    lines!(ax, k, w .* kol)
    vlines!(ax, 2 * pi / l * div(n_les, 3))
    fig
end

setup = let
    l = 1.0
    n_les = 64
    Δ = 2 * l / n_les
    (;
        visc = 1e-4,
        D = 3,
        l = 1.0,
        n_dns = 256,
        n_les,
        kpeak = 5,
        Δ,
        backend = CUDABackend(),
    )
end

data = let
    # filename = joinpath(outdir, "data-$(setup.n_les).jld2")
    filename = joinpath(outdir, "data-$(setup.D)D-gaussian-$(setup.n_les).jld2")
    if false
        d = create_data(
            setup;
            t_warmup = 0.5,
            cfl = 0.35,
            nstep = 30,
            nsubstep = 20,
            setup.kpeak,
            rng = Xoshiro(0),
            setup.Δ,
        )
        jldsave(filename; data = d)
    end
    load(filename, "data")
end;

data[2][end] |> typeof

m_nomo = let
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    nx = space_ndrange(g)
    nt = tensordim(g)
    τ = fill!(stack(spacetensorfield(g)), 0)
    m_nomo(G) = τ
end

m_tbnn, train_tbnn = let
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    kern = ntuple(Returns(1), setup.D)
    net = Chain(
        Conv(kern, Spectral.ninvariant(g) => 64, gelu),
        Conv(kern, 64 => 64, gelu),
        Conv(kern, 64 => 128, gelu),
        Conv(kern, 128 => Spectral.nbasis(g); use_bias = false),
    ) # 13_056 parameters
    net |> display
    ps, st = Lux.setup(Xoshiro(0), net) |> f64 |> adapt(setup.backend)
    for l in ps
        l.weight .*= 0.1
    end
    file = joinpath(outdir, "ps-tbnn-$(setup.n_les).jld2")
    if false
        (; ps, st, losses_train, losses_valid) = train(;
            loss = create_loss_tbnn(g),
            setup,
            dataloader = create_dataloader_tbnn(
                setup,
                data;
                batchsize = 10,
                rng = Xoshiro(0),
            ),
            nepoch = 10,
            learning_rate = 1e-3,
            net_stuff = (; net, ps, st),
        )
        ps = ps |> cpu_device()
        jldsave(file; ps, losses_train, losses_valid)
    end
    ps, losses_train, losses_valid = load(file, "ps", "losses_train", "losses_valid")
    ps = ps |> adapt(setup.backend)
    chain = tbnn(net, ps, st, g)
    chain, (; losses_train, losses_valid)
end;

m_equi, train_equi = let
    net_stuff = equivariant_net(
        setup,
        [12, 16, 16, 24], # 40_328 actual params
    )
    st = net_stuff.st
    file = joinpath(outdir, "ps_equi.jld2")
    if false
        (; ps, st, losses_train, losses_valid) = train(;
            loss = create_loss(net_stuff.project),
            setup,
            dataloader = create_dataloader(setup, data; batchsize = 10),
            nepoch = 1,
            learning_rate = 1e-3,
            net_stuff,
        )
        ps = ps |> cpu_device()
        jldsave(file; ps, losses_train, losses_valid)
    end
    ps, losses_train, losses_valid = load(file, "ps", "losses_train", "losses_valid")
    # ps = net_stuff.ps
    ps = ps |> adapt(setup.backend)
    # ps |> cpu_device() |> ComponentArray |> length |> display; error()
    (; net, project) = net_stuff
    chain = fullchain(setup, net, project, ps, st)
    chain, (; losses_train, losses_valid)
end;

m_conv, train_conv = let
    net_stuff = cnn(
        setup,
        [48, 128, 128, 128]; # 40_550 parameters
        same_as_equi = false,
    )
    for ps in net_stuff.ps
        # Initialize weights are too large
        hasfield(typeof(ps), :weight) && (ps.weight .*= 0.1)
    end
    st = net_stuff.st
    file = joinpath(outdir, "ps_conv.jld2")
    if false
        (; ps, st, losses_train, losses_valid) = train(;
            loss = create_loss(net_stuff.project),
            setup,
            dataloader = create_dataloader(setup, data; batchsize = 10),
            nepoch = 10,
            learning_rate = 1e-3,
            net_stuff,
        )
        ps = ps |> cpu_device()
        jldsave(file; ps, losses_train, losses_valid)
    end
    # ps = net_stuff.ps
    ps, losses_train, losses_valid = load(file, "ps", "losses_train", "losses_valid")
    ps = ps |> adapt(setup.backend)
    # ps |> cpu_device() |> ComponentArray |> length |> display; error()
    (; net, project) = net_stuff
    chain = fullchain(setup, net, project, ps, st)
    chain, (; losses_train, losses_valid)
end;

let
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        # xscale = log10,
        # yscale = log10,
        xlabel = "Iteration",
        ylabel = "Loss",
    )
    t = train_tbnn
    # t = train_conv
    # t = train_equi
    lines!(ax, t.losses_train; label = "Train")
    lines!(ax, t.losses_valid; label = "Valid")
    axislegend(ax; position = :rt)
    fig
end

m_smag = create_smagorinsky(
    0.17,
    setup.Δ,
    Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend),
)

m_clar = create_clark(setup.Δ, Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend))

equi_errors_post = let
    grid = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    models = (;
        nomo = m_nomo,
        tbnn = m_tbnn,
        conv = m_conv,
        equi = m_equi,
        smag = m_smag,
        clar = m_clar,
    )
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

equi_errors_post = let
    filename = joinpath(outdir, "equi-errors-post-$(setup.D)D-$(setup.n_les).jld2")
    # jldsave(filename; equi_errors_post)
    load(filename, "equi_errors_post")
end

equi_errors_post |> e -> map(x -> round(x; sigdigits = 4), e) |> pairs

# :nomo => 9.999e-16
# :tbnn => 9.382e-16
# :conv => 0.0165
# :equi => 9.01e-16

u_dns, u_les = let
    g_dns = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    ustart = randomfield(g_dns; rng = Xoshiro(123), setup.kpeak)
    models = (;
        nomo = m_nomo,
        tbnn = m_tbnn,
        conv = m_conv,
        equi = m_equi,
        smag = m_smag,
        clar = m_clar,
    )
    inference_post(; ustart, setup, models, setup.Δ, tstop = 1e-1, cfl = 0.35)
end;

u_dns, u_les = let
    filename = joinpath(outdir, "u-post-$(setup.D)D-$(setup.n_les).jld2")
    # jldsave(filename; u_dns = u_dns |> cpu_device(), u_les = u_les |> cpu_device())
    load(filename, "u_dns", "u_les") |> adapt(setup.backend)
end;

get_errors(setup, u_les);

# :nomo => 0.4443
# :tbnn => 0.3037
# :conv => 0.2488
# :equi => 0.259
# :smag => 0.3598
# :clar => 0.272

let
    g_dns = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    g_les = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    labels = (;
        ref = "Filtered DNS",
        nomo = "No-model",
        tbnn = "TBNN",
        conv = "Conv",
        equi = "G-Conv",
        smag = "Smagorinsky",
        clar = "Clark",
    )
    D = dim(g_dns)
    stat = turbulence_statistics(u_dns, setup.visc, g_dns)
    stat |> pairs |> display
    s = spectrum(u_dns, g_dns)
    s_les = map(u -> spectrum(u, g_les), u_les)
    fig = Figure(; size = (500, 300))
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
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
    # axislegend(ax; position = :lb)
    Legend(fig[1, 2], ax)
    # ylims!(1e-7, 1)
    save("$(plotdir)/spectrum-$(D)D-$(setup.n_les).pdf", fig; backend = CairoMakie)
    fig
end

let
    models = (; tbnn = m_tbnn, conv = m_conv, equi = m_equi, smag = m_smag, clar = m_clar)
    labels = (;
        ref = "Reference",
        tbnn = "TBNN",
        conv = "Conv",
        equi = "G-Conv",
        smag = "Smagorinsky",
        clar = "Clark",
    )
    plot_densities(setup, u_dns, u_les, models, labels, plotdir)
end

apriori_errors = let
    models = (;
        nomo = m_nomo,
        tbnn = m_tbnn,
        conv = m_conv,
        equi = m_equi,
        smag = m_smag,
        clar = m_clar,
    )
    labels = (;
        nomo = "No-model",
        tbnn = "TBNN",
        conv = "Conv",
        equi = "G-Conv",
        smag = "Smagorinsky",
        clar = "Clark",
    )
    apriori_error(setup, u_dns, u_les, models, labels, plotdir)
end

apriori_errors |> e -> map(x -> round(x; sigdigits = 4), e) |> pairs

apriori_equi_errors = let
    models = (; tbnn = m_tbnn, conv = m_conv, equi = m_equi, smag = m_smag, clar = m_clar)
    labels = (;
        tbnn = "TBNN",
        conv = "Conv",
        equi = "G-Conv",
        smag = "Smagorinsky",
        clar = "Clark",
    )
    apriori_equivariance_error(; setup, u_les, models, labels, plotdir, groupindex = 6)
end

apriori_equi_errors |> e -> map(x -> round(x; sigdigits = 4), e) |> pairs

let
    g = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    u = u_dns
    s = turbulence_statistics(u, setup.visc, g)
    s |> pairs
end

let
    comp = :x
    fig = plot_velocities(setup, u_dns, u_les, comp)
    save("$(plotdir)/velocities-$(comp)-$(setup.D)D-$(setup.n_les).pdf", fig; backend = CairoMakie)
    fig
end
