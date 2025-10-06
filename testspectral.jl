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

# setup = setup_laptop()
setup = setup_turbulator()
# setup = setup_snellius()
# plotdir = "~/Projects/SymmetryPaper/figures" |> expanduser |> mkpath
plotdir = setup.outdir
# plotdir = joinpath(setup.outdir, "snelliusplots") |> mkpath

let
    s = group_stuff(3)
    # map(display, s.mats)
    m = @SMatrix [
        -1 0 0
        0 1 0
        0 0 -1
    ]
    findfirst(isequal(m), s.mats)
end

let
    m = @SMatrix [
        -1 0 0
        0 -1 0
        0 0 -1
    ]
    x = @SMatrix [
        11 12 13
        21 22 23
        31 32 33
    ]
    m * x * m'
end

let
    l = 10.0
    Δ = l / 20
    n = 512
    x = range(-l / 2, l / 2, n + 1)[1:(end-1)]
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
    save("$(plotdir)/filterkernel.pdf", fig; backend = CairoMakie)
    fig
end

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

dnsfile = joinpath(setup.outdir, "dns.jld2")

let
    dns, times, energies = create_dns(setup; t_warmup = 0.5, cfl = 0.35, rng = Xoshiro(0))
    jldsave(dnsfile; u = cpu_device()(dns), times, energies)
end

let
    times, energies = load(dnsfile, "times", "energies")
    fig, ax, l = lines(times, energies)
    save(joinpath(plotdir, "energy.pdf"), fig)
    fig
end

# Plot DNS spectrum
let
    (; D, l, n_dns, n_les, backend, visc, ou_radius) = setup
    u = load(dnsfile, "u") |> adapt(backend)
    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)
    ubar = vectorfield(g_les)
    for (ubar, u) in zip(ubar, u)
        apply!(cutoff!, g_les, (ubar, u))
        apply!(gaussianfilter!, g_les, (ubar, setup.Δ, g_les))
    end
    D = dim(g_dns)
    stat = turbulence_statistics(u, visc, g_dns)
    @show stat.Re_tay
    s_dns = spectrum(u, g_dns)
    s_les = spectrum(ubar, g_les)
    fig = Figure(; size = (400, 340))
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        xlabel = "Normalized wavenumber",
        ylabel = "Normalized spectrum",
    )
    if D == 2
        k = [3, g_dns.n / 10]
        kolmo = @. stat.diss^(-1 / 3) * k^(-3)
        escale = stat.diss^(-1 / 3) * stat.l_kol^(-3)
    elseif D == 3
        k = [3, g_dns.n / 8]
        kolmo = @. 6.5e-1 * stat.diss^(2 / 3) * k^(-5 / 3)
        escale = stat.diss^(-2 / 3) * stat.l_kol^(-5 / 3)
        # escale = 1
    end
    kscale = stat.l_kol / l
    span = [1, ou_radius] * kscale
    oucolor = Makie.wong_colors()[4]
    vspan!(ax, span...; alpha = 0.3, color = oucolor)
    b = sqrt(prod(extrema(escale * s_dns.s)))
    a = 1.1 * kscale * ou_radius
    c = sqrt(prod(span))
    w = D == 2 ? 1 : 1.5
    text!(ax, a, b / w; color = oucolor, text = "Force")
    arr = D == 2 ? 100 : 5
    arrows2d!(
        ax,
        Point2(c, b / arr),
        Point2(c, b * arr) - Point2(c, b / arr);
        color = oucolor,
    )
    lines!(ax, kscale * s_dns.k, escale * s_dns.s; label = "DNS")
    lines!(ax, kscale * s_les.k, escale * s_les.s; label = "Filtered DNS")
    lines!(kscale * k, escale * kolmo; label = "Kolmogorov")
    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        tellheight = true,
        framevisible = false,
        horizontal = true,
        nbanks = 4,
    )
    rowgap!(fig.layout, 5)
    save("$(plotdir)/spectrum-dns.pdf", fig; backend = CairoMakie)
    fig
end

data, datatiming = let
    filename = joinpath(setup.outdir, "data.jld2")
    if true
        t = time()
        u = load(dnsfile, "u") |> adapt(setup.backend)
        d = create_data(
            u,
            setup;
            cfl = 0.35,
            nstep = setup.D == 2 ? 1000 : 50,
            nsubstep = 10,
            setup.Δ,
        )
        t = time() - t
        jldsave(filename; data = d, timing = t)
    end
    load(filename, "data", "timing")
end;

data[2][end] |> typeof

let
    n = setup.n_les
    x = zeros(n, n, n)
    plan = plan_rfft(x)
    y = data[2][2].xx
    ldiv!(x, plan, copy(y))
    x .*= n^setup.D
    @show sum(>(0), x) / length(x)
    val = kde(x |> vec |> Array) |> x -> (; x.x, x.density)
    fig, ax, l = lines(val.x, val.density; axis = (yscale = log10,))
    ylims!(ax, 1e-3, 16e1)
    xlims!(ax, -0.1, 0.2)
    fig
end

Base.summarysize(data) * 1e-9

m_nomo = let
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    nx = space_ndrange(g)
    nt = tensordim(g)
    m_nomo(G) = fill!(stack(spacetensorfield(g)), 0)
end

m_smag = create_smagorinsky(
    0.17,
    setup.Δ,
    Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend),
)

m_clar = create_clark(setup.Δ, Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend))

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
    file = joinpath(setup.outdir, "ps-tbnn.jld2")
    if true
        t = time()
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
        t = time() - t
        ps = ps |> cpu_device()
        jldsave(file; ps, losses_train, losses_valid, timing = t)
    end
    ps, losses_train, losses_valid, timing =
        load(file, "ps", "losses_train", "losses_valid", "timing")
    ps = ps |> adapt(setup.backend)
    chain = tbnn(net, ps, st, g)
    chain, (; losses_train, losses_valid, timing)
end;

m_equi, train_equi = let
    net_stuff = equivariant_net(
        setup,
        # [12, 16, 16, 24], # 40_328 actual params
        [8, 8, 8, 16], # 12_544 actual params
    )
    st = net_stuff.st
    file = joinpath(setup.outdir, "ps-equi.jld2")
    if true
        t = time()
        (; ps, st, losses_train, losses_valid) = train(;
            loss = create_loss(net_stuff.project),
            setup,
            dataloader = create_dataloader(setup, data; batchsize = 10),
            nepoch = 5,
            learning_rate = 1e-3,
            net_stuff,
        )
        t = time() - t
        ps = ps |> cpu_device()
        jldsave(file; ps, losses_train, losses_valid, timing = t)
    end
    ps, losses_train, losses_valid, timing =
        load(file, "ps", "losses_train", "losses_valid", "timing")
    # ps = net_stuff.ps
    ps = ps |> adapt(setup.backend)
    ps |> cpu_device() |> ComponentArray |> length |> display
    (; net, project) = net_stuff
    chain = fullchain(setup, net, project, ps, st)
    chain, (; losses_train, losses_valid, timing)
end;

m_conv, train_conv = let
    net_stuff = cnn(
        setup,
        # [48, 128, 128, 128]; # 40_550 parameters
        [48, 64, 64, 64]; # 12_326 parameters
        same_as_equi = false,
    )
    for ps in net_stuff.ps
        # Initialize weights are too large
        hasfield(typeof(ps), :weight) && (ps.weight .*= 0.1)
    end
    st = net_stuff.st
    file = joinpath(setup.outdir, "ps-conv.jld2")
    if false
        t = time()
        (; ps, st, losses_train, losses_valid) = train(;
            loss = create_loss(net_stuff.project),
            setup,
            dataloader = create_dataloader(setup, data; batchsize = 10),
            nepoch = 5,
            learning_rate = 1e-3,
            net_stuff,
        )
        ps = ps |> cpu_device()
        t = time() - t
        jldsave(file; ps, losses_train, losses_valid, timing = t)
    end
    # ps = net_stuff.ps
    ps, losses_train, losses_valid, timing =
        load(file, "ps", "losses_train", "losses_valid", "timing")
    ps = ps |> adapt(setup.backend)
    ps |> cpu_device() |> ComponentArray |> length |> display
    (; net, project) = net_stuff
    chain = fullchain(setup, net, project, ps, st)
    chain, (; losses_train, losses_valid, timing)
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
    # t = train_tbnn
    t = train_conv
    # t = train_equi
    lines!(ax, t.losses_train; label = "Train")
    lines!(ax, t.losses_valid; label = "Valid")
    axislegend(ax; position = :rt)
    fig
end

equi_errors_post = let
    grid = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    models = (;
        # nomo = m_nomo,
        # smag = m_smag,
        # clar = m_clar,
        # tbnn = m_tbnn,
        # equi = m_equi,
        conv = m_conv,
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

let
    filename = joinpath(setup.outdir, "equi-errors-post.jld2")
    jldsave(filename; equi_errors_post)
end

equi_errors_post = let
    filename = joinpath(setup.outdir, "equi-errors-post.jld2")
    load(filename, "equi_errors_post")
end

equi_errors_post |> e -> map(x -> round(x; sigdigits = 4), e) |> pairs

# :nomo => 1.418e-15
# :smag => 9.936e-16
# :clar => 1.451e-15
# :tbnn => 1.132e-15
# :equi => 1.162e-15
# :conv => 0.01899

upostfiles = map(
    name -> joinpath(setup.outdir, "u-post-$(name).jld2"),
    (;
        dns = "dns",
        ref = "ref",
        # nomo = "nomo",
        # smag = "smag",
        clar = "clar",
        clar2 = "clar2",
        clar22 = "clar22",
        # tbnn = "tbnn",
        # equi = "equi",
        # conv = "conv",
    ),
)
let
    models = (;
        # nomo = m_nomo,
        # smag = m_smag,
        clar = m_clar,
        clar2 = x -> sqrt(2) * m_clar(x),
        clar22 = x -> 2 * m_clar(x),
        # tbnn = m_tbnn,
        # equi = m_equi,
        # conv = m_conv,
    )
    u_dns = load(dnsfile, "u") |> adapt(setup.backend)
    inference_post(;
        u_dns,
        setup,
        models,
        files = upostfiles,
        cfl = 0.35,
        tstop = 1e-1,
        setup.Δ,
    )
end

map(f -> load(f, "timing"), upostfiles) |> pairs

# :dns  => 1025.77
# :ref  => 1025.77
# :nomo => 1.62064
# :smag => 0.904956
# :clar => 0.980243
# :tbnn => 6.24895
# :equi => 40.2744
# :conv => 5.49185

u = map(f -> load(f, "u"), upostfiles);
u = map(f -> load(f, "u"), upostfiles[(:dns, :ref, :conv)]);

get_errors(setup, u);

# :nomo => 0.1752
# :smag => 0.1223
# :clar => 0.157
# :tbnn => 0.1222
# :equi => 0.1171
# :conv => 0.1175

let
    (; D, l, n_dns, n_les, backend, visc) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)
    labels = (;
        dns = "DNS",
        ref = "Filtered DNS",
        nomo = "No-model",
        smag = "Smagorinsky",
        clar = "Clark",
        tbnn = "TBNN",
        equi = "G-Conv",
        conv = "Conv",
    )
    u_dns = u.dns
    keys_les = filter(!=(:dns), keys(u))
    u_les = (; map(k -> k => u[k], keys_les)...)
    D = dim(g_dns)
    stat = turbulence_statistics(u_dns |> adapt(backend), visc, g_dns)
    stat |> pairs |> display
    s = spectrum(u_dns |> adapt(backend), g_dns)
    s_les = map(u -> spectrum(u |> adapt(backend), g_les), u_les)
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
    save("$(plotdir)/spectrum-les.pdf", fig; backend = CairoMakie)
    fig
end

# :uavg   => 1.05459
# :diss   => 0.499842
# :l_int  => 2.3465
# :l_tay  => 0.00943405
# :l_kol  => 0.000598187
# :t_int  => 2.22503
# :t_tay  => 0.00894569
# :t_kol  => 0.00894569
# :Re_int => 61865.1
# :Re_tay => 248.727
# :Re_kol => 15.7711

let
    # setup = (; Main.setup..., Δ = Main.setup.Δ * 2)
    m = create_clark(setup.Δ, Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend))
    a = sqrt(2)
    # a = 1.0
    # b = sqrt(2)
    b = a
    w = reshape([a, a, a, b, b, b], 1,1,1,:) |> CuArray
    models = (;
        # smag = m_smag,
        clar = m_clar,
        # clar2 = x -> 1.8 * m_clar(x) .+ 0.01 * o,
        # clar2 = x -> sqrt(2) * m(x),
        # clar2 = x -> w .* m(x),
        # tbnn = m_tbnn,
        # equi = m_equi,
        conv = m_conv,
        # conv = x -> sqrt(2) * m_conv(x),
    )
    labels = (;
        ref = "Reference",
        smag = "Smagorinsky",
        clar = "Clark",
        clar2 = "Clark2",
        tbnn = "TBNN",
        equi = "G-Conv",
        conv = "Conv",
    )
    u_dns = load(dnsfile, "u") |> adapt(setup.backend)
    plot_densities(; u_dns, setup, models, labels, plotdir, dolog = true)
end

let
    k = m_conv.ps.sink.weight[1,1,1,:,1] |> Array |> kde
    lines(k.x, k.density)
end

m_conv.ps.sink.bias

apriori_errors = let
    models = (;
        nomo = m_nomo,
        smag = m_smag,
        clar = m_clar,
        # tbnn = m_tbnn,
        # equi = m_equi,
        conv = m_conv,
    )
    labels = (;
        nomo = "No-model",
        tbnn = "TBNN",
        conv = "Conv",
        equi = "G-Conv",
        smag = "Smagorinsky",
        clar = "Clark",
    )
    u_dns = load(dnsfile, "u") |> adapt(setup.backend)
    apriori_error(; u_dns, setup, models, labels, plotdir)
end

apriori_errors |> e -> map(x -> round(x; sigdigits = 4), e) |> pairs

# :nomo => 1.0
# :smag => 0.9922
# :clar => 0.7288
# :tbnn => 0.6543
# :equi => 0.6319
# :conv => 0.6319

apriori_equi_errors = let
    models = (; smag = m_smag, clar = m_clar, tbnn = m_tbnn, equi = m_equi, conv = m_conv)
    apriori_equivariance_error(; u, setup, models, labels, plotdir, groupindex = 6)
end

let
    (; dets) = group_stuff(setup.D)
    fig = Figure(; size = (500, 400))
    ax = Axis(
        fig[1, 1];
        yscale = log10,
        xlabel = "Group element",
        ylabel = "Error",
        xticks = [1, 8, 16, 24, 32, 40, 48],
    )
    ylims!(ax, 1e-17, 1)
    i = 1:48
    colors = (;
        smag = Cycled(2),
        clar = Cycled(3),
        tbnn = Cycled(4),
        equi = Cycled(5),
        conv = Cycled(6),
    )
    labels = (;
        smag = "Smagorinsky",
        clar = "Clark",
        tbnn = "TBNN",
        equi = "G-Conv",
        conv = "Conv",
    )
    markers =
        (; smag = :circle, clar = :rect, tbnn = :diamond, equi = :rtriangle, conv = :x)
    for key in keys(apriori_equi_errors)
        e = apriori_equi_errors[key]
        e = max.(e, 1e-30) # Encode true zeros as 1e-30
        scatterlines!(
            ax,
            i,
            e;
            label = labels[key],
            marker = markers[key],
            color = colors[key],
        )
    end
    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        tellheight = true,
        framevisible = false,
        horizontal = true,
        nbanks = 5,
    )
    rowgap!(fig.layout, 5)
    save("$(plotdir)/apriori-equi-errors.pdf", fig; backend = CairoMakie)
    fig
end

let
    s = group_stuff(3)
    s.mats[43]
end

apriori_equi_errors.conv[43]

apriori_equi_errors |> e -> map(x -> round(mean(x); sigdigits = 4), e) |> pairs

# :smag => 7.244e-17
# :clar => 7.201e-17
# :tbnn => 2.557e-16
# :equi => 6.712e-16
# :conv => 0.05898

dissipation_errors = let
    u_dns = u.dns |> adapt(setup.backend)
    models = (;
        nomo = m_nomo,
        smag = m_smag,
        clar = m_clar,
        tbnn = m_tbnn,
        conv = m_conv,
        equi = m_equi,
    )
    labels = (;
        nomo = "No-model",
        tbnn = "TBNN",
        conv = "Conv",
        equi = "G-Conv",
        smag = "Smagorinsky",
        clar = "Clark",
    )
    get_dissipation_errors(; setup, u_dns, models)
end;
dissipation_errors |> e -> map(x -> round(x; sigdigits = 4), e) |> pairs

# :ref  => -0.1498
# :nomo => 0.0
# :smag => -0.1959
# :clar => -0.05622
# :tbnn => -0.2081
# :conv => -0.1714
# :equi => -0.1765

let
    g = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    u = u_dns
    s = turbulence_statistics(u, setup.visc, g)
    s |> pairs
end

let
    comp = :x
    fig = plot_velocities(setup, u, comp)
    save("$(plotdir)/velocities-$(comp).png", fig; backend = CairoMakie)
    fig
end
