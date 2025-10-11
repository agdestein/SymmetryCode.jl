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
# lines([1, 2, 3])

# setup = setup_laptop()
setup = setup_turbulator()
# setup = setup_snellius()

create_dns(setup; t_warmup = 0.5, cfl = 0.35, rng = Xoshiro(0))

let
    times, energies = load(dnsfile, "times", "energies")
    fig, ax, l = lines(times, energies)
    save(joinpath(setup.plotdir, "energy.pdf"), fig)
    fig
end

# Plot DNS spectrum
plot_spectrum_dns(setup)

data, datatiming = let
    filename = joinpath(setup.outdir, "data.jld2")
    if false
        t = time()
        u = load(dnsfile, "u") |> adapt(setup.backend)
        d = create_data(
            u,
            setup;
            cfl = 0.35,
            nstep = setup.D == 2 ? 1000 : 30,
            nsubstep = 10,
            setup.Δ,
        )
        t = time() - t
        jldsave(filename; data = d, timing = t)
    end
    load(filename, "data", "timing")
end;

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
    ) # 13_888 parameters
    net |> display
    ps, st = Lux.setup(Xoshiro(0), net) |> f64 |> adapt(setup.backend)
    for l in ps
        l.weight .*= 0.1
    end
    file = joinpath(setup.outdir, "ps-tbnn.jld2")
    if false
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
            nepoch = 5,
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
    fig = Figure(; size = (400, 340))
    ax = Axis(
        fig[1, 1];
        # xscale = log10,
        # yscale = log10,
        xlabel = "Iteration",
        ylabel = "Loss",
    )
    lines!(ax, train_tbnn.losses_valid; label = "TBNN")
    lines!(ax, train_equi.losses_valid; label = "G-Conv")
    lines!(ax, train_conv.losses_valid; label = "Conv")
    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        tellheight = true,
        framevisible = false,
        horizontal = true,
        nbanks = 3,
    )
    rowgap!(fig.layout, 5)
    save("$(setup.plotdir)/training.pdf", fig; backend = CairoMakie)
    fig
end

map(
    t -> round(t; digits = 1),
    (; tbnn = train_tbnn.timing, conv = train_conv.timing, equi = train_equi.timing),
) |> pairs

upostfiles = map(
    name -> joinpath(setup.outdir, "u-post-$(name).jld2"),
    (;
        dns = "dns",
        ref = "ref",
        nomo = "nomo",
        smag = "smag",
        clar = "clar",
        tbnn = "tbnn",
        equi = "equi",
        conv = "conv",
    ),
)

let
    models = (;
        nomo = m_nomo,
        smag = m_smag,
        clar = m_clar,
        tbnn = m_tbnn,
        equi = m_equi,
        conv = m_conv,
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

map(f -> load(f, "timing"), upostfiles) |> t -> map(x -> round(x; digits = 1), t) |> pairs

# :dns  => 1025.77
# :ref  => 1025.77
# :nomo => 1.62064
# :smag => 0.904956
# :clar => 0.980243
# :tbnn => 6.24895
# :equi => 40.2744
# :conv => 5.49185

u = map(f -> load(f, "u"), upostfiles);

get_errors(setup, u);

# :nomo => 0.1752
# :smag => 0.1223
# :clar => 0.157
# :tbnn => 0.1222
# :equi => 0.1171
# :conv => 0.1175

# Plot LES spectrum
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
    fig = Figure(; size = (400, 360))
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
    # lines!(ax, kscale * s.k, escale * s.s; label = "DNS")
    # lines!(kscale * k, escale * kolmo)
    for (key, val) in pairs(s_les)
        lines!(ax, kscale * val.k, escale * val.s; label = labels[key])
    end
    # axislegend(ax; position = :lb)
    Legend(
        fig[0, :],
        ax;
        tellwidth = false,
        tellheight = true,
        framevisible = false,
        horizontal = true,
        nbanks = 3,
    )
    rowgap!(fig.layout, 5)
    # ylims!(1e-7, 1)
    save("$(setup.plotdir)/spectrum-les.pdf", fig; backend = CairoMakie)
    fig
end

let
    models = (; smag = m_smag, clar = m_clar, tbnn = m_tbnn, equi = m_equi, conv = m_conv)
    labels = (;
        ref = "Reference",
        smag = "Smagorinsky",
        clar = "Clark",
        tbnn = "TBNN",
        equi = "G-Conv",
        conv = "Conv",
    )
    u_dns = load(dnsfile, "u") |> adapt(setup.backend)
    plot_densities(; u_dns, setup, models, labels, dolog = true)
end

prediction_error_prior_file = joinpath(setup.outdir, "prediction-error-prior.jld2")

let
    models = (;
        nomo = m_nomo,
        smag = m_smag,
        clar = m_clar,
        tbnn = m_tbnn,
        equi = m_equi,
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
    e = apriori_error(; u_dns, setup, models, labels, setup.plotdir)
    save_object(prediction_error_prior_file, e)
end

prediction_error_prior = load_object(prediction_error_prior_file)

prediction_error_prior |> e -> map(x -> round(x.relerr; sigdigits = 4), e) |> pairs
prediction_error_prior |> e -> map(x -> round(x.crosscor; sigdigits = 4), e) |> pairs

##############################
# A-priori equivariance errors
##############################

equi_errors_prior_file = joinpath(setup.outdir, "equi-errors-prior.jld2")

let
    models = (; smag = m_smag, clar = m_clar, tbnn = m_tbnn, equi = m_equi, conv = m_conv)
    errors = apriori_equivariance_error(; u, setup, models, setup.plotdir)
    save_object(equi_errors_prior_file, errors)
end

equi_errors_prior = load_object(equi_errors_prior_file)

equi_errors_prior |> e -> map(x -> round(mean(x); sigdigits = 4), e) |> pairs

##################################
# A-posteriori equivariance errors
##################################

equi_errors_post_file = joinpath(setup.outdir, "equi-errors-post.jld2")

let
    grid = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    models = (;
        nomo = m_nomo,
        smag = m_smag,
        clar = m_clar,
        tbnn = m_tbnn,
        equi = m_equi,
        conv = m_conv,
    )
    ustart = data[1][end] |> adapt(setup.backend)
    (; elements) = group_stuff(setup.D)
    errors =
        map(keys(models)) do key
            model = models[key]
            @info "Computing equivariance error for $(key)"
            e = map(eachindex(elements)) do i
                @info "Element $(i) of $(length(elements))"
                test_equivariance_post(;
                    ustart,
                    setup,
                    grid,
                    model,
                    groupindex = i,
                    rng = Xoshiro(123),
                    tstop = 1e-1,
                    cfl = 0.35,
                    dolog = false,
                )
            end
            key => e
        end |> NamedTuple
    save_object(equi_errors_post_file, errors)
end

equi_errors_post = load_object(equi_errors_post_file)

equi_errors_post |> e -> map(x -> round(mean(x); sigdigits = 4), e) |> pairs

let
    for (errs, name) in [
        (equi_errors_prior, "equi-errors-prior.pdf"),
        (equi_errors_post, "equi-errors-post.pdf"),
    ]
        fig = plot_equivariance_errors(errs, setup)
        save("$(setup.plotdir)/$(name)", fig; backend = CairoMakie)
        display(fig)
    end
end

let
    s = group_stuff(3)
    s.mats[43]
end

##################################
# Dissipation
##################################

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

let
    # comp = :x
    comp = :z
    fig = plot_velocities(setup, u, comp)
    save("$(setup.plotdir)/velocities-$(comp).png", fig; backend = CairoMakie)
    fig
end

let
    models = (;
        # nomo = m_nomo,
        smag = m_smag,
        clar = m_clar,
        tbnn = m_tbnn,
        equi = m_equi,
        conv = m_conv,
    )
    u = load(dnsfile, "u") |> adapt(setup.backend)
    fig = plot_sfs(setup, u, models)
    save("$(setup.plotdir)/sfs.png", fig; backend = CairoMakie)
    fig
end

let
    (; D, l, n_dns, visc, backend) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)
    u = load(dnsfile, "u") |> adapt(backend)
    stat = turbulence_statistics(u |> adapt(backend), visc, g_dns)
end |> pairs

qr_file = joinpath(setup.outdir, "qr.jld2")

let
    qr = compute_qr(u, setup)
    save_object(qr_file, qr)
end

qr = load_object(qr_file)

let
    fig = plot_qr(setup, (; qr..., dns = qrdns.dns))
    save("$(setup.plotdir)/qr.pdf", fig; backend = CairoMakie)
    fig
end

let
    fig = Figure()
end

let
    g = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    u = load(dnsfile, "u") |> adapt(setup.backend)
    G = getgradient(u, g)
    q = spacescalarfield(g)
    r = spacescalarfield(g)
    apply!(qr_kernel!, g, (q, r, G, g); ndrange = space_ndrange(g))
    q = q |> cpu_device() |> vec
    r = r |> cpu_device() |> vec
    t_kol = 1 / sum(G -> sum(abs2, G) * (g.l / g.n)^3, G) |> sqrt
    q .*= t_kol^2
    r .*= t_kol^3
    k = kde((r, q)) #; npoints = (5000, 5000))
    p = k.density
    @show extrema(q) extrema(r) extrema(p)
    p = max.(p, 1e-20)
    # p |> extrema
    # @. p = log(max(1e-20, p))
    # @show maximum(p)
    # heatmap(k.x, k.y, p; colorscale = log10, colorrange = (1e-5, maximum(p)))
    # fig = Figure(; size = (400, 300))
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "R", ylabel = "Q")
    ran = 1e-4, 1e1
    contour!(
        ax,
        k.x,
        k.y,
        p;
        # labels = true,
        levels = logrange(ran..., 6),
        colorrange = ran,
        colorscale = log10,
    )
    qtest = range(-10, 0, 200)
    rtest1 = @. 2 / 3 / sqrt(3) * (-qtest)^(3 / 2)
    rtest2 = @. -2 / 3 / sqrt(3) * (-qtest)^(3 / 2)
    lines!(ax, rtest1, qtest; color = :red)
    lines!(ax, rtest2, qtest; color = :red)
    xlims!(ax, -2.5, 2.5)
    ylims!(ax, -4, 4)
    fig
end
