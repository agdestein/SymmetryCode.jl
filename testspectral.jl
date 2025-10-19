# if false
#     include("src/SymmetryCode.jl")
#     using .SymmetryCode
#     using .SymmetryCode.Spectral
# end

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

create_dns(setup; t_warmup = 0.5, cfl = 0.35, rng = Xoshiro(0))

let
    times, energies = load("$(setup.outdir)/dns.jld2", "times", "energies")
    fig, _, _ = lines(times, energies)
    save(joinpath(setup.plotdir, "energy.pdf"), fig; backend = CairoMakie)
    fig
end

# Plot DNS spectrum
plot_spectrum_dns(setup)

data, datatiming = let
    filename = joinpath(setup.outdir, "data.jld2")
    if true
        t = time()
        d = create_data(
            setup;
            cfl = 0.35,
            nstep = setup.D == 2 ? 1000 : 30,
            nsubstep = 10,
        )
        t = time() - t
        jldsave(filename; data = d, timing = t)
    end
    load(filename, "data", "timing")
end;

Base.summarysize(data) * 1e-9

m_nomo = let
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    m_nomo(_) = fill!(stack(spacetensorfield(g)), 0)
end

m_smag = create_smagorinsky(
    0.17,
    setup.Δ,
    Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend),
)

m_vers = create_verstappen(
    sqrt(3 / 2) / π, # 0.3898, in original paper
    # 0.527, # Higher value from Trias "building proper invariants" paper
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
    # net = Chain(
    #     Conv(kern, Spectral.ninvariant(g) => 16, gelu),
    #     Conv(kern, 16 => 32, gelu),
    #     Conv(kern, 32 => 64, gelu),
    #     Conv(kern, 64 => Spectral.nbasis(g); use_bias = false),
    # ) # 3_200 parameters
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
                batchsize = 20,
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
    chain = tbnn(net, ps, st, setup.Δ, g)
    chain, (; losses_train, losses_valid, timing)
end;

m_equi, train_equi = let
    net_stuff = equivariant_net(
        setup,
        # [12, 16, 16, 24], # 40_328 actual params
        [8, 8, 8, 16], # 12_544 actual params
        # [4, 4, 4, 8], # 3_200 actual params
    )
    st = net_stuff.st
    file = joinpath(setup.outdir, "ps-equi.jld2")
    if true
        t = time()
        (; ps, st, losses_train, losses_valid) = train(;
            loss = create_loss(net_stuff.project),
            setup,
            dataloader = create_dataloader(setup, data; batchsize = 20),
            nepoch = 10,
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
    chain = fullchain(setup, net, project, ps, st, setup.Δ)
    chain, (; losses_train, losses_valid, timing)
end;

m_conv, train_conv = let
    net_stuff = cnn(
        setup,
        # [48, 128, 128, 128]; # 40_550 parameters
        [48, 64, 64, 64]; # 12_320 parameters
        # [16, 32, 64]; # 3_200 parameters
        same_as_equi = false,
    )
    for ps in net_stuff.ps
        # Initialize weights are too large
        hasfield(typeof(ps), :weight) && (ps.weight .*= 0.1)
    end
    st = net_stuff.st
    file = joinpath(setup.outdir, "ps-conv.jld2")
    if true
        t = time()
        (; ps, st, losses_train, losses_valid) = train(;
            loss = create_loss(net_stuff.project),
            setup,
            dataloader = create_dataloader(setup, data; batchsize = 20),
            nepoch = 10,
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
    chain = fullchain(setup, net, project, ps, st, setup.Δ)
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
        vers = "vers",
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
        vers = m_vers,
        clar = m_clar,
        tbnn = m_tbnn,
        equi = m_equi,
        conv = m_conv,
    )
    u_dns = load("$(setup.outdir)/dns.jld2", "u") |> adapt(setup.backend)
    inference_post(;
        u_dns,
        setup,
        models,
        files = upostfiles,
        cfl = 0.35,
        tstop = 1e-1,
        dodns = true,
    )
end

map(f -> load(f, "timing"), upostfiles) |> t -> map(x -> round(x; digits = 1), t) |> pairs

u = map(f -> load(f, "u"), upostfiles);

get_errors(setup, u);

# Plot LES spectrum
plot_spectrum_les(setup, u)

let
    models = (; smag = m_smag, clar = m_clar, tbnn = m_tbnn, equi = m_equi, conv = m_conv)
    u_dns = load("$(setup.outdir)/dns.jld2", "u") |> adapt(setup.backend)
    plot_densities(; u_dns, setup, models, dolog = true)
end

prediction_error_prior_file = joinpath(setup.outdir, "prediction-error-prior.jld2")

let
    models = (;
        nomo = m_nomo,
        smag = m_smag,
        vers = m_vers,
        clar = m_clar,
        tbnn = m_tbnn,
        equi = m_equi,
        conv = m_conv,
    )
    u_dns = load("$(setup.outdir)/dns.jld2", "u") |> adapt(setup.backend)
    e = apriori_error(; u_dns, setup, models)
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
        fig = plot_equivariance_errors(errs)
        save("$(setup.plotdir)/$(name)", fig; backend = CairoMakie)
        display(fig)
    end
end

##################################
# Dissipation
##################################

dissipation_errors = let
    u_dns = load("$(setup.outdir)/dns.jld2", "u") |> adapt(setup.backend)
    models = (;
        nomo = m_nomo,
        smag = m_smag,
        vers = m_vers,
        clar = m_clar,
        tbnn = m_tbnn,
        conv = m_conv,
        equi = m_equi,
    )
    get_dissipation_errors(; setup, u_dns, models)
end;

dissipation_errors |> e -> map(x -> round(x; sigdigits = 4), e) |> pairs

let
    # comp = :x
    comp = :z
    uplot = filter(!=(u.vers), u)
    fig = plot_velocities(setup, uplot, comp)
    save("$(setup.plotdir)/velocities-$(comp).png", fig; backend = CairoMakie)
    fig
end

let
    models = (;
        # nomo = m_nomo,
        smag = m_smag,
        # vers = m_vers,
        clar = m_clar,
        tbnn = m_tbnn,
        equi = m_equi,
        conv = m_conv,
    )
    u = load("$(setup.outdir)/dns.jld2", "u") |> adapt(setup.backend)
    fig = plot_sfs(setup, u, models)
    save("$(setup.plotdir)/sfs.png", fig; backend = CairoMakie)
    fig
end

let
    (; D, l, n_dns, visc, backend) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)
    u = load("$(setup.outdir)/dns.jld2", "u") |> adapt(setup.backend)
    turbulence_statistics(u, visc, g_dns)
end |> pairs

qr_file = joinpath(setup.outdir, "qr.jld2")

let
    qr = compute_qr(u, setup)
    save_object(qr_file, qr)
end

qr = load_object(qr_file);

let
    # fig = plot_qr(setup, qr)
    fig = plot_qr(setup, filter(!=(qr.vers), qr))
    save("$(setup.plotdir)/qr.pdf", fig; backend = CairoMakie)
    fig
end
