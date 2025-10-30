if false
    include("src/SymmetryCode.jl")
    using .SymmetryCode
    using .SymmetryCode.Spectral
end

@info "Loading packages"
flush(stderr)

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
using WGLMakie
lines([1, 2, 3])

setup = setup_laptop()
setup = setup_turbulator()
setup = setup_snellius()
setup |> pairs

setup = setup_turbulator();
create_dns(setup);
plot_spectrum_dns(setup)

# Warmup simulation
create_dns(setup)

# Plot DNS spectrum
plot_spectrum_dns(setup)

# Plot time series
plot_evolution_dns(setup)

# Plot dissipation vs finite difference of energy
plot_dissipation_finite_difference(setup)

# set_theme!(;
#     fonts = (;
#         regular = "Dejavu",
#         # regular = "Palatino",
#     ),
# )

setup = setup_turbulator();
create_data(setup)

data = joinpath(setup.outdir, "data.jld2") |> load_object;

let
    u = load("$(setup.outdir)/dns.jld2", "u") |> adapt(setup.backend)
    g = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    v = spacescalarfield(g)
    p = plan_rfft(v)
    ldiv!(v, p, u.z)
    v .*= g.n^3 # FFT factor
    r = 5e-1
    field = v[:, :, end] |> Array
    image(
        field;
        # colorrange = (-r, r),
        # colormap = :balance,
        colormap = :RdBu,
        # colormap = :seaborn_icefire_gradient,
    )
end

let
    u = data.inputs[1] |> adapt(setup.backend)
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    v = spacescalarfield(g)
    p = plan_rfft(v)
    ldiv!(v, p, u.x)
    v[:, :, end] |> Array |> heatmap
end

Base.summarysize(data) * 1e-9

data |> pairs

getindex.(data.statistics_dns, :diss)
getindex.(data.statistics_dns, :uavg) .^ 2 / 2 * 3
getindex.(data.statistics_dns, :Re_tay)
getindex.(data.statistics_dns, :t_int)
getindex.(data.statistics_dns, :l_int)
getindex.(data.statistics_dns, :l_kol)

let
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
    fig
end

let
    s_dns = mean(data.spectra_dns)
    s_les = mean(data.spectra_les)
    diss = mean(s -> s.diss, data.statistics_dns)
    eta = mean(s -> s.l_kol, data.statistics_dns)
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10)
    k_dns = 2π / setup.l * eachindex(s_dns)
    k_les = 2π / setup.l * eachindex(s_les)
    C = 1.6
    s_kol = C * diss^(2/3) * k_dns .^ (-5/3)
    escale = C^(-1) * diss^(-2/3) * eta^(-5/3)
    lines!(ax, eta * k_dns, escale * s_dns)
    lines!(ax, eta * k_les, escale * s_les)
    lines!(ax, eta * k_dns, escale * s_kol)
    fig
end

m_nomo = let
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    m_nomo(_) = fill!(stack(spacetensorfield(g)), 0)
end

m_smag = create_smagorinsky(
    0.17,
    setup.Δ,
    Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend),
)

# m_vers = create_verstappen(
#     sqrt(3 / 2) / π, # 0.3898, in original paper
#     # 0.527, # Higher value from Trias "building proper invariants" paper
#     setup.Δ,
#     Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend),
# )

m_clar = create_clark(setup.Δ, Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend))

m_tbnn, train_tbnn = let
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    kern = ntuple(Returns(1), setup.D)
    net = Chain(
        Conv(kern, ninvariant(g) => 64, gelu),
        Conv(kern, 64 => 64, gelu),
        Conv(kern, 64 => 128, gelu),
        Conv(kern, 128 => nbasis(g); use_bias = false),
    ) # 13_888 parameters
    # net = Chain(
    #     Conv(kern, Spectral.ninvariant(g) => 16, gelu),
    #     Conv(kern, 16 => 32, gelu),
    #     Conv(kern, 32 => 64, gelu),
    #     Conv(kern, 64 => Spectral.nbasis(g); use_bias = false),
    # ) # 3_200 parameters
    net |> display
    flush(stdout)
    ps, st = Lux.setup(Xoshiro(0), net) |> f64 |> adapt(setup.backend)
    for l in ps
        l.weight .*= 0.1
    end
    file = joinpath(setup.outdir, "ps-tbnn.jld2")
    if false
        @info "Training TBNN"
        flush(stderr)
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
    if false
        @info "Training G-conv"
        flush(stderr)
        t = time()
        (; ps, st, losses_train, losses_valid) = train(;
            loss = create_loss(net_stuff.project),
            setup,
            dataloader = create_dataloader(setup, data; batchsize = 20),
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
    flush(stdout)
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
    if false
        @info "Training Conv"
        flush(stderr)
        t = time()
        (; ps, st, losses_train, losses_valid) = train(;
            loss = create_loss(net_stuff.project),
            setup,
            dataloader = create_dataloader(setup, data; batchsize = 20),
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
    flush(stdout)
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
) |>
pairs |>
display
flush(stdout)

upostfiles = map(
    name -> joinpath(setup.outdir, "u-post-$(name).jld2"),
    (;
        # dns = "dns",
        # ref = "ref",
        nomo = "nomo",
        smag = "smag",
        # vers = "vers",
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
        # vers = m_vers,
        clar = m_clar,
        tbnn = m_tbnn,
        equi = m_equi,
        conv = m_conv,
    )
    inference_post(; data, setup, models, files = upostfiles)
end

map(f -> load_object(f).timing, upostfiles) |>
t -> map(x -> round(x; digits = 1), t) |> pairs |> display
flush(stdout)

# u = map(f -> load_object(f).u, upostfiles);

e_post = compute_les_statistics(setup, data, upostfiles);

e_post = SymmetryCode.get_errors(setup, data, upostfiles);

map(e -> round(mean(e); sigdigits = 4), e_post) |> pairs

let
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Quantity",
            # yscale = log10,
            # xscale = log10,
              )
    t = data.times
    labels = getlabels()
    for k in keys(e_post)
        e = e_post[k]
        lines!(ax, t[2:end], e[2:end]; label = labels[k])
    end
    # eps = 0.1
    # ylims!(ax, -eps, 1 + eps)
    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        tellheight = true,
        framevisible = false,
        horizontal = true,
        nbanks = 3,
    )
    fig
end

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
        # vers = m_vers,
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
    errors = apriori_equivariance_error(; u, setup, models)
    save_object(equi_errors_prior_file, errors)
end

equi_errors_prior = load_object(equi_errors_prior_file)

equi_errors_prior |> e -> map(x -> round(mean(x); sigdigits = 4), e) |> pairs |> display
flush(stdout)

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
                flush(stderr)
                test_equivariance_post(;
                    ustart,
                    setup,
                    grid,
                    model,
                    groupindex = i,
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
flush(stdout)

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

@info "Done."
flush(stderr)
exit()

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
    # comp = :z
    for comp in [:z, :x]
        uplot = filter(!=(u.vers), u)
        fig = plot_velocities(setup, uplot, comp)
        save("$(setup.plotdir)/velocities-$(comp).png", fig; backend = CairoMakie)
        fig
    end
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
    setup = setup_turbulator()
    (; D, l, n_dns, visc, backend) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)
    u = load("$(setup.outdir)/dns.jld2", "u") |> adapt(setup.backend)
    turbulence_statistics(u, visc, g_dns)
end |> pairs |> display
flush(stdout)

let
    (; D, l, n_dns, visc, backend) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)
    u = load("$(setup.outdir)/dns.jld2", "u") |> adapt(setup.backend)
    stat = turbulence_statistics(u, visc, g_dns)
    diss1 = stat.diss
    dd = similar(u.x, typeof(l))
    diss2 = get_dissipation!(dd, u, visc, g_dns)
    @show diss1 diss2
end;

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
