@info "Loading packages"
flush(stderr)

using Adapt
using CairoMakie
using CUDA, cuDNN
using FFTW
using LinearAlgebra
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

# Warmup simulation
create_dns(setup)

# Plot DNS spectrum
plot_spectrum_dns(setup)

# Plot time series
plot_evolution_dns(setup)

# # Plot dissipation vs finite difference of energy
# plot_dissipation_finite_difference(setup)

# set_theme!(;
#     fonts = (;
#         regular = "Dejavu",
#         # regular = "Palatino",
#     ),
# )

create_data(setup);

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
    ldiv!(v, p, u.z)
    v .*= g.n^3 # FFT factor
    field = v[:, :, end] |> Array
    image(field; colormap = :RdBu)
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
    save(joinpath(setup.plotdir, "evolution_data.pdf"), fig; backend = CairoMakie)
    fig
end

plot_spectrum_data(setup, data)

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

m_tbnn, train_tbnn = create_tbnn(setup, data, false);

m_equi, train_equi = create_equi(setup, data, false);

m_conv, train_conv = create_conv(setup, data, false);

plot_training(setup, train_tbnn, train_equi, train_conv)

map(
    t -> round(t; digits = 1),
    (; tbnn = train_tbnn.timing, conv = train_conv.timing, equi = train_equi.timing),
) |>
pairs |>
display
flush(stdout)

upostfiles = map(
    name -> "$(setup.outdir)/u-post-$(name).jld2",
    (;
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
    solve_les(; data, setup, models, files = upostfiles)
end

map(f -> load_object(f).timing, upostfiles) |>
t -> map(x -> round(x; digits = 1), t) |> pairs |> display
flush(stdout)

# u = map(f -> load_object(f).u, upostfiles);

les_stat = get_les_statistics(setup, data, upostfiles);

map(s -> round(mean(s.e_post); sigdigits = 4), les_stat) |> pairs

let
    fig = Figure(; size = (400, 360))
    ax = Axis(
        fig[1, 1];
        xlabel = "Time",
        ylabel = "Error",
        # yscale = log10,
        # xscale = log10,
    )
    t = data.times
    labels = getlabels()
    start = 1
    for k in keys(les_stat)
        e = les_stat[k].e_post
        lines!(ax, t[start:end], e[start:end]; label = labels[k])
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
    rowgap!(fig.layout, 5)
    save("$(setup.plotdir)/error_post.pdf", fig; backend = CairoMakie)
    fig
end

# Plot LES spectrum
plot_spectrum_les(setup, data, les_stat)

predict_sfs(
    setup,
    data,
    (;
        #
        smag = m_smag,
        clar = m_clar,
        tbnn = m_tbnn,
        equi = m_equi,
        conv = m_conv,
    ),
)

compute_densities(setup, data, [
    #
    :smag,
    :clar,
    :tbnn,
    :equi,
    :conv,
])

plot_densities(setup; dolog = true)

prediction_error_prior_file = joinpath(setup.outdir, "tensor_error.jld2")

prediction_error_prior = let
    modelkeys = [
        :nomo,
        :smag,
        # :vers,
        :clar,
        :tbnn,
        :equi,
        :conv,
    ]
    apriori_error(setup, data, modelkeys)
end

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

plot_velocities(setup, data, upostfiles, :x)
plot_velocities(setup, data, upostfiles, :z)

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
    save("$(setup.plotdir)/sfs.png", fig; backend = CairoMakie)
    fig
end

plot_sfs(setup, data)

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

compute_qr(setup, data, upostfiles)

plot_qr(setup)
