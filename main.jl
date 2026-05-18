@info "Loading packages"
flush(stderr)

using Adapt
using CairoMakie
using CUDA, cuDNN
using FFTW
using JLD2
using LinearAlgebra
using Random
using StaticArrays
using Statistics
using WGLMakie

import SymmetryCode as S

lines([1, 2, 3])

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

let
    (; D) = setup
    u = load("$(setup.outdir)/dns.jld2", "u") |> u -> map(copy, u) |> adapt(setup.backend)
    g = S.Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    v = S.spacescalarfield(g)
    p = plan_rfft(v)
    fac = S.get_fft_fac(g)
    if D == 2
        ldiv!(v, p, u.x)
        v .*= fac
        field = v |> Array
    else
        ldiv!(v, p, u.z)
        v .*= fac
        field = v[:, :, end] |> Array
    end
    fig, _ = heatmap(field; colormap = :RdBu)
    save("$(setup.plotdir)/dnsfield.png", fig)
    fig
end

let
    (; D) = setup
    data = joinpath(setup.outdir, "data.jld2") |> load_object
    u = map(copy, data.inputs[1]) |> adapt(setup.backend)
    g = S.Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    v = S.spacescalarfield(g)
    p = plan_rfft(v)
    fac = S.get_fft_fac(g)
    if D == 2
        ldiv!(v, p, u.x)
        v .*= fac
        field = v |> Array
    else
        ldiv!(v, p, u.z)
        v .*= fac
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

m_nomo = let
    g = S.Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    m_nomo(_, _) = fill!(stack(S.spacetensorfield(g)), 0)
end

m_smag = S.create_smagorinsky(
    0.1,
    setup.Δ,
    S.Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend),
)

m_dynsmag = S.create_dynamic_smagorinsky(
    setup.Δ,
    S.Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend),
)

# m_vers = create_verstappen(
#     sqrt(3 / 2) / π, # 0.3898, in original paper
#     # 0.527, # Higher value from Trias "building proper invariants" paper
#     setup.Δ,
#     Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend),
# )

m_clar = S.create_clark(setup.Δ, S.Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend))

# m_tbnn, train_tbnn = S.create_tbnn(setup, data, false);
#
# m_equi, train_equi = S.create_equi(setup, data, false);
#
# m_conv, train_conv = S.create_conv(setup, data, false);
#
# S.plot_training(setup, train_tbnn, train_equi, train_conv)
#
# map(
#     t -> round(t; digits = 1),
#     (;
#         tbnn = train_tbnn.timing,
#         equi = train_equi.timing,
#         conv = train_conv.timing,
#     ),
# ) |> pairs |> display
# flush(stdout)

upostfiles = map(
    name -> "$(setup.outdir)/u-post-$(name).jld2",
    (;
        nomo = "nomo",
        smag = "smag",
        dynsmag = "dynsmag",
        # vers = "vers",
        clar = "clar",
        # tbnn = "tbnn",
        # equi = "equi",
        # conv = "conv",
    ),
)

S.solve_les(
    setup,
    (;
        nomo = m_nomo,
        smag = m_smag,
        dynsmag = m_dynsmag,
        # vers = m_vers,
        clar = m_clar,
        # tbnn = m_tbnn,
        # equi = m_equi,
        # conv = m_conv,
    ),
)

# round(data.timing; digits = 1)

let
    keys = [
        :nomo,
        :smag,
        :dynsmag,
        # :vers,
        :clar,
        # :tbnn,
        # :equi,
        # :conv,
    ]
    files = S.get_upostfiles(setup)
    map(keys) do k
        t = load_object(files[k]).timing
        tround = round(t; digits = 1)
        k => tround
    end |> NamedTuple |> pairs |> display
    flush(stdout)
end

let
    keys = [
        :nomo,
        :smag,
        :dynsmag,
        # :vers,
        :clar,
        # :tbnn,
        # :equi,
        # :conv,
    ]
    les_stat = S.get_les_statistics(setup, keys)
    save_object("$(setup.outdir)/les_stat.jld2", les_stat)
end

les_stat = load_object("$(setup.outdir)/les_stat.jld2");

map(s -> round(mean(s.e_post); sigdigits = 4), les_stat) |> pairs

let
    data = joinpath(setup.outdir, "data.jld2") |> load_object
    fig = Figure(; size = (400, 360))
    ax = Axis(
        fig[1, 1];
        xlabel = "Time",
        ylabel = "Relative error",
        # yscale = log10,
        # xscale = log10,
    )
    t = data.times
    labels = S.getlabels()
    for k in keys(les_stat)
        e = les_stat[k].e_post
        ntime = length(e)
        lines!(ax, t[1:ntime], e; label = labels[k])
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
    # ylims!(ax, -0.03, 0.5)
    rowgap!(fig.layout, 5)
    save("$(setup.plotdir)/error_post.pdf", fig; backend = CairoMakie)
    fig
end

# Plot LES spectrum
S.plot_spectrum_les(setup, les_stat)

S.predict_sfs(
    setup,
    (;
        #
        smag = m_smag,
        dynsmag = m_dynsmag,
        clar = m_clar,
        # tbnn = m_tbnn,
        # equi = m_equi,
        # conv = m_conv,
    ),
)

S.compute_densities(
    setup, [
        :smag,
        :dynsmag,
        :clar,
        # :tbnn,
        # :equi,
        # :conv,
    ]
)

S.plot_densities(setup; dolog = true)

prediction_error_prior = let
    modelkeys = [
        :nomo,
        :smag,
        :dynsmag,
        :clar,
        # :tbnn,
        # :equi,
        # :conv,
    ]
    S.apriori_error(setup, modelkeys)
end

map(x -> round(x.relerr; sigdigits = 4), prediction_error_prior) |> pairs
map(x -> round(x.crosscor; sigdigits = 4), prediction_error_prior) |> pairs

##############################
# A-priori equivariance errors
##############################

equi_errors_prior_file = joinpath(setup.outdir, "equi-errors-prior.jld2")

let
    # u = map(copy, data.inputs) |> adapt(setup.backend)
    models = (;
        smag = m_smag,
        dynsmag = m_dynsmag,
        clar = m_clar,
        # tbnn = m_tbnn,
        # equi = m_equi,
        # conv = m_conv,
    )
    errors = S.apriori_equivariance_error(; u, setup, models)
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
    grid = S.Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    models = (;
        nomo = m_nomo,
        smag = m_smag,
        clar = m_clar,
        # tbnn = m_tbnn,
        # equi = m_equi,
        # conv = m_conv,
    )
    ustart = data[1][end] |> adapt(setup.backend)
    (; elements) = S.group_stuff(setup.D)
    errors =
        map(keys(models)) do key
        model = models[key]
        @info "Computing equivariance error for $(key)"
        e = map(eachindex(elements)) do i
            @info "Element $(i) of $(length(elements))"
            flush(stderr)
            S.test_equivariance_post(;
                ustart,
                setup,
                grid,
                model,
                groupindex = i,
                tstop = 1.0e-1,
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
        fig = S.plot_equivariance_errors(errs)
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
    S.get_dissipation_errors(; setup, u_dns, models)
end;

dissipation_errors |> e -> map(x -> round(x; sigdigits = 4), e) |> pairs

S.plot_velocities(setup, :x)
S.plot_velocities(setup, :z)

S.plot_sfs(setup)

S.compute_qr(
    setup,
    [
        :ref,
        :nomo,
        :smag,
        :dynsmag,
        :clar,
        # :tbnn,
        # :equi,
        # :conv,
    ],
)

S.plot_qr(
    setup,
    [
        :ref,
        :nomo,
        :smag,
        :dynsmag,
        :clar,
        # :tbnn,
        # :equi,
        # :conv,
    ],
)
