@info "Loading packages"
flush(stderr)

using Adapt
using CairoMakie
using CUDA, cuDNN
using JLD2
using Statistics: mean
using WGLMakie

import SymmetryCode as S

# Warmup plot
lines([1, 2, 3])

# Choose setup
setup = S.setup_laptop()
setup = S.setup_turbulator_small()
setup = S.setup_turbulator_medium()
setup = S.setup_turbulator_large()
setup = S.setup_snellius()
setup |> pairs

#######################
# Define closure models
#######################

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

m_clar = S.create_clark(setup.Δ, S.Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend))

m_tbnn, train_tbnn = S.create_tbnn(setup, :resume);

m_equi, train_equi = S.create_equi(setup, :resume);

m_conv, train_conv = S.create_conv(setup, :resume);

S.plot_training(setup, train_tbnn, train_equi, train_conv)

map(
    t -> round(t; digits = 1),
    (;
        tbnn = train_tbnn.timing,
        equi = train_equi.timing,
        conv = train_conv.timing,
    ),
) |> pairs |> display
flush(stdout)

#######################
# Deploy closure models
#######################

S.solve_les(
    setup,
    (;
        nomo = m_nomo,
        smag = m_smag,
        dynsmag = m_dynsmag,
        # vers = m_vers,
        clar = m_clar,
        tbnn = m_tbnn,
        # equi = m_equi,
        conv = m_conv,
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
        :tbnn,
        # :equi,
        :conv,
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
        :tbnn,
        # :equi,
        :conv,
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
        tbnn = m_tbnn,
        # equi = m_equi,
        conv = m_conv,
    ),
)

S.compute_densities(
    setup, [
        :smag,
        :dynsmag,
        :clar,
        :tbnn,
        # :equi,
        :conv,
    ]
)

S.plot_densities(
    setup, [
        :ref,
        :smag,
        :dynsmag,
        :clar,
        :tbnn,
        # :equi,
        :conv,
    ]; dolog = true
)

prediction_error_prior = let
    modelkeys = [
        :nomo,
        :smag,
        :dynsmag,
        :clar,
        :tbnn,
        # :equi,
        :conv,
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

let
    keys = [
        # :dns,
        :ref,
        :nomo,
        :smag,
        :dynsmag,
        :clar,
        :tbnn,
        # :equi,
        :conv,
    ]
    # S.plot_velocities(setup, :x, keys)
    S.plot_velocities(setup, :z, keys)
end

S.plot_sfs(
    setup, [
        :smag,
        :dynsmag,
        :clar,
        :tbnn,
        # :equi,
        :conv,
    ],
)

S.compute_qr(
    setup, [
        :ref,
        :nomo,
        :smag,
        :dynsmag,
        :clar,
        :tbnn,
        # :equi,
        :conv,
    ],
)

S.plot_qr(
    setup, [
        :ref,
        :nomo,
        :smag,
        :dynsmag,
        :clar,
        :tbnn,
        # :equi,
        :conv,
    ],
)
