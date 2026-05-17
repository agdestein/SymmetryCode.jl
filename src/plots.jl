# All plot_* routines and the shared label table they read from.

getlabels() = (;
    dns = "DNS",
    ref = "Filtered DNS",
    nomo = "No-model",
    smag = "Smagorinsky",
    dynsmag = "Dynamic Smagorinsky",
    vers = "Verstappen",
    clar = "Clark",
    tbnn = "TBNN",
    equi = "G-Conv",
    conv = "Conv",
)

"""
Inertial-range reference spectrum and the normalization that collapses the
data onto the universal `κ̃^{-p}` shape.
3D: Kolmogorov  `E = C ε^{2/3} κ^{-5/3}`,  `κ̃ = κ·l_kol`  (κη).
2D: Kraichnan–Batchelor enstrophy cascade  `E = C η_Ω^{2/3} κ^{-3}`,
    `κ̃ = κ·l_kra`  (κη_ω).
`stats` may be a Vector of stat NamedTuples (averaged) or a single one.
Plotting `(kscale·k, escale·E)` puts the inertial range on the universal
`κ̃^{-p}` line and the dissipation scale at `κ̃ ≈ kdiss = 1`.
"""
function spectrum_reference(setup, stats)
    (; D, l) = setup
    sm(f) = stats isa AbstractVector ? mean(f, stats) : f(stats)
    if D == 2
        χ, l_d, p, C = sm(s -> s.enstrophy_diss), sm(s -> s.l_kra), 3, 1.4
        xlabel = L"\kappa \eta_\omega"
        ylabel = L"C^{-1} \eta_\Omega^{-2/3} \eta_\omega^{-3}\, E(\kappa)"
        label = "Kraichnan −3"
    else
        χ, l_d, p, C = sm(s -> s.diss), sm(s -> s.l_kol), 5 / 3, 1.6
        xlabel = L"\kappa \eta"
        ylabel = L"C^{-1} \epsilon^{-2/3} \eta^{-5/3}\, E(\kappa)"
        label = "Kolmogorov −5/3"
    end
    kscale = l_d
    escale = 1 / (C * χ^(2 / 3) * l_d^(-p))      # = C^{-1} χ^{-2/3} l_d^{-p}
    # Reference drawn from ~2× the forcing-shell wavenumber up to the
    # dissipation scale κ̃ ≈ 1 (normalized units).
    k_f = 2 * (2π / l) * 2 * l_d
    k_ref = logrange(k_f, 1.0, 100)
    E_ref = k_ref .^ (-p)
    return (; kscale, escale, p, C, k_ref, E_ref, kdiss = 1.0, xlabel, ylabel, label)
end

function plot_training(setup, train_tbnn, train_equi, train_conv)
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
    eps = 0.1
    ylims!(ax, -eps, 1 + eps)
    rowgap!(fig.layout, 5)
    save("$(setup.plotdir)/training.pdf", fig; backend = CairoMakie)
    return fig
end

function plot_densities(setup; dolog)
    (; outdir, plotdir, name) = setup
    yscale = dolog ? log10 : identity

    mkeys = [
        :ref,
        :smag,
        :dynsmag,
        :clar,
        # :tbnn,
        # :equi,
        # :conv,
    ]
    # t_kol = mean(x -> x.t_kol, data.statistics_les)

    fig = Figure(; size = (800, 300))
    labels = getlabels()

    # Axes
    ax = (;
        xx = Axis(
            fig[1, 1];
            xlabel = L"\tau_{1 1}",
            ylabel = "Density",
            xlabelsize = 20,
            yscale,
        ),
        xy = Axis(
            fig[1, 2]; xlabel = L"\tau_{1 2}", xlabelsize = 20, yscale,
            yticksvisible = false,
            yticklabelsvisible = false,
        ),
        diss = Axis(
            fig[1, 3]; xlabel = L"\tau_{i j} S_{i j}", xlabelsize = 20, yscale,
            yticksvisible = false,
            yticklabelsvisible = false,
        ),
    )

    for fkey in [:xx, :xy, :diss], mkey in mkeys
        dens = "$(outdir)/kde_$(mkey)_$(fkey).jld2" |> load_object
        # if fkey == :diss && mkey == :smag
        #     @info "Hi"
        #     # The line at 0 is not visible for smagorinsky, append a zero
        #     lines!(ax[fkey], [dens.x; 2 * dens.x[end]], [dens.density; 1e-10]; label = labels[mkey])
        # else
        lines!(ax[fkey], dens.x, max.(dens.density, 1.0e-16); label = labels[mkey])
        # end
    end

    if name == "laptop"
        xlims!(ax.xx, -0.1, 0.3)
        ylims!(ax.xx, 2.0e-2, 3.0e2)
    elseif name == "turbulator"
        xlims!(ax.xx, -0.2, 0.2)
        ylims!(ax.xx, 2.0e-4, 3.0e2)
    elseif name == "snellius"
        xlims!(ax.xx, -0.1, 0.12)
        ylims!(ax.xx, 4.0e-4, 4.0e2)
    end

    # XY-component
    if name == "laptop"
        xlims!(ax.xy, -0.1, 0.1)
        ylims!(ax.xy, 1.0e-1, 5.0e2)
    elseif name == "turbulator"
        xlims!(ax.xy, -0.15, 0.15)
        ylims!(ax.xy, 1.0e-3, 3.0e2)
    elseif name == "snellius"
        xlims!(ax.xy, -0.12, 0.12)
        ylims!(ax.xy, 4.0e-4, 4.0e2)
    end

    # Dissipation
    if name == "laptop"
        xlims!(ax.diss, -0.3, 0.3)
        ylims!(ax.diss, 1.0e-1, 1.0e2)
    elseif name == "turbulator"
        xlims!(ax.diss, -0.5, 0.15)
        ylims!(ax.diss, 1.0e-3, 1.0e2)
    elseif name == "snellius"
        xlims!(ax.diss, -0.5, 0.12)
        ylims!(ax.diss, 4.0e-4, 4.0e2)
    end

    Legend(
        fig[0, :],
        ax.xx;
        tellwidth = false,
        tellheight = true,
        framevisible = false,
        orientation = :horizontal,
        # nbanks = 5,
    )
    rowgap!(fig.layout, 5)

    # Save plot
    file = "$(plotdir)/tensor-distributions.pdf"
    @info "Saving density plot to $(file)"
    save(file, fig; backend = CairoMakie)
    return fig
end

function plot_velocities(setup, data, upostfiles, comp)
    (; D, l, n_les, backend) = setup
    fig = Figure(; size = (800, 470))
    g = Grid{D}(; l, n = n_les, backend)
    ui = scalarfield(g)
    ui_space = spacescalarfield(g)
    plan = plan_rfft(ui_space)
    labels = getlabels()
    modelkeys = [
        # :dns,
        :ref,
        :nomo,
        :smag,
        :dynsmag,
        :clar,
        # :tbnn,
        # :equi,
        # :conv,
    ]
    for (k, key) in enumerate(modelkeys)
        @info "Plotting velocity for $(key)"
        flush(stderr)
        title = labels[key]
        useries = if key == :ref
            data.inputs
        else
            upost = load_object(upostfiles[key])
            upost.u
        end
        for (i, t) in enumerate([20, 30, 50, 100])
            t > length(useries) && continue # Clark series exploded and stop early
            ax = Axis(
                fig[i, k];
                ylabel = "t = $(round(data.times[t]; sigdigits = 2))",
                ylabelvisible = k == 1,
                xticksvisible = false,
                xticklabelsvisible = false,
                yticksvisible = false,
                yticklabelsvisible = false,
                aspect = DataAspect(),
                title,
                titlevisible = i == 1,
            )
            ui = useries[t][comp] |> adapt(backend)
            apply!(twothirds!, g, (ui, g))
            to_phys!(ui_space, ui, plan, g)
            range = (:, :)
            # range = (40:60, 40:60)
            slice = ui_space[range..., end] |> Array
            image!(ax, slice; colormap = :RdBu, interpolate = false)
        end
    end
    rowgap!(fig.layout, 10)
    colgap!(fig.layout, 10)
    save("$(setup.plotdir)/velocities-$(comp).png", fig; backend = CairoMakie)
    return fig
end

function plot_qr(setup)
    (; name) = setup
    modelkeys = [
        :ref,
        :nomo,
        :smag,
        :dynsmag,
        :clar,
        # :tbnn,
        # :equi,
        # :conv,
    ]
    qr = map(key -> key => load_object("$(setup.outdir)/qr_$(key).jld2"), modelkeys)
    qr = NamedTuple(qr)

    fig = Figure(; size = (600, 440))
    labels = getlabels()
    colorvec = Makie.wong_colors()
    lescolor = 2
    colors = (;
        line = :red,
        dns = colorvec[3],
        ref = colorvec[1],
        nomo = colorvec[lescolor],
        smag = colorvec[lescolor],
        dynsmag = colorvec[lescolor],
        vers = colorvec[lescolor],
        clar = colorvec[lescolor],
        tbnn = colorvec[lescolor],
        conv = colorvec[lescolor],
        equi = colorvec[lescolor],
    )

    for (k, key) in modelkeys |> enumerate
        title = labels[key]
        j, i = CartesianIndices((3, 2))[k].I
        ax = Axis(
            fig[i, j];
            xlabelvisible = i == 2,
            xticksvisible = i == 2,
            xticklabelsvisible = i == 2,
            ylabelvisible = j == 1,
            yticksvisible = j == 1,
            yticklabelsvisible = j == 1,
            xlabel = L"r",
            ylabel = L"q",
            xlabelsize = 20,
            ylabelsize = 20,
            title,
        )
        if name == "turbulator"
            ran = 1.0e-3, 1.0e1
            ncat = 6
        elseif name == "snellius"
            ran = 1.0e-4, 1.0e1
            ncat = 7
        end
        # key => extrema(qr.density) |> display
        isref = key == :dns || key == :ref
        isref || contour!(
            ax,
            qr.ref.x,
            qr.ref.y,
            max.(qr.ref.density, 1.0e-20);
            levels = logrange(ran..., ncat),
            color = colors.ref,
        )
        contour!(
            ax,
            qr[key].x,
            qr[key].y,
            max.(qr[key].density, 1.0e-20);
            levels = logrange(ran..., ncat),
            color = colors[key],
        )
        qtest = range(-10, 0, 200)
        rtest1 = @. 2 / 3 / sqrt(3) * (-qtest)^(3 / 2)
        rtest2 = @. -2 / 3 / sqrt(3) * (-qtest)^(3 / 2)
        lines!(ax, rtest1, qtest; color = colors.line)
        lines!(ax, rtest2, qtest; color = colors.line)
        if name == "turbulator"
            xlims!(ax, -1.5, 1.5)
            ylims!(ax, -3, 3)
        elseif name == "snellius"
            xlims!(ax, -2.0, 2.0)
            ylims!(ax, -3, 4)
        end
    end
    save("$(setup.plotdir)/qr.pdf", fig; backend = CairoMakie)
    return fig
end

function plot_equivariance_errors(errs)
    fig = Figure(; size = (400, 340))
    ax = Axis(
        fig[1, 1];
        yscale = log10,
        xlabel = "Group element",
        ylabel = "Error",
        xticks = [1, 8, 16, 24, 32, 40, 48],
    )
    ylims!(ax, 1.0e-17, 1)
    i = 1:48
    colors = (;
        nomo = Cycled(1),
        smag = Cycled(2),
        clar = Cycled(3),
        tbnn = Cycled(4),
        equi = Cycled(5),
        conv = Cycled(6),
    )
    labels = getlabels()
    markers = (;
        nomo = :utriangle,
        smag = :circle,
        clar = :rect,
        tbnn = :diamond,
        equi = :rtriangle,
        conv = :x,
    )
    for key in keys(errs)
        e = errs[key]
        e = max.(e, 1.0e-30) # Encode true zeros as 1e-30
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
        # orientation = :horizontal,
        nbanks = 3,
    )
    rowgap!(fig.layout, 5)
    return fig
end

function plot_sfs(setup, data)
    (; D, l, n_les, backend) = setup
    g = Grid{D}(; l, n = n_les, backend)

    τ = spacetensorfield(g)
    τhat = scalarfield(g)
    plan = plan_rfft(τ.xx)

    # Time index
    t = 30

    τcpu = data.outputs[t]
    for (τ, τcpu) in zip(τ, τcpu)
        copyto!(τhat, τcpu)
        apply!(twothirds!, g, (τhat, g))
        to_phys!(τ, τhat, plan, g)
    end
    τ_ref = τ |> cpu_device()

    modelkeys = [
        :smag,
        :dynsmag,
        :clar,
        # :tbnn,
        # :equi,
        # :conv,
    ]
    τ_les = map(modelkeys) do key
        τles = load_object("$(setup.outdir)/sfs_$(key).jld2")[t]
        make_tracefree!(τles, g)
        key => τles
    end |> NamedTuple

    τ_all = (; ref = τ_ref, τ_les...)
    labels = getlabels()
    fig = Figure(; size = (800, 550))
    for (i, comp) in enumerate([:xx, :xy, :zx, :zz])
        for (j, key) in τ_all |> keys |> enumerate
            title = labels[key]
            ax = Axis(
                fig[i, j];
                xlabelvisible = false,
                xticksvisible = false,
                xticklabelsvisible = false,
                ylabelvisible = j == 1,
                yticksvisible = false,
                yticklabelsvisible = false,
                aspect = DataAspect(),
                ylabel = (;
                    xx = L"\tau_{1 1}",
                    xy = L"\tau_{1 2}",
                    zx = L"\tau_{3 1}",
                    zz = L"\tau_{3 3}",
                )[comp],
                ylabelsize = 20,
                title,
                titlevisible = i == 1,
            )
            slice = τ_all[key][comp][:, :, end]
            image!(
                ax,
                slice;
                colormap = :RdBu,
                colorrange = extrema(τ_all.ref[comp][:, :, end]),
                interpolate = false,
            )
        end
    end
    rowgap!(fig.layout, 10)
    colgap!(fig.layout, 10)

    save("$(setup.plotdir)/sfs.png", fig; backend = CairoMakie)
    return fig
end

function plot_evolution_dns(setup)
    times, energies, dissipations =
        load("$(setup.outdir)/dns.jld2", "times", "energies", "dissipations")
    # a = @. dissipations / 2 / energies

    # Create plot
    fig = Figure(; size = (400, 340))
    ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Normalized quantity")
    lines!(ax, times, energies / maximum(energies); label = "Energy")
    lines!(ax, times, dissipations / maximum(dissipations); label = "Dissipation")
    # lines!(ax, times, a / maximum(a); linestyle = :dash, label = "Forcing")
    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        tellheight = true,
        framevisible = false,
        horizontal = true,
        nbanks = 3,
    )
    rowgap!(fig.layout, 10)

    # Save plot
    file = joinpath(setup.plotdir, "evolution-dns.pdf")
    @info "Saving DNS time series plot to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

function plot_evolution_data(setup, data)
    (; D) = setup
    times_warmup, energies_warmup, dissipations_warmup =
        load("$(setup.outdir)/dns.jld2", "times", "energies", "dissipations")
    times_warmup .-= times_warmup[end] # Use negative times for warmup

    times = data.times
    energies = getindex.(data.statistics_dns, :uavg) .^ 2 / 2 * D
    dissipations = getindex.(data.statistics_dns, :diss)

    emax = max(maximum(energies), maximum(energies_warmup))
    dmax = max(maximum(dissipations), maximum(dissipations_warmup))

    # Create plot
    fig = Figure(; size = (400, 340))

    ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Normalized quantity")
    lines!(ax, times, energies / emax; label = "Energy", color = Cycled(1))
    lines!(ax, times_warmup, energies_warmup / emax; linestyle = :dash, color = Cycled(1))
    lines!(ax, times, dissipations / dmax; label = "Dissipation", color = Cycled(2))
    lines!(ax, times_warmup, dissipations_warmup / dmax; linestyle = :dash, color = Cycled(2))
    eps = 0.1
    ylims!(ax, -eps, 1 + eps)

    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        tellheight = true,
        framevisible = false,
        horizontal = true,
        nbanks = 2,
    )

    rowgap!(fig.layout, 10)

    # Save plot
    file = joinpath(setup.plotdir, "evolution_data.pdf")
    @info "Saving energy and dissipation time series plot to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

function plot_dissipation_finite_difference(setup)
    times, energies, dissipations =
        load("$(setup.outdir)/dns.jld2", "times", "energies", "dissipations")

    # Create plot
    fig = Figure(; size = (400, 340))
    ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Quantity")
    lines!(ax, times, 6 / 5 * dissipations; label = "Dissipation")
    lines!(
        ax,
        times[2:end],
        -diff(energies) ./ diff(times);
        label = "Finite difference of energy",
    )
    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        tellheight = true,
        framevisible = false,
        horizontal = true,
        nbanks = 2,
    )
    rowgap!(fig.layout, 10)

    # Save plot
    file = joinpath(setup.plotdir, "dissipation_finite_difference.pdf")
    @info "Saving DNS dissipation plot to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

function plot_spectrum_data(setup, data)
    (; D, l, n_dns, backend) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)

    s_dns = mean(data.spectra_dns)
    s_les = mean(data.spectra_les)
    r = spectrum_reference(setup, data.statistics_dns)

    k_dns = 2π / setup.l * eachindex(s_dns)
    k_les = 2π / setup.l * eachindex(s_les)

    fig = Figure(; size = (400, 340))
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        xlabel = r.xlabel,
        ylabel = r.ylabel,
        xlabelsize = 20,
        ylabelsize = 20,
    )

    # Banded force stuff
    band = getband(g_dns, 3)
    k2min = minimum(band.k2)
    k2max = maximum(band.k2)
    kforce = 2π / l * [sqrt(k2min), sqrt(k2max)]
    span = kforce * r.kscale
    forcecolor = Makie.wong_colors()[4]
    b = sqrt(prod(extrema(r.escale * s_dns)))
    a = 1.1 * span[2]
    c = sqrt(prod(span))
    w = D == 2 ? 1 : 1.5
    arr = D == 2 ? 100 : 5
    vspan!(ax, span...; alpha = 0.3, color = forcecolor)
    text!(ax, a, b / w; color = forcecolor, text = "Force")
    arrows2d!(
        ax,
        Point2(c, b / arr),
        Point2(c, b * arr) - Point2(c, b / arr);
        color = forcecolor,
    )

    lines!(ax, r.kscale * k_dns, r.escale * s_dns; label = "DNS")
    lines!(ax, r.kscale * k_les, r.escale * s_les; label = "Filtered DNS")
    lines!(ax, r.k_ref, r.E_ref; label = r.label)
    vlines!(ax, r.kdiss; color = (:gray, 0.5), linestyle = :dash)

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

    save(joinpath(setup.plotdir, "spectrum_data.pdf"), fig; backend = CairoMakie)
    return fig
end

function plot_spectrum_dns(setup)
    (; outdir, plotdir, D, l, n_dns, n_les, backend, visc) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)
    u = load("$(outdir)/dns.jld2", "u") |> adapt(backend)
    ubar = vectorfield(g_les)
    for (ubar, u) in zip(ubar, u)
        apply!(cutoff!, g_les, (ubar, u))
        apply!(gaussianfilter!, g_les, (ubar, setup.Δ, g_les))
    end
    D = dim(g_dns)
    stuff_dns = spectral_stuff(g_dns)
    stuff_les = spectral_stuff(g_les)
    stat = turbulence_statistics(u, visc, g_dns)
    stat |> pairs |> display
    s_dns = spectrum(u, g_dns, stuff_dns)
    s_les = spectrum(ubar, g_les, stuff_les)
    r = spectrum_reference(setup, stat)
    # l_int_new = pi / 2 / stat.uavg * sum(eachindex(s_dns.s)) do i
    #     s_dns.s[i] / stuff_dns.k[i]
    # end
    # @show stat.l_int l_int_new; error()
    fig = Figure(; size = (400, 340))
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        xlabel = r.xlabel,
        ylabel = r.ylabel,
        xlabelsize = 20,
        ylabelsize = 20,
    )
    # ylims!(ax, 1.0e-4, 1.0e8)
    # ylims!(ax, 1.0e-14, 1.0e0)

    # # Banded force stuff
    # band = getband(g_dns, force[1])
    # k2min = minimum(band.k2)
    # k2max = maximum(band.k2)
    # kforce = 2π / l * [sqrt(k2min), sqrt(k2max)]
    # span = kforce * kscale
    # forcecolor = Makie.wong_colors()[4]
    # vspan!(ax, span...; alpha = 0.3, color = forcecolor)
    # b = sqrt(prod(extrema(escale * s_dns.s)))
    # a = 1.1 * span[2]
    # c = sqrt(prod(span))
    # w = D == 2 ? 1 : 1.5
    # text!(ax, a, b / w; color = forcecolor, text = "Force")
    # arr = D == 2 ? 100 : 5
    # arrows2d!(
    #     ax,
    #     Point2(c, b / arr),
    #     Point2(c, b * arr) - Point2(c, b / arr);
    #     color = forcecolor,
    # )

    lines!(ax, r.kscale * s_dns.k, r.escale * s_dns.s; label = "DNS")
    lines!(ax, r.kscale * s_les.k, r.escale * s_les.s; label = "Filtered DNS")
    lines!(ax, r.k_ref, r.E_ref; label = r.label)
    vlines!(ax, r.kdiss; color = (:gray, 0.5), linestyle = :dash)
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

    # Save plot
    file = "$(plotdir)/spectrum-dns.pdf"
    @info "Saving DNS spectrum to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

function plot_spectrum_les(setup, data, les_stat)
    (; D) = setup
    s_ref = mean(data.spectra_les)
    s_les = map(stat -> mean(stat.s), les_stat)
    r = spectrum_reference(setup, data.statistics_dns)
    k = 2π / setup.l * eachindex(s_ref)
    labels = getlabels()

    fig = Figure(; size = (400, 360))
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        xlabel = r.xlabel,
        ylabel = r.ylabel,
        xlabelsize = 20,
        ylabelsize = 20,
    )
    lines!(ax, r.kscale * k, r.escale * s_ref; label = "Reference")
    for key in keys(s_les)
        lines!(ax, r.kscale * k, r.escale * s_les[key]; label = labels[key])
    end
    lines!(ax, r.k_ref, r.E_ref; label = r.label)
    vlines!(ax, r.kdiss; color = (:gray, 0.5), linestyle = :dash)
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

    # Save plot
    file = "$(setup.plotdir)/spectrum-les.pdf"
    @info "Saving LES spectrum plot to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end
