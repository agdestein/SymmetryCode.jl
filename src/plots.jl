# All plot_* routines and the shared label table they read from.

getlabels() = (;
    dns = "DNS",
    ref = "Filtered DNS",
    nomo = "No-model",
    smag = "Smagorinsky",
    # dynsmag = "Dynamic Smagorinsky",
    dynsmag = "Dyn. Smag.",
    vers = "Verstappen",
    clar = "Clark",
    bard = "Bardina",
    tbnn = "TBNN",
    equi = "G-CNN",
    conv = "MLP",
    convsym = "MLP (sym)",
)

"""
Canonical per-model plot style — the single source of truth, so a model keeps
the same color/linestyle/marker in *every* figure. References are black: the
DNS solid, the filtered-DNS target dashed, so "dashed black" reads as "the
curve to match" throughout the paper. Each entry is
`(; color, linestyle, marker)`.
"""
function getstyles()
    c = Makie.wong_colors()
    # Wong's yellow is too light for thin lines on white; darken it for the
    # symmetrized MLP (the only model beyond the 6 well-separated Wong hues).
    convsym_color = Makie.RGBf(0.72, 0.6, 0.1)
    entry(color, linestyle, marker) = (; color, linestyle, marker)
    return (;
        dns = entry(:black, :solid, :star5),
        ref = entry(:black, :dash, :circle),
        nomo = entry(c[1], :solid, :utriangle),
        smag = entry(c[2], :dot, :pentagon),
        dynsmag = entry(c[2], :solid, :circle),
        vers = entry(c[3], :dot, :hexagon),
        bard = entry(c[3], :dashdot, :star4),
        clar = entry(c[3], :solid, :rect),
        conv = entry(c[4], :solid, :xcross),
        equi = entry(c[5], :solid, :rtriangle),
        tbnn = entry(c[6], :solid, :diamond),
        convsym = entry(convsym_color, :solid, :cross),
    )
end

"""
Display label for a closure `m`: a classical symbol indexes [`getlabels`](@ref)
directly; a learned coordinate `(; arch, …)` takes its architecture's label, with
a `+Re` suffix when the Re_Δ feature is on.
"""
plotlabel(m::Symbol) = getlabels()[m]
plotlabel(m::NamedTuple) = m.use_redelta ? "$(getlabels()[m.arch])+Re" : getlabels()[m.arch]

"""
Plot style for a closure `m` (see [`getstyles`](@ref)). A learned coordinate uses
its architecture's color; the Re_Δ-augmented variant shares that color but is
drawn dashed so the on/off pair reads apart in every figure.
"""
plotstyle(m::Symbol) = getstyles()[m]
plotstyle(m::NamedTuple) =
    m.use_redelta ? (; getstyles()[m.arch]..., linestyle = :dash) : getstyles()[m.arch]

"""
Inertial-range reference spectrum and the normalization that collapses the
data onto the universal `κ̃^{-p}` shape.
3D: Kolmogorov  `E = C ε^{2/3} κ^{-5/3}`,  `κ̃ = κ·l_kol`  (κη).
2D: Kraichnan–Batchelor enstrophy cascade  `E = C η_Ω^{2/3} κ^{-3}`,
    `κ̃ = κ·l_kra`  (κη_ω).
`stats` is a single NamedTuple.
Plotting `(kscale·k, escale·E)` puts the inertial range on the universal
`κ̃^{-p}` line and the dissipation scale at `κ̃ ≈ kdiss = 1`.
"""
function spectrum_reference(setup, stats)
    (; D, l) = setup
    if D == 2
        χ, l_d, p, C = stats.enstrophy_diss, stats.l_kra, 3, 1.4
        xlabel = L"\kappa \eta_\omega"
        ylabel = L"C^{-1} \eta_\Omega^{-2/3} \eta_\omega^{-3}\, E(\kappa)"
        # label = "Kraichnan −3"
        label = "Kraichnan"
    else
        χ, l_d, p, C = stats.diss, stats.l_kol, 5 / 3, 1.6
        xlabel = L"\kappa \eta"
        ylabel = L"C^{-1} \epsilon^{-2/3} \eta^{-5/3}\, E(\kappa)"
        # label = "Kolmogorov −5/3"
        label = "Kolmogorov"
    end
    kscale = l_d
    escale = 1 / C * χ^(-2 / 3) * l_d^(-p)      # = C^{-1} χ^{-2/3} l_d^{-p}
    # Reference drawn from ~2× the forcing-shell wavenumber up to the
    # dissipation scale κ̃ ≈ 1 (normalized units).
    k_f = 2 * (2π / l) * 2 * l_d
    k_ref = logrange(k_f, 1.0, 100)
    E_ref = k_ref .^ (-p)
    return (; kscale, escale, p, C, k_ref, E_ref, kdiss = 1.0, xlabel, ylabel, label)
end

"""
Plot validation-loss curves for every learned-model coordinate in `models` whose
`psfile` is on disk. Classical symbols and models without a persisted artifact are
silently skipped, so the same call works regardless of which subset was trained.
Models span the trainpool, so this is one figure for the whole sweep.
"""
function plot_training(case, models)
    curves = [
        (m, load(psfile(case, m), "losses_valid"))
            for m in models if m isa NamedTuple && isfile(psfile(case, m))
    ]
    isempty(curves) && return nothing

    fig = Figure(; size = (400, 340))
    ax = Axis(fig[1, 1]; xlabel = "Iteration", ylabel = "Loss")
    for (m, c) in curves
        lines!(ax, c; label = plotlabel(m), color = plotstyle(m).color)
    end
    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        tellheight = true,
        framevisible = false,
        horizontal = true,
        nbanks = length(curves),
    )
    rowgap!(fig.layout, 5)
    save("$(case.plotdir)/training.pdf", fig; backend = CairoMakie)
    return fig
end

"""
A-posteriori relative solution error `e_post(t) = ‖u_les − ūbar‖/‖ūbar‖` for each
closure in `models` at evaluation point (dns, Δf), against the reference
large-eddy turnover time `t_int`. Reads [`apostfile`](@ref). `:ref` (≡ 0) is
skipped. (Seed bands and the cross-model inset move to the Re_Δ trend figure.)
"""
function plot_error_post(case, dns, Δf, models)
    t_int = load(dnsmetafile(case, dns), "t_int")
    fig = Figure(; size = (450, 380))
    ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Relative error")
    for m in models
        m === :ref && continue   # the reference error is identically zero
        a = load_object(apostfile(case, dns, Δf, m))
        t = (a.t .- a.t[1]) ./ t_int
        lines!(
            ax, t, a.e_post;
            label = plotlabel(m), color = plotstyle(m).color, linestyle = plotstyle(m).linestyle,
        )
    end
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
    save(joinpath(figdir(case, dns, Δf), "error-post.pdf"), fig; backend = CairoMakie)
    return fig
end

"""
Density of the pointwise SFS dissipation rate `ε_sfs = -τᵢⱼSᵢⱼ` per closure in
`models` at evaluation point (dns, Δf) — the backscatter evidence (mass at
`ε_sfs < 0`). Reads the dissipation KDE from [`sfsstatsfile`](@ref); the τ-component
PDFs were dropped (ReExperiment.md). `:nomo`'s degenerate all-zero samples are
skipped. Drain convention: `ε_sfs > 0` drain, `< 0` backscatter.
"""
function plot_densities(case, dns, Δf, models; dolog = true)
    yscale = dolog ? log10 : identity
    fig = Figure(; size = (450, 340))
    ax = Axis(fig[1, 1]; xlabel = "SFS dissipation rate", ylabel = "Density", yscale)
    for m in models
        dens = load_object(sfsstatsfile(case, dns, Δf, m)).kde.diss
        isempty(dens.x) && continue   # :nomo has degenerate (all-zero) samples
        lines!(
            ax, dens.x, max.(dens.density, 1.0e-16);
            label = plotlabel(m), color = plotstyle(m).color, linestyle = plotstyle(m).linestyle,
        )
    end
    Legend(
        fig[0, 1], ax;
        tellwidth = false, tellheight = true, framevisible = false, orientation = :horizontal,
    )
    rowgap!(fig.layout, 5)
    file = joinpath(figdir(case, dns, Δf), "dissipation-density.pdf")
    @info "Saving density plot to $(file)"
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Bar plot of median pointwise SFS dissipation rate `ε_sfs = -τᵢⱼSᵢⱼ` per
model, normalized by the filtered-DNS reference median. Bars above 1.0 are
over-dissipative (smag/dynsmag); bars below 1.0 are under-dissipative
(clark). The reference is shown as a horizontal dashed line at 1.0.

Drain convention (positive = drain); same as [`compute_sfs_stats`](@ref).
`models` must include `:ref` (the normalization baseline).
"""
function plot_dissipation_bar(case, dns, Δf, models)
    refstats = load_object(sfsstatsfile(case, dns, Δf, :ref)).diss
    plot_models = filter(!=(:ref), collect(models))
    for (momkey, momlabel) in [(:median, "Median"), (:mean, "Mean")]
        ref_med = refstats[momkey]
        normalized = [
            load_object(sfsstatsfile(case, dns, Δf, m)).diss[momkey] / ref_med
                for m in plot_models
        ]
        fig = Figure(; size = (450, 340))
        ax = Axis(
            fig[1, 1];
            xticks = (1:length(plot_models), [plotlabel(m) for m in plot_models]),
            ylabel = "$(momlabel) dissipation",
            xticklabelrotation = π / 6,
        )
        barplot!(
            ax, 1:length(plot_models), normalized;
            bar_labels = :y, color = [plotstyle(m).color for m in plot_models],
        )
        hlines!(ax, [1.0]; color = :black, linestyle = :dash, label = "Reference")
        Legend(
            fig[0, :], ax;
            tellwidth = false, tellheight = true, framevisible = false, horizontal = true,
        )
        ylims!(ax, 0, maximum(normalized) + 0.15)
        rowgap!(fig.layout, 5)
        file = joinpath(figdir(case, dns, Δf), "dissipation-$(momkey)-bar.pdf")
        @info "Saving dissipation bar plot to $(file)"
        save(file, fig; backend = CairoMakie)
    end
    return
end

"""
Bar plot of the local backscatter fraction per model — the fraction of
points where the pointwise SFS dissipation rate `ε_sfs = -τᵢⱼSᵢⱼ` is
negative (equivalently, `τᵢⱼSᵢⱼ > 0`). Smagorinsky-type closures of the
form `τ = -2νₜS` with `νₜ ≥ 0` are mathematically pinned to zero; the
filtered-DNS reference is shown as a horizontal dashed line at its
absolute fraction.

`models` must include `:ref` (drawn as the dashed baseline).
"""
function plot_backscatter_bar(case, dns, Δf, models)
    ref_bs = load_object(sfsstatsfile(case, dns, Δf, :ref)).diss.backscatter
    plot_models = filter(!=(:ref), collect(models))
    bars = [load_object(sfsstatsfile(case, dns, Δf, m)).diss.backscatter for m in plot_models]

    fig = Figure(; size = (520, 340))
    ax = Axis(
        fig[1, 1];
        xticks = (1:length(plot_models), [plotlabel(m) for m in plot_models]),
        ylabel = "Backscatter fraction",
        xticklabelrotation = π / 6,
    )
    barplot!(
        ax, 1:length(plot_models), bars;
        bar_labels = :y, color = [plotstyle(m).color for m in plot_models],
    )
    hlines!(
        ax, [ref_bs];
        color = :black, linestyle = :dash,
        label = "Reference ($(round(ref_bs * 100; digits = 1))%)",
    )
    ylims!(ax, 0, max(maximum(bars; init = 0.0), ref_bs) * 1.15)
    axislegend(ax; position = :rt, framevisible = false)

    file = joinpath(figdir(case, dns, Δf), "backscatter-bar.pdf")
    @info "Saving backscatter bar plot to $(file)"
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Two-panel bar plot of a-priori predictive metrics per model:
relative L² error of the predicted SFS tensor (left, lower is better) and
cross-correlation with the filtered-DNS reference (right, higher is better).

`models` should not include `:ref` (self-comparison is trivial).
"""
function plot_apriori_bar(case, dns, Δf, models)
    plot_models = collect(models)
    stats = [load_object(sfsstatsfile(case, dns, Δf, m)).apriori for m in plot_models]
    re = [s.relerr for s in stats]
    cc = [s.crosscor for s in stats]
    barcolors = [plotstyle(m).color for m in plot_models]

    fig = Figure(; size = (820, 340))
    xtks = (1:length(plot_models), [plotlabel(m) for m in plot_models])
    ax_re = Axis(fig[1, 1]; xticks = xtks, ylabel = "Relative error", xticklabelrotation = π / 6)
    barplot!(ax_re, 1:length(plot_models), re; bar_labels = :y, color = barcolors)
    ax_cc = Axis(fig[1, 2]; xticks = xtks, ylabel = "Cross-correlation", xticklabelrotation = π / 6)
    barplot!(ax_cc, 1:length(plot_models), cc; bar_labels = :y, color = barcolors)

    ylims!(ax_re, 0.0, 1.1)
    ylims!(ax_cc, 0.0, 1.1)

    file = joinpath(figdir(case, dns, Δf), "apriori-bar.pdf")
    @info "Saving a-priori bar plot to $(file)"
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Two-panel plot of the a-posteriori resolved-KE budget over the eval window:
left = `ke(t) = ⟨½ uᵢuᵢ⟩`, right = SFS dissipation rate
`ε_sfs(t) = -⟨τᵢⱼ Sᵢⱼ⟩` (positive = drain on resolved KE; negative = net
backscatter). Same drain convention as [`plot_densities`](@ref),
[`plot_dissipation_bar`](@ref), and [`plot_spectral_transfer`](@ref).

`models` should include `:ref` and any closure. Reads [`apostfile`](@ref).
"""
function plot_budget(case, dns, Δf, models)
    t_int = load(dnsmetafile(case, dns), "t_int")
    fig = Figure(; size = (820, 360))
    ax_ke = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Kinetic energy")
    ax_eps = Axis(fig[1, 2]; xlabel = "Time", ylabel = "SFS dissipation rate")
    for m in models
        a = load_object(apostfile(case, dns, Δf, m))
        t = (a.t .- a.t[1]) ./ t_int
        kw = (; label = plotlabel(m), color = plotstyle(m).color, linestyle = plotstyle(m).linestyle)
        lines!(ax_ke, t, a.ke; kw...)
        lines!(ax_eps, t, a.eps_sfs; kw...)
    end
    Legend(
        fig[0, :], ax_ke;
        tellwidth = false, tellheight = true, framevisible = false, orientation = :horizontal,
    )
    rowgap!(fig.layout, 5)
    file = joinpath(figdir(case, dns, Δf), "budget.pdf")
    @info "Saving budget plot to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Read a published Taylor-Green reference table (whitespace/comma separated,
`#` comments allowed) with columns `t*  E_k  ε`, already in the standard
nondimensional units (`V0 = L = 1`). Returns `nothing` if the file is absent.
"""
function read_tgv_reference(reffile)
    isfile(reffile) || return nothing
    t, E, eps = Float64[], Float64[], Float64[]
    for line in eachline(reffile)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        cols = split(s, r"[,\s]+")
        length(cols) < 3 && continue
        push!(t, parse(Float64, cols[1]))
        push!(E, parse(Float64, cols[2]))
        push!(eps, parse(Float64, cols[3]))
    end
    return (; t, E, eps)
end

"""
Taylor-Green dissipation benchmark for TGV run `tgv` at filter `Δf`: two panels in
standard nondimensional units (`t* = t V0 / L`), left = kinetic energy `E_k / V0²`,
right = dissipation rate `ε L / V0³`. Overlays the full-grid DNS (gold reference,
[`dnsmetafile`](@ref) `statistics_dns`), the published `Re = 1600` curve from
`reffile`, and each closure's *effective* dissipation `ε_visc + ε_sfs` evaluated
a-posteriori on its rollout ([`apostfile`](@ref); `:nomo` is resolved-viscous only).
"""
function plot_dissipation_tgv(
        case, tgv, Δf, models;
        reffile = joinpath(@__DIR__, "..", "reference", "tgv_re1600.csv"),
    )
    times, statistics_dns, V0 = load(dnsmetafile(case, tgv), "times", "statistics_dns", "V0")
    labels = getlabels()

    fig = Figure(; size = (820, 360))
    ax_e = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Kinetic energy")
    ax_eps = Axis(fig[1, 2]; xlabel = "Time", ylabel = "Dissipation rate")

    # Full-grid DNS over the whole trajectory (gold reference).
    tdns = times .* V0
    lines!(ax_e, tdns, [s.e for s in statistics_dns] ./ V0^2; color = :black, label = labels.dns)
    lines!(ax_eps, tdns, [s.diss for s in statistics_dns] ./ V0^3; color = :black, label = labels.dns)

    # Published Re=1600 reference (already nondimensional), if available.
    # Gray dotted, so it cannot be confused with the black dashed filtered-DNS.
    ref = read_tgv_reference(reffile)
    if isnothing(ref)
        @warn "Taylor-Green reference not found at $(reffile); skipping published overlay"
    else
        lines!(ax_e, ref.t, ref.E; color = :gray, linestyle = :dot, label = "Ref. Re=1600")
        lines!(ax_eps, ref.t, ref.eps; color = :gray, linestyle = :dot, label = "Ref. Re=1600")
    end

    # LES closures: effective dissipation ε_visc + ε_sfs on each rollout.
    for m in models
        b = load_object(apostfile(case, tgv, Δf, m))
        t = b.t .* V0
        kw = (; label = plotlabel(m), color = plotstyle(m).color, linestyle = plotstyle(m).linestyle)
        lines!(ax_e, t, b.ke ./ V0^2; kw...)
        lines!(ax_eps, t, (b.eps_visc .+ b.eps_sfs) ./ V0^3; kw...)
    end

    Legend(
        fig[0, :], ax_eps;
        tellwidth = false, tellheight = true, framevisible = false,
        horizontal = true, nbanks = 4,
    )
    rowgap!(fig.layout, 5)
    file = joinpath(figdir(case, tgv, Δf), "dissipation-tgv.pdf")
    @info "Saving Taylor-Green dissipation plot to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Classical-closure metric at one eval point (no seed spread): `:diss_median` is the
median SFS dissipation normalized by the `:ref` median, `:relerr` the a-priori
tensor error, `:e_post` the time-mean a-posteriori solution error.
"""
function classical_metric(case, dns, Δf, c, metric)
    if metric === :diss_median
        ref = load_object(sfsstatsfile(case, dns, Δf, :ref)).diss.median
        return load_object(sfsstatsfile(case, dns, Δf, c)).diss.median / ref
    elseif metric === :relerr
        return load_object(sfsstatsfile(case, dns, Δf, c)).apriori.relerr
    else # :e_post
        return mean(load_object(apostfile(case, dns, Δf, c)).e_post)
    end
end

"""
The H2 deliverable: three trends against the global filter-scale Reynolds number
`Re_Δ` (per eval point = the mean over the test series of the `redelta` stored in
[`fieldsfile`](@ref)) — median SFS dissipation ratio (→ 1 ideal, log axis), the
a-priori relative SFS tensor error, and the time-mean a-posteriori solution error.

`evalpoints` is a list of `(dns, Δf)` evaluation points; `families` a list of
learned-model coordinates `(; arch, tier, use_redelta)`, each drawn in its
architecture's color, solid for `use_redelta = false` and dashed for `true`, so
the on/off pair shows whether the Re_Δ feature buys generalization. Per-point
whiskers are ± one std over `netseeds` (from the cached [`get_seed_statistics`](@ref)
aggregate). `classical` closures are overlaid as single-value reference trends. If
`trainpoints` is given, the training Re_Δ span is shaded so the OOD region reads.
"""
function plot_trend_vs_redelta(
        case, evalpoints, families;
        netseeds = 0:0, classical = (:clar,), trainpoints = nothing,
    )
    redelta_of(dns, Δf) = mean(load(fieldsfile(case, dns, Δf), "redelta"))
    re = [redelta_of(dns, Δf) for (dns, Δf) in evalpoints]
    agg = [get_seed_statistics(case, families, dns, Δf, netseeds) for (dns, Δf) in evalpoints]
    order = sortperm(re)
    re, agg, evalpoints = re[order], agg[order], evalpoints[order]

    xlab = "Filter-scale Reynolds number"
    fig = Figure(; size = (1100, 380))
    ax_diss = Axis(
        fig[1, 1];
        xscale = log10, yscale = log10, xlabel = xlab,
        ylabel = "Median SFS dissipation / reference",
    )
    ax_re = Axis(fig[1, 2]; xscale = log10, xlabel = xlab, ylabel = "A-priori relative SFS error")
    ax_post = Axis(fig[1, 3]; xscale = log10, xlabel = xlab, ylabel = "A-posteriori solution error")
    axs = (ax_diss, ax_re, ax_post)
    metrics = (:diss_median, :relerr, :e_post)

    # Shade the training Re_Δ span so in-distribution vs OOD is legible.
    if !isnothing(trainpoints)
        tre = [redelta_of(dns, Δf) for (dns, Δf) in trainpoints]
        for ax in axs
            vspan!(ax, minimum(tre), maximum(tre); color = (:gray, 0.12))
        end
    end

    for fam in families
        fname = familyname(fam)
        st = plotstyle(fam)
        for (ax, metric) in zip(axs, metrics)
            xs, yc, ys = Float64[], Float64[], Float64[]
            for (i, a) in enumerate(agg)
                haskey(a, fname) || continue
                v = collect(skipmissing(getproperty(a[fname], metric)))
                isempty(v) && continue
                push!(xs, re[i])
                push!(yc, mean(v))
                push!(ys, length(v) > 1 ? std(v) : 0.0)
            end
            isempty(xs) && continue
            scatterlines!(
                ax, xs, yc;
                label = plotlabel(fam), color = st.color, marker = st.marker, linestyle = st.linestyle,
            )
            isp = findall(>(0), ys)
            isempty(isp) ||
                errorbars!(ax, xs[isp], yc[isp], ys[isp]; whiskerwidth = 6, color = st.color)
        end
    end

    for c in classical
        st = plotstyle(c)
        for (ax, metric) in zip(axs, metrics)
            ys = [classical_metric(case, dns, Δf, c, metric) for (dns, Δf) in evalpoints]
            scatterlines!(
                ax, re, ys;
                label = plotlabel(c), color = st.color, marker = st.marker, linestyle = st.linestyle,
            )
        end
    end

    hlines!(ax_diss, [1.0]; color = :black, linestyle = :dash, label = "Reference")
    Legend(
        fig[0, :], ax_diss;
        tellwidth = false, tellheight = true, framevisible = false,
        orientation = :horizontal, nbanks = 3,
    )
    rowgap!(fig.layout, 5)
    file = "$(case.plotdir)/trend-vs-redelta.pdf"
    @info "Saving Re_Δ trend figure to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Spectral SFS dissipation rate `ε_sfs(k)` (positive = drain at that shell;
negative = local backscatter at that shell) averaged over the eval window.
Compares the closure's spectral footprint against the filtered-DNS
reference; reveals whether a model dissipates at the correct wavenumbers
(e.g. Smag's bias toward over-dissipation near the cutoff). Same drain
convention as [`plot_budget`](@ref) and [`plot_densities`](@ref). Reads the
`transfer` entry of [`apostfile`](@ref).

The axes match the energy-spectrum plots: wavenumber rescaled by the
small-scale length (Kolmogorov η in 3D, Kraichnan η_K in 2D); dissipation
rescaled by the mean total energy dissipation rate `ε`, so `Σ_k ε_sfs(k)/ε ≈ 1`
for a balanced closure.
"""
function plot_spectral_transfer(case, dns, Δf, models)
    stats = mean_of_named_tuple_series(load(dnsmetafile(case, dns), "statistics_dns"))
    r = spectrum_reference(case, stats)
    ε = stats.diss

    fig = Figure(; size = (450, 360))
    ax = Axis(fig[1, 1]; xscale = log10, xlabel = "Wavenumber", ylabel = "SFS dissipation rate")
    for m in models
        m === :nomo && continue
        t = load_object(apostfile(case, dns, Δf, m)).transfer
        lines!(
            ax, r.kscale * t.k, t.eps_sfs ./ ε;
            label = plotlabel(m), color = plotstyle(m).color, linestyle = plotstyle(m).linestyle,
        )
    end
    hlines!(ax, [0.0]; color = :gray, linestyle = :dot)
    Legend(
        fig[0, 1], ax;
        tellwidth = false, tellheight = true, framevisible = false,
        horizontal = true, nbanks = 4,
    )
    rowgap!(fig.layout, 5)
    file = joinpath(figdir(case, dns, Δf), "spectral-transfer.pdf")
    @info "Saving spectral transfer plot to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Velocity (or out-of-plane vorticity, `comp = :vortz`) slices over the rollout for
the **showcase** eval point (dns, Δf): `:ref` uses the filtered-DNS series
([`fieldsfile`](@ref) `inputs`), every other model its [`apostfieldsfile`](@ref)
field series (so the model must have been run with `savefields = true`). Each
column is a closure, each row a snapshot.
"""
function plot_velocities(case, dns, Δf, models, comp)
    (; D, l, n_les, backend) = case

    times_eval = load(dnsmetafile(case, dns), "times")
    inputs_eval = load(fieldsfile(case, dns, Δf), "inputs")

    # Width scales with the number of model columns (ref + closures).
    fig = Figure(; size = (115 * length(models), 470))
    g = Grid{D}(; l, n = n_les, backend)
    # `comp = :vortz` plots the out-of-plane vorticity ω_z = ∂_x u_y - ∂_y u_x
    # (the in-plane swirl on the shown z-slice) instead of a velocity component.
    vortz = comp === :vortz
    ubar = vortz ? vectorfield(g) : nothing
    ui = scalarfield(g)
    ui_space = spacescalarfield(g)
    plan = plan_rfft(ui_space)
    nrow = 4
    ntime = length(times_eval)
    time_inds = map(x -> round(Int, x), range(1, ntime, nrow + 1))[2:end]

    # Loop over figure columns
    for (k, m) in enumerate(models)
        @info "Plotting velocity for $(modelname(m))"
        flush(stderr)

        title = plotlabel(m)
        useries = m === :ref ? inputs_eval : load_object(apostfieldsfile(case, dns, Δf, m)).u

        # Make all plots in current column
        for (i, t) in enumerate(time_inds)
            # Don't plot anything if series exploded before current time
            t > length(useries) && continue

            ax = Axis(
                fig[i, k];
                ylabel = "t = $(round(times_eval[t] - times_eval[1]; sigdigits = 2))",
                ylabelvisible = k == 1,
                xticksvisible = false,
                xticklabelsvisible = false,
                yticksvisible = false,
                yticklabelsvisible = false,
                aspect = DataAspect(),
                title,
                titlevisible = i == 1,
            )

            spec = if vortz
                # ω_z needs the full velocity field; copy onto the grid/backend.
                for c in keys(ubar)
                    copyto!(ubar[c], useries[t][c])
                end
                apply!(vorticity_z!, g, (ui, ubar, g))
                ui
            else
                useries[t][comp] |> adapt(backend)
            end
            apply!(twothirds!, g, (spec, g))
            to_phys!(ui_space, spec, plan, g)
            range = (:, :)
            # range = (40:60, 40:60)
            slice = if D == 2
                ui_space[range...]
            else
                ui_space[range..., end]
            end |> Array

            image!(ax, slice; colormap = :RdBu, interpolate = false)
        end
    end

    rowgap!(fig.layout, 10)
    colgap!(fig.layout, 10)

    save(joinpath(figdir(case, dns, Δf), "velocities-$(comp).png"), fig; backend = CairoMakie)

    return fig
end

"""
    plot_field_evolution_tgv(case, tgv, Δf; field = :vortz, ntime = 5, zslice = 1)

Time-evolution montage of the filtered-DNS field for TGV run `tgv` at filter `Δf`:
a row of 2D sections from the initial condition, through transition into turbulence
(anchored on the peak-dissipation snapshot), and into the decay. Shows either a
velocity component (`field ∈ (:x, :y, :z)`) or the out-of-plane vorticity
`ω_z = ∂_x u_y - ∂_y u_x` (`field = :vortz`, default; it sharply marks the roll-up).
In 3D a `z = zslice` section is taken. A shared, zero-centered color range makes the
amplitude decay legible. Reads [`fieldsfile`](@ref) `inputs` + [`dnsmetafile`](@ref).
"""
function plot_field_evolution_tgv(case, tgv, Δf; field = :vortz, ntime = 5, zslice = 1)
    (; D, l, n_les, backend) = case

    inputs = load(fieldsfile(case, tgv, Δf), "inputs")
    times, statistics_dns, V0 = load(dnsmetafile(case, tgv), "times", "statistics_dns", "V0")
    # Dimensionless convective time t* = t·V₀/L (L = 1), matching the rest of the
    # TGV pipeline (e.g. `plot_dissipation_tgv`).
    times = (times .- times[1]) .* V0
    nt = length(times)

    # Sample the IC, the peak-dissipation snapshot (transition into turbulence),
    # and evenly spaced snapshots through the post-peak decay.
    diss = [s.diss for s in statistics_dns]
    ipk = argmax(diss)
    inds = round.(Int, [1; ipk; range(ipk, nt; length = ntime - 1)[2:end]])
    inds = sort(unique(clamp.(inds, 1, nt)))

    g = Grid{D}(; l, n = n_les, backend)
    ubar = vectorfield(g)
    ω = scalarfield(g)
    f_space = spacescalarfield(g)
    plan = plan_rfft(f_space)

    # First pass: build the physical-space sections and a shared symmetric range.
    slices = map(inds) do t
        for c in keys(ubar)
            copyto!(ubar[c], inputs[t][c])
        end
        spec = if field === :vortz
            apply!(vorticity_z!, g, (ω, ubar, g))
            ω
        else
            ubar[field]
        end
        apply!(twothirds!, g, (spec, g))
        to_phys!(f_space, spec, plan, g)
        sl = D == 2 ? f_space[:, :] : f_space[:, :, clamp(zslice, 1, n_les)]
        return Array(sl)
    end
    amp = maximum(s -> maximum(abs, s), slices)
    colorrange = (-amp, amp)

    fig = Figure(; size = (180 * length(inds) + 80, 230))
    local hm
    for (i, t) in enumerate(inds)
        ax = Axis(
            fig[1, i];
            # title = L"t^* = %$(round(times[t]; sigdigits = 2))",
            title = "t = $(round(times[t]; sigdigits = 2))",
            xticksvisible = false,
            xticklabelsvisible = false,
            yticksvisible = false,
            yticklabelsvisible = false,
            aspect = DataAspect(),
        )
        hm = image!(ax, slices[i]; colormap = :RdBu, colorrange, interpolate = false)
    end
    # label = field === :vortz ? L"\bar{\omega}_z" : "ubar_$(field)"
    label = field === :vortz ? "Vorticity" : "Velocity"
    Colorbar(fig[1, length(inds) + 1], hm; label)

    rowgap!(fig.layout, 10)
    colgap!(fig.layout, 10)

    save(joinpath(figdir(case, tgv, Δf), "field-evolution-$(field).png"), fig; backend = CairoMakie)

    return fig
end

"""
A-priori equivariance commutation error per octahedral group element, for each
learned model in `models` with an [`equipriorfile`](@ref) on (dns, Δf). The
equivariant closures sit at machine-eps; the non-equivariant MLP is visibly
above. Reads the series persisted by [`apriori_equivariance_error`](@ref).
"""
function plot_equivariance_errors(case, dns, Δf, models)
    fig = Figure(; size = (400, 340))
    ax = Axis(
        fig[1, 1];
        yscale = log10,
        xlabel = "Group element",
        ylabel = "Error",
        xticks = [1, 8, 16, 24, 32, 40, 48],
    )
    ylims!(ax, 1.0e-17, 1)
    for m in models
        m isa NamedTuple || continue                       # learned closures only
        isfile(equipriorfile(case, dns, Δf, m)) || continue
        e = max.(load_object(equipriorfile(case, dns, Δf, m)), 1.0e-30)  # zeros → 1e-30
        scatterlines!(
            ax, eachindex(e), e;
            label = plotlabel(m), marker = plotstyle(m).marker, color = plotstyle(m).color,
        )
    end
    Legend(
        fig[0, 1], ax;
        tellwidth = false, tellheight = true, framevisible = false, horizontal = true, nbanks = 3,
    )
    rowgap!(fig.layout, 5)
    save(joinpath(figdir(case, dns, Δf), "equi-errors.pdf"), fig; backend = CairoMakie)
    return fig
end

"DNS warm-up energy/dissipation time series (from [`dnsfile`](@ref))."
function plot_evolution_dns(case, dns)
    times, stats = load(dnsfile(case, dns), "times", "statistics")
    e = map(s -> s.e, stats)
    diss = map(s -> s.diss, stats)

    # Create plot
    fig = Figure(; size = (400, 340))
    ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Normalized quantity")
    lines!(ax, times, e / maximum(e); label = "Energy")
    lines!(ax, times, diss / maximum(diss); label = "Dissipation")
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
    file = joinpath(dnsfigdir(case, dns), "evolution-dns.pdf")
    @info "Saving DNS time series plot to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Energy/dissipation/Reynolds time series over the data-generation window
([`dnsmetafile`](@ref) `statistics_dns`), with the warm-up tail
([`dnsfile`](@ref)) drawn dashed at negative times.
"""
function plot_evolution_data(case, dns)
    times_warmup, stats_warmup = load(dnsfile(case, dns), "times", "statistics")
    times_warmup = times_warmup .- times_warmup[end] # Use negative times for warmup
    energies_warmup = map(s -> s.e, stats_warmup)
    dissipations_warmup = map(s -> s.diss, stats_warmup)
    Re_tay_warmup = map(s -> s.Re_tay, stats_warmup)
    t_int_warmup = map(s -> s.t_int, stats_warmup)

    times, statistics_dns = load(dnsmetafile(case, dns), "times", "statistics_dns")
    energies = map(s -> s.e, statistics_dns)
    dissipations = map(s -> s.diss, statistics_dns)
    Re_tay = map(s -> s.Re_tay, statistics_dns)
    t_int = map(s -> s.t_int, statistics_dns)

    emax = max(maximum(energies), maximum(energies_warmup))
    dmax = max(maximum(dissipations), maximum(dissipations_warmup))
    Rmax = max(maximum(Re_tay), maximum(Re_tay_warmup))
    tmax = max(maximum(t_int), maximum(t_int_warmup))

    # Create plot
    fig = Figure(; size = (400, 340))

    ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Normalized quantity")
    lines!(ax, times, energies / emax; label = "Energy", color = Cycled(1))
    lines!(ax, times_warmup, energies_warmup / emax; linestyle = :dash, color = Cycled(1))
    lines!(ax, times, dissipations / dmax; label = "Dissipation", color = Cycled(2))
    lines!(ax, times_warmup, dissipations_warmup / dmax; linestyle = :dash, color = Cycled(2))
    lines!(ax, times, Re_tay / Rmax; label = "Taylor Reynolds", color = Cycled(3))
    lines!(ax, times_warmup, Re_tay_warmup / Rmax; linestyle = :dash, color = Cycled(3))
    lines!(ax, times, t_int / tmax; label = "Integral time", color = Cycled(4))
    lines!(ax, times_warmup, t_int_warmup / tmax; linestyle = :dash, color = Cycled(4))
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
    file = joinpath(dnsfigdir(case, dns), "evolution-data.pdf")
    @info "Saving energy and dissipation time series plot to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

"DNS warm-up dissipation vs −dE/dt finite-difference check (from [`dnsfile`](@ref))."
function plot_dissipation_finite_difference(case, dns)
    times, stats = load(dnsfile(case, dns), "times", "statistics")
    e = map(s -> s.e, stats)
    diss = map(s -> s.diss, stats)

    # Create plot
    fig = Figure(; size = (400, 340))
    ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Quantity")
    lines!(ax, times, 6 / 5 * diss; label = "Dissipation")
    lines!(
        ax,
        times[2:end],
        -diff(e) ./ diff(times);
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
    file = joinpath(dnsfigdir(case, dns), "dissipation_finite_difference.pdf")
    @info "Saving DNS dissipation plot to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

"Get named tuple of mean values from a vector of named tuples."
function mean_of_named_tuple_series(series)
    k = keys(series[1])
    n = length(series)
    p = map(k) do k
        m = sum(s -> s[k], series) / n
        k => m
    end
    return NamedTuple(p)
end

"""
Time-averaged DNS vs filtered-DNS energy spectra over the data-generation window
([`dnsmetafile`](@ref) `spectra_dns` + [`lesmetafile`](@ref) `spectra_les`), with
the forcing band shaded.
"""
function plot_spectrum_data(case, dns, Δf)
    (; D, l, n_dns, backend) = case
    g_dns = Grid{D}(; l, n = n_dns, backend)

    spectra_dns, statistics_dns = load(dnsmetafile(case, dns), "spectra_dns", "statistics_dns")
    spectra_les = load(lesmetafile(case, dns, Δf), "spectra_les")

    s_dns = mean(spectra_dns)
    s_les = mean(spectra_les)
    r = spectrum_reference(case, mean_of_named_tuple_series(statistics_dns))

    k_dns = 2π / l * eachindex(s_dns)
    k_les = 2π / l * eachindex(s_les)

    fig = Figure(; size = (400, 340))
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        # xlabel = r.xlabel,
        # ylabel = r.ylabel,
        # xlabelsize = 20,
        # ylabelsize = 20,
        xlabel = "Wavenumber",
        ylabel = "Energy",
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
    # vlines!(ax, r.kdiss; color = (:gray, 0.5), linestyle = :dash)

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

    save(joinpath(figdir(case, dns, Δf), "spectrum-data.pdf"), fig; backend = CairoMakie)
    return fig
end

"""
DNS vs filtered-DNS energy spectrum of the warmed field ([`dnsfile`](@ref)),
filtered at the `Δf` filter width.
"""
function plot_spectrum_dns(case, dns, Δf)
    (; D, l, n_dns, n_les, backend) = case
    Δ = Δf * l / n_les
    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)
    statistics, u = load(dnsfile(case, dns), "statistics", "u")
    u = u |> adapt(backend)
    ubar = vectorfield(g_les)
    for (ubar, u) in zip(ubar, u)
        apply!(cutoff!, g_les, (ubar, u))
        apply!(gaussianfilter!, g_les, (ubar, Δ, g_les))
    end
    stuff_dns = spectral_stuff(g_dns)
    stuff_les = spectral_stuff(g_les)
    stat = statistics[end]
    s_dns = spectrum(u, g_dns, stuff_dns)
    s_les = spectrum(ubar, g_les, stuff_les)
    r = spectrum_reference(case, stat)

    fig = Figure(; size = (400, 340))
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        # xlabel = r.xlabel,
        # ylabel = r.ylabel,
        # xlabelsize = 20,
        # ylabelsize = 20,
        xlabel = "Wavenumber",
        ylabel = "Energy",
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
    # vlines!(ax, r.kdiss; color = (:gray, 0.5), linestyle = :dash)
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
    file = joinpath(figdir(case, dns, Δf), "spectrum-dns.pdf")
    @info "Saving DNS spectrum to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Two-panel LES energy-spectrum comparison. Left: the time-averaged spectra on
the usual log-log axes (the models are nearly indistinguishable there). Right:
the same spectra *divided by the reference spectrum*, on a linear ordinate —
this is where the per-model deviations (Smagorinsky's intermediate-wavenumber
excess, the learned models' high-wavenumber deficit) actually become visible.
The under-dissipative models (No-model, Clark) leave the right panel's frame
through their high-wavenumber pile-up; the axis is clamped so the ±10%
deviations of the well-behaved closures stay readable.
"""
function plot_spectrum_les(case, dns, Δf, models)
    # Reference LES spectrum = filtered-DNS spectra averaged over the test series.
    s_ref = mean(load(lesmetafile(case, dns, Δf), "spectra_les"))
    r = spectrum_reference(case, mean_of_named_tuple_series(load(dnsmetafile(case, dns), "statistics_dns")))
    k = 2π / case.l * eachindex(s_ref)
    styles = getstyles()

    fig = Figure(; size = (820, 360))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "Wavenumber", ylabel = "Energy")
    ax_ratio = Axis(
        fig[1, 2];
        xscale = log10, xlabel = "Wavenumber", ylabel = "Energy relative to reference",
    )
    lines!(
        ax, r.kscale * k, r.escale * s_ref;
        label = "Reference", color = styles.ref.color, linestyle = styles.ref.linestyle,
    )
    for m in models
        m === :ref && continue
        s = mean(load_object(apostfile(case, dns, Δf, m)).spectra_les)
        kw = (; label = plotlabel(m), color = plotstyle(m).color, linestyle = plotstyle(m).linestyle)
        lines!(ax, r.kscale * k, r.escale * s; kw...)
        lines!(ax_ratio, r.kscale * k, s ./ s_ref; kw...)
    end
    hlines!(ax_ratio, [1.0]; color = styles.ref.color, linestyle = styles.ref.linestyle)
    ylims!(ax_ratio, 0, 2)
    Legend(
        fig[0, :], ax;
        tellwidth = false, tellheight = true, framevisible = false, orientation = :horizontal,
    )
    rowgap!(fig.layout, 5)
    file = joinpath(figdir(case, dns, Δf), "spectrum-les.pdf")
    @info "Saving LES spectrum plot to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

# --- Paper-ready LaTeX tables ---

"""
Fixed-point decimal string with exactly `digits` decimals (no Printf dep).
Built from the integer of the scaled value, so it stays correct for small
magnitudes that `string` would otherwise render in scientific notation
(`string(6.0e-5) == \"6.0e-5\"`).
"""
function fixeddecimals(x, digits)
    neg = x < 0
    scaled = round(Int, abs(x) * 10.0^digits)
    s = lpad(string(scaled), digits + 1, '0')
    frac = digits == 0 ? "" : "." * s[(end - digits + 1):end]
    return (neg ? "-" : "") * s[1:(end - digits)] * frac
end

"Minimal number of decimals needed to write `x` exactly (robust vs float `log10` edge cases)."
function _ndecimals(x)
    x == 0 && return 0
    d = 0
    while round(x; digits = d) != x && d < 15
        d += 1
    end
    return d
end

"""
Number of decimals needed to show `x`'s leading significant figure(s) as
fixed-point: `sigdecimals(3.0e-4) == 4`, `sigdecimals(0.25) == 1`,
`sigdecimals(0) == 0`. Drives the per-column precision in
[`write_errors_table`](@ref) so no fixed-point cell is rounded away to `0.000`.
"""
sigdecimals(x; sigdigits = 1) = x == 0 ? 0 : _ndecimals(round(x; sigdigits))

"Fixed-point cell body `c` (or `c \\pm s`; `s === nothing` for a lone value), both at `d` decimals."
function format_fixed(c, s, d)
    isnothing(s) && return fixeddecimals(c, d)
    return "$(fixeddecimals(c, d)) \\pm $(fixeddecimals(s, d))"
end

"""
Scientific cell body `m.mmm\\e{n}` (the paper's `\\e` ×10ⁿ macro), or
`(m.mmm \\pm s.sss)\\e{n}` when seeded — the std shares the mean's exponent so
both align. Used for the O(1e-15) equivariance errors, which sit below any
sensible fixed-point resolution.
"""
function format_sci(c, s)
    e = floor(Int, log10(abs(c)))
    cs = fixeddecimals(c / exp10(e), 3)
    isnothing(s) && return "$(cs)\\e{$(e)}"
    return "($(cs) \\pm $(fixeddecimals(s / exp10(e), 3)))\\e{$(e)}"
end

"""
Write a paper-ready LaTeX `tabular` of the aggregate per-model metrics at
evaluation point (dns, Δf) to `<plotdir>/<filename>`. Learned-model coordinates in
`models` are summarized as mean ± std over `netseeds` (from the cached
[`get_seed_statistics`](@ref) aggregate); classical symbols are single values.

Columns: a-priori SFS tensor error, a-priori cross-correlation, time-mean
a-posteriori solution error, mean a-priori equivariance error (skipped with
`include_equi = false`), median pointwise SFS dissipation normalized by the
reference, and the local backscatter fraction. Cells whose artifact is missing
print `--`.

Each column is printed at a *uniform* decimal precision: a per-column floor,
widened so the cell with the smallest (seed std / value) still shows its leading
significant figure, so seeded and non-seeded rows line up and nothing is rounded
away to `0.000`. Values too small for that (the O(1e-15) equivariance errors) fall
back to shared-exponent `\\e{n}` scientific notation per cell.

Reads [`sfsstatsfile`](@ref), [`apostfile`](@ref) (e_post), [`equipriorfile`](@ref)
(via the seed aggregate). Copy the output over the corresponding `tables/*.tex` in
the paper repo (the reference backscatter fraction, needed in the caption, is a
comment).
"""
function write_errors_table(
        case, dns, Δf, models;
        netseeds = 0:0, include_equi = true, filename = "errors.tex",
    )
    refstats = load_object(sfsstatsfile(case, dns, Δf, :ref))
    refmed = refstats.diss.median
    families = [m for m in models if m isa NamedTuple]
    seed = get_seed_statistics(case, families, dns, Δf, netseeds)

    # Values below this many fixed-point decimals print in `\e{}` scientific
    # notation instead of widening a whole column (keeps O(1e-15) equi errors out).
    maxdec = 6

    # Cell descriptor: `nothing` (→ "--"), "--" for an undefined (NaN) entry, or
    # `(; c, s)` with central value `c` and spread `s` (`nothing` for a single
    # value / exact reproducibility).
    function aggcell(v)
        vv = collect(skipmissing(v))
        isempty(vv) && return nothing
        c = mean(vv)
        isnan(c) && return "--"
        s = length(vv) > 1 && std(vv) > 0 ? std(vv) : nothing
        return (; c = float(c), s)
    end
    function onecell(x)
        isnothing(x) && return nothing
        isnan(x) && return "--"
        return (; c = float(x), s = nothing)
    end

    # One descriptor per cell, in column order, for a learned family or classical.
    function rowdescs(m)
        if m isa NamedTuple
            s = seed[familyname(m)]
            ds = Any[aggcell(s.relerr), aggcell(s.crosscor), aggcell(s.e_post)]
            include_equi && push!(ds, aggcell(s.equi))
            push!(ds, aggcell(s.diss_median))
            push!(ds, aggcell(s.backscatter))
            return ds
        end
        sf = sfsstatsfile(case, dns, Δf, m)
        stats = isfile(sf) ? load_object(sf) : nothing
        fa = apostfile(case, dns, Δf, m)
        ds = Any[
            onecell(isnothing(stats) ? nothing : stats.apriori.relerr),
            onecell(isnothing(stats) ? nothing : stats.apriori.crosscor),
            onecell(isfile(fa) ? mean(load_object(fa).e_post) : nothing),
        ]
        include_equi && push!(ds, m === :nomo ? "N.A." : nothing)   # classical: not equivariance-tested
        push!(ds, onecell(isnothing(stats) ? nothing : stats.diss.median / refmed))
        push!(ds, onecell(isnothing(stats) ? nothing : stats.diss.backscatter))
        return ds
    end
    descs = map(rowdescs, collect(models))

    # Per-column uniform precision: each numeric column floors at `bases[j]` and
    # widens to the most decimals any of its fixed-point cells needs to show its
    # leading significant figure (the std's, when seeded). Sub-`maxdec` cells stay
    # scientific. Result: every fixed cell in a column shares one decimal place.
    bases = include_equi ? [4, 4, 4, 4, 3, 3] : [4, 4, 4, 3, 3]
    isfixed(d) = d isa NamedTuple && (d.c == 0 || sigdecimals(d.c) <= maxdec)
    coldigits = map(eachindex(bases)) do j
        reqs = [
            isnothing(d.s) ? sigdecimals(d.c) : max(sigdecimals(d.c), sigdecimals(d.s))
                for d in getindex.(descs, j) if isfixed(d)
        ]
        max(bases[j], maximum(reqs; init = 0))
    end

    fmt(d, dcol) =
        d === nothing ? "--" :
        d isa AbstractString ? d :
        isfixed(d) ? "\$$(format_fixed(d.c, d.s, dcol))\$" : "\$$(format_sci(d.c, d.s))\$"

    # Format every cell, then pad each column to its widest entry so the source
    # `.tex` stays column-aligned (trailing pad on the last column is trimmed).
    cellrows = [[fmt(ds[j], coldigits[j]) for j in eachindex(bases)] for ds in descs]
    widths = [maximum(length(cr[j]) for cr in cellrows) for j in eachindex(bases)]
    rows = map(zip(collect(models), cellrows)) do (m, cells)
        padded = join((rpad(cells[j], widths[j]) for j in eachindex(bases)), " & ")
        rstrip("    $(rpad(plotlabel(m), 11)) & " * padded) * " \\\\"
    end

    header = [
        "Model",
        "            & Closure \\eqref{eq:tensor-error-prior}",
        "            & Cross-corr.",
        "            & Solution \\eqref{eq:tensor-error-post}",
    ]
    include_equi &&
        push!(header, "            & Equivariance \\eqref{eq:equi-error-prior}")
    append!(
        header, [
            "            & Median diss.",
            "            & Backscatter \\\\",
        ]
    )

    ncol = include_equi ? 6 : 5
    file = joinpath(case.plotdir, filename)
    open(file, "w") do io
        println(io, "% Generated by SymmetryCode write_errors_table; do not edit by hand.")
        println(io, "% Reference backscatter fraction: $(round(refstats.diss.backscatter; digits = 3)).")
        println(io, "\\begin{tabular}{l $(join(fill("r", ncol), " "))}")
        println(io, "    \\toprule")
        foreach(h -> println(io, "    ", h), header)
        println(io, "    \\midrule")
        foreach(r -> println(io, r), rows)
        println(io, "    \\bottomrule")
        println(io, "\\end{tabular}")
    end
    @info "Saving LaTeX errors table to $(file)"
    flush(stderr)
    return file
end

"""
Plot the Phase-0 Re_Δ binning diagnostic produced by
[`compute_redelta_binning`](@ref): median (band = inter-quartile range) of the
two scale-invariant targets vs pointwise `Re_Δ`, with the within-flow slope in
each panel title. A flat line ⇒ no usable within-flow Re_Δ signal; compare the
slope *sign* to `fig:dissipation-vs-re` (`Notes/ReDependence.md`).
"""
function plot_redelta_binning(case, dns, Δf)
    r = load_object(redeltabinningfile(case, dns, Δf))
    fig = Figure(; size = (820, 340))
    panels = (
        (r.diss, r.slope.diss, L"-\tau_{ij}S_{ij}\,/\,(\Delta^2|\bar A|^3)", "Normalized SFS dissipation"),
        (r.stress, r.slope.stress, L"\|\tau\|_F\,/\,(\Delta^2|\bar A|^2)", "Normalized SFS stress"),
    )
    for (col, (st, slope, ylabel, title)) in enumerate(panels)
        ax = Axis(
            fig[1, col];
            xscale = log10,
            # xlabel = L"\mathrm{Re}_\Delta = \Delta^2|\bar A|/\nu",
            xlabel = "Filter-scale Reynolds number",
            # ylabel,
            ylabel = title,
            title = "$(title) — slope $(round(slope; sigdigits = 2))/decade",
        )
        ok = findall(b -> st.count[b] > 0 && isfinite(st.median[b]), eachindex(st.median))
        band!(ax, r.centers[ok], st.q25[ok], st.q75[ok]; color = (:black, 0.12))
        lines!(ax, r.centers[ok], st.median[ok]; color = :black)
        scatter!(ax, r.centers[ok], st.median[ok]; color = :black, markersize = 6)
    end
    file = joinpath(figdir(case, dns, Δf), "redelta-binning.pdf")
    save(file, fig; backend = CairoMakie)
    @info "Saving Re_Δ binning diagnostic to $(file)"
    flush(stderr)
    return fig
end
