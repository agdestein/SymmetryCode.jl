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
        (m, load_object(psfile(case, m)).losses_valid)
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
Taylor-Green dissipation benchmark: two panels in standard nondimensional units
(`t* = t V0 / L`), left = kinetic energy `E_k / V0²`, right = dissipation rate
`ε L / V0³`. Overlays, for the full transition:

- the full-grid DNS (`data.statistics_dns`: `e`, `diss`) — the gold reference;
- the published `Re = 1600` curve from `reffile` (if present);
- per-key LES results as the *effective* dissipation `ε_visc + ε_sfs` evaluated
  a-posteriori on each rollout (`budget_<k>.jld2`), where `:ref` is the
  filtered-DNS budget and `:nomo` is resolved-viscous only (`ε_sfs = 0`).

`keys` should include `:ref` and the closure keys. Requires `setup.V0`.
"""
function plot_dissipation_tgv(
        setup, keys;
        reffile = joinpath(@__DIR__, "..", "reference", "tgv_re1600.csv"),
    )
    (; outdir, plotdir, V0) = setup
    labels = getlabels()
    styles = getstyles()
    data = joinpath(outdir, "data.jld2") |> load_object

    fig = Figure(; size = (820, 360))
    ax_e = Axis(
        fig[1, 1];
        # xlabel = L"t^* = t V_0 / L",
        # ylabel = L"E_k / V_0^2",
        xlabel = "Time",
        ylabel = "Kinetic energy",
    )
    ax_eps = Axis(
        fig[1, 2];
        # xlabel = L"t^* = t V_0 / L",
        # ylabel = L"\varepsilon\, L / V_0^3",
        xlabel = "Time",
        ylabel = "Dissipation rate",
    )

    # Full-grid DNS over the whole trajectory (gold reference).
    tdns = data.times .* V0
    lines!(ax_e, tdns, [s.e for s in data.statistics_dns] ./ V0^2; color = :black, label = labels.dns)
    lines!(ax_eps, tdns, [s.diss for s in data.statistics_dns] ./ V0^3; color = :black, label = labels.dns)

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
    for k in keys
        b = load_object("$(outdir)/budget_$(k).jld2")
        t = b.t .* V0
        kw = (; label = labels[k], color = styles[k].color, linestyle = styles[k].linestyle)
        lines!(ax_e, t, b.ke ./ V0^2; kw...)
        lines!(ax_eps, t, (b.eps_visc .+ b.eps_sfs) ./ V0^3; kw...)
    end

    Legend(
        fig[0, :], ax_eps;
        tellwidth = false, tellheight = true, framevisible = false,
        horizontal = true, nbanks = 4,
    )
    rowgap!(fig.layout, 5)
    file = "$(plotdir)/dissipation-tgv.pdf"
    @info "Saving Taylor-Green dissipation plot to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Cross-Reynolds aggregation for the Taylor-Green generalization sweep.

`setups` is a vector of per-Re Taylor-Green setups (each carrying `Re_target`
and its own `outdir` with `sfs_stats_<k>.jld2` artifacts). For each model in
`keys` (excluding `:ref`/`:nomo`) we plot two a-priori trends against the
integral Reynolds number:
- left: the **over-dissipation ratio** `median(τ:S) / median(τ:S)_ref`, the
  diagnostic that exposes regime mis-calibration (ratio → 1 = correctly
  dissipative; the learned closures, trained on forced HIT, drift above 1 as
  the test flow becomes less turbulent);
- right: the a-priori relative SFS tensor error, to show that the tensor
  *structure* stays accurate even where the dissipation *magnitude* drifts.

The Reynolds number on the axis is the *measured* integral-scale `Re_int` from
[`turbulence_statistics`](@ref), computed identically for the test flows and
the anchor so they share one axis. Each non-stationary TGV setup is summarized
by its **peak-dissipation** instant (the most turbulent state, where ε is
well-defined — `Re_int = u'⁴/(εν)` diverges in the laminar/decay phases — and
which dominates the SFS dissipation the ratio measures), read from that
setup's `data.jld2`.

`train_anchor`, if given, is the forced *training setup*; its **eval-window
mean** `Re_int` (the stationary value) positions the star markers overlaying
the training regime's ratios/errors (from `train_anchor.outdir`'s `sfs_stats`)
— the point the trends should extrapolate toward. Missing artifacts are
skipped with a warning so the plot still renders from whatever is present.

The dissipation ratio is drawn on a log axis so over- (2×) and
under-dissipation (0.5×) sit symmetrically about the reference line. With
`seeds`, per-point whiskers show ± one std over the training seeds whose
`sfs_stats_<seed_key>.jld2` are present in each setup's outdir.
"""
function plot_dissipation_vs_re(
        setups, keys;
        train_anchor = nothing,
        plotdir = last(setups).plotdir,
        Re_key = :Re_int,
        seeds = nothing,
    )
    labels = getlabels()
    styles = getstyles()
    plot_keys = filter(k -> k ∉ (:ref, :nomo), collect(keys))

    Re_labels = (;
        Re_int = "Integral Reynolds number",
        Re_tay = "Taylor Reynolds number",
    )

    # Per-setup loader: (ratio, relerr) for one model key, or `nothing` if its
    # stats (or the :ref baseline) are missing in that setup's outdir.
    load_metrics(outdir, k) = let
        f, fref = "$(outdir)/sfs_stats_$(k).jld2", "$(outdir)/sfs_stats_ref.jld2"
        if !isfile(f) || !isfile(fref)
            @warn "Missing SFS stats for $(k) in $(outdir); skipping point"
            nothing
        else
            s = load_object(f)
            (; ratio = s.diss.median / load_object(fref).diss.median, relerr = s.apriori.relerr)
        end
    end

    # Per-setup seed spread (std over seeds) of the same two metrics, or
    # `nothing` when fewer than two seed artifacts are present.
    seed_spread(s, k) = let
        isnothing(seeds) && return nothing
        m = filter(
            !isnothing,
            [load_metrics(s.outdir, seed_key(s, k, sd)) for sd in seeds],
        )
        length(m) < 2 ? nothing :
            (; ratio = std(x.ratio for x in m), relerr = std(x.relerr for x in m))
    end

    # Measured integral Reynolds number per flow (one definition for both the
    # axis points and the anchor). The TGV is non-stationary, so each setup is
    # summarized by its peak-dissipation instant; loading the (large) data.jld2
    # is acceptable since the sweep plot is produced once.
    peak_re_int(outdir) = let
        d = load_object("$(outdir)/data.jld2")
        re = argmax(s -> s.diss, d.statistics_dns)[Re_key]
        d = nothing
        GC.gc()
        re
    end

    fig = Figure(; size = (820, 360))
    ax_d = Axis(
        fig[1, 1];
        xlabel = Re_labels[Re_key],
        ylabel = "Median SFS dissipation / reference",
        yscale = log10,
        yticks = ([0.5, 1.0, 1.5, 2.0, 2.5], ["0.5", "1", "1.5", "2", "2.5"]),
    )
    ax_e = Axis(fig[1, 2]; xlabel = Re_labels[Re_key], ylabel = "A-priori relative SFS error")

    re_all = [peak_re_int(s.outdir) for s in setups]
    order = sortperm(re_all)
    sorted = setups[order]
    res = re_all[order]
    for k in plot_keys
        m = [load_metrics(s.outdir, k) for s in sorted]
        keep = .!isnothing.(m)
        any(keep) || continue
        kw = (; label = labels[k], color = styles[k].color, marker = styles[k].marker)
        scatterlines!(ax_d, res[keep], [x.ratio for x in m[keep]]; kw...)
        scatterlines!(ax_e, res[keep], [x.relerr for x in m[keep]]; kw...)
        sp = [seed_spread(s, k) for s in sorted]
        keepsp = keep .& .!isnothing.(sp)
        if any(keepsp)
            errorbars!(
                ax_d, res[keepsp], [x.ratio for x in m[keepsp]],
                [x.ratio for x in sp[keepsp]];
                whiskerwidth = 6, color = styles[k].color,
            )
            errorbars!(
                ax_e, res[keepsp], [x.relerr for x in m[keepsp]],
                [x.relerr for x in sp[keepsp]];
                whiskerwidth = 6, color = styles[k].color,
            )
        end
    end

    # Forced-training anchor: the regime the learned closures were fit to, drawn
    # as star markers (one shared legend entry) colored to match each model — the
    # point the per-model trends should extrapolate toward. All stars sit at the
    # forced case's eval-window mean Re_int (its stationary integral Reynolds
    # number), the like-for-like counterpart of the TGV peak Re_int.
    if !isnothing(train_anchor)
        anchor = [
            (k, m) for (k, m) in
                ((k, load_metrics(train_anchor.outdir, k)) for k in plot_keys)
                if !isnothing(m)
        ]
        if !isempty(anchor)
            d = load_object("$(train_anchor.outdir)/data.jld2")
            anchor_re = mean(s -> s[Re_key], d.statistics_dns[data_ranges(train_anchor).eval])
            d = nothing
            GC.gc()
            re = fill(anchor_re, length(anchor))
            label = "Training (Re=$(round(Int, anchor_re)))"
            color = [styles[k].color for (k, _) in anchor]
            scatter!(ax_d, re, [m.ratio for (_, m) in anchor]; marker = :star5, markersize = 14, color, label)
            scatter!(ax_e, re, [m.relerr for (_, m) in anchor]; marker = :star5, markersize = 14, color)
        end
    end

    hlines!(ax_d, [1.0]; color = :black, linestyle = :dash, label = "Reference")
    Legend(
        fig[0, :], ax_d;
        tellwidth = false, tellheight = true, framevisible = false,
        horizontal = true,
        nbanks = 7,
    )
    rowgap!(fig.layout, 5)

    file = "$(plotdir)/dissipation-vs-$(Re_key).pdf"
    @info "Saving Taylor-Green Reynolds-sweep plot to $(file)"
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

function plot_velocities(setup, comp, modelkeys)
    (; D, l, n_les, backend) = setup

    data = joinpath(setup.outdir, "data.jld2") |> load_object

    # Both reference (filtered DNS) and model rollouts live on the eval window.
    eval_range = data_ranges(setup).eval
    times_eval = data.times[eval_range]
    inputs_eval = data.inputs[eval_range]

    # Width scales with the number of model columns (ref + closures).
    fig = Figure(; size = (115 * length(modelkeys), 470))
    g = Grid{D}(; l, n = n_les, backend)
    # `comp = :vortz` plots the out-of-plane vorticity ω_z = ∂_x u_y - ∂_y u_x
    # (the in-plane swirl on the shown z-slice) instead of a velocity component.
    vortz = comp === :vortz
    ubar = vortz ? vectorfield(g) : nothing
    ui = scalarfield(g)
    ui_space = spacescalarfield(g)
    plan = plan_rfft(ui_space)
    labels = getlabels()
    nrow = 4
    ntime = length(times_eval)
    time_inds = map(x -> round(Int, x), range(1, ntime, nrow + 1))[2:end]

    # Loop over figure columns
    for (k, key) in enumerate(modelkeys)
        @info "Plotting velocity for $(key)"
        flush(stderr)

        title = labels[key]
        useries = if key == :ref
            inputs_eval
        else
            upost = load_object(upostfile(setup, key))
            upost.u
        end

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

    save("$(setup.plotdir)/velocities-$(comp).png", fig; backend = CairoMakie)

    return fig
end

"""
    plot_field_evolution_tgv(setup; field = :vortz, ntime = 5, zslice = 1)

Time-evolution montage of the **filtered DNS** field for the decaying Taylor-Green
test: a row of 2D sections sampled from the initial condition, through transition
into turbulence (anchored on the peak-dissipation snapshot), and into the decay.
Only the filtered velocity is stored (`data.inputs`, spectral `ubar`), so that is
what is shown — either a velocity component (`field ∈ (:x, :y, :z)`) or the
out-of-plane vorticity `ω_z = ∂_x u_y - ∂_y u_x` (`field = :vortz`, the default;
it sharply marks the roll-up into turbulence). In 3D a `z = zslice` section is
taken. A single shared, zero-centered color range across panels makes the decay
of the field amplitude legible. Mirrors `plot_velocities` in conventions.
"""
function plot_field_evolution_tgv(setup; field = :vortz, ntime = 5, zslice = 1)
    (; D, l, n_les, backend, V0) = setup

    data = joinpath(setup.outdir, "data.jld2") |> load_object
    # Dimensionless convective time t* = t·V₀/L (L = 1), matching the rest of the
    # TGV pipeline (e.g. `plot_dissipation_tgv`) and comparable across the Re sweep.
    times = (data.times .- data.times[1]) .* V0
    nt = length(times)

    # Sample the IC, the peak-dissipation snapshot (transition into turbulence),
    # and evenly spaced snapshots through the post-peak decay.
    diss = [s.diss for s in data.statistics_dns]
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
            copyto!(ubar[c], data.inputs[t][c])
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

    save("$(setup.plotdir)/field-evolution-$(field).png", fig; backend = CairoMakie)

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

function plot_sfs(setup, modelkeys)
    (; D, l, n_les, backend) = setup

    @assert D == 3 "TODO: Make this plot 2D compatible"

    data = joinpath(setup.outdir, "data.jld2") |> load_object
    eval_range = data_ranges(setup).eval

    g = Grid{D}(; l, n = n_les, backend)

    τ = spacetensorfield(g)
    τhat = scalarfield(g)
    plan = plan_rfft(τ.xx)

    # Snapshot index inside the eval window (sfs_*.jld2 is eval-only).
    t = min(30, length(eval_range))

    τcpu = data.outputs[eval_range[t]]
    for (τ, τcpu) in zip(τ, τcpu)
        copyto!(τhat, τcpu)
        apply!(twothirds!, g, (τhat, g))
        to_phys!(τ, τhat, plan, g)
    end
    τ_ref = τ |> cpu_device()

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
    times, stats = load("$(setup.outdir)/dns.jld2", "times", "statistics")
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
    file = joinpath(setup.plotdir, "evolution-dns.pdf")
    @info "Saving DNS time series plot to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

function plot_evolution_data(setup)
    (; D) = setup

    data = joinpath(setup.outdir, "data.jld2") |> load_object

    times_warmup, stats_warmup = load("$(setup.outdir)/dns.jld2", "times", "statistics")
    times_warmup .-= times_warmup[end] # Use negative times for warmup
    energies_warmup = map(s -> s.e, stats_warmup)
    dissipations_warmup = map(s -> s.diss, stats_warmup)
    Re_tay_warmup = map(s -> s.Re_tay, stats_warmup)
    t_int_warmup = map(s -> s.t_int, stats_warmup)

    times = data.times
    energies = map(s -> s.e, data.statistics_dns)
    dissipations = map(s -> s.diss, data.statistics_dns)
    Re_tay = map(s -> s.Re_tay, data.statistics_dns)
    t_int = map(s -> s.t_int, data.statistics_dns)

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
    file = joinpath(setup.plotdir, "evolution-data.pdf")
    @info "Saving energy and dissipation time series plot to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

function plot_dissipation_finite_difference(setup)
    times, stats = load("$(setup.outdir)/dns.jld2", "times", "statistics")
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
    file = joinpath(setup.plotdir, "dissipation_finite_difference.pdf")
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

function plot_spectrum_data(setup)
    (; D, l, n_dns, backend) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)

    data = joinpath(setup.outdir, "data.jld2") |> load_object

    s_dns = mean(data.spectra_dns)
    s_les = mean(data.spectra_les)
    r = spectrum_reference(setup, mean_of_named_tuple_series(data.statistics_dns))

    k_dns = 2π / setup.l * eachindex(s_dns)
    k_les = 2π / setup.l * eachindex(s_les)

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

    save(joinpath(setup.plotdir, "spectrum-data.pdf"), fig; backend = CairoMakie)
    return fig
end

function plot_spectrum_dns(setup)
    (; outdir, plotdir, D, l, n_dns, n_les, backend, visc) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)
    statistics = load("$(outdir)/dns.jld2", "statistics")
    u = load("$(outdir)/dns.jld2", "u") |> adapt(backend)
    ubar = vectorfield(g_les)
    for (ubar, u) in zip(ubar, u)
        apply!(cutoff!, g_les, (ubar, u))
        apply!(gaussianfilter!, g_les, (ubar, setup.Δ, g_les))
    end
    D = dim(g_dns)
    stuff_dns = spectral_stuff(g_dns)
    stuff_les = spectral_stuff(g_les)
    stat = statistics[end]
    s_dns = spectrum(u, g_dns, stuff_dns)
    s_les = spectrum(ubar, g_les, stuff_les)
    r = spectrum_reference(setup, stat)

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
    file = "$(plotdir)/spectrum-dns.pdf"
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
Write a paper-ready LaTeX `tabular` of the aggregate per-model metrics to
`<plotdir>/<filename>`, with mean ± std over training seeds for every model
that `seed_stat` (from [`get_seed_statistics_cached`](@ref)) covers.

Columns: a-priori SFS tensor error, a-priori cross-correlation, time-mean
a-posteriori solution error, mean a-priori equivariance error (skipped with
`include_equi = false`, e.g. for the Taylor-Green case where it is not
computed), median pointwise SFS dissipation normalized by the reference, and
the local backscatter fraction. Cells whose artifact is missing print `--`.

Each column is printed at a *uniform* decimal precision: a per-column floor,
widened so the cell with the smallest (seed std / value) still shows its leading
significant figure, so seeded and non-seeded rows line up and nothing is rounded
away to `0.000`. Values too small for that (the O(1e-15) equivariance errors)
fall back to shared-exponent `\\e{n}` scientific notation per cell.

Reads `sfs_stats_<key>.jld2`, `les_stat.jld2`, and
`equi-errors-prior-<key>.jld2` for the canonical values. Copy the output over
the corresponding `tables/*.tex` in the paper repository (the reference
backscatter fraction, needed in the caption, is written as a comment).
"""
function write_errors_table(
        setup, mkeys;
        seed_stat = nothing, include_equi = true, filename = "errors.tex",
    )
    (; outdir, plotdir) = setup
    labels = getlabels()
    refstats = load_object("$(outdir)/sfs_stats_ref.jld2")
    refmed = refstats.diss.median
    lesfile = "$(outdir)/les_stat.jld2"
    les_stat = isfile(lesfile) ? load_object(lesfile) : (;)

    # Values below this many fixed-point decimals print in `\e{}` scientific
    # notation instead of widening a whole column to show them (keeps the
    # O(1e-15) equivariance errors out of the decimal columns).
    maxdec = 6

    seedvals(k, metric) =
    if !isnothing(seed_stat) && haskey(seed_stat, k)
        v = seed_stat[k][metric]
        any(!ismissing, v) ? v : nothing
    else
        nothing
    end
    # Raw cell descriptor: `nothing` (→ "--"), a passthrough LaTeX string
    # (`N.A.` / `\mathrm{NaN}`), or `(; c, s)` with central value `c` and seed
    # std `s` (`nothing` for a single seed / exact reproducibility) — from the
    # seed sweep if present, else the canonical `fallback` artifact (a thunk so
    # missing files are only probed once).
    function cellval(k, metric, fallback)
        v = seedvals(k, metric)
        if isnothing(v)
            c = fallback()
            isnothing(c) && return nothing
            s = nothing
        else
            vv = collect(skipmissing(v))
            isempty(vv) && return nothing
            c = mean(vv)
            s = length(vv) > 1 && std(vv) > 0 ? std(vv) : nothing
        end
        isnan(c) && return "--"  # undefined (e.g. 0/0 cross-corr when a model's predicted stress vanishes)
        return (; c = float(c), s = isnothing(s) ? nothing : float(s))
    end

    # One descriptor per cell, in column order.
    descs = map(collect(mkeys)) do k
        f = "$(outdir)/sfs_stats_$(k).jld2"
        stats = isfile(f) ? load_object(f) : nothing
        fe = "$(outdir)/equi-errors-prior-$(k).jld2"
        ds = Any[
            cellval(k, :relerr, () -> isnothing(stats) ? nothing : stats.apriori.relerr),
            cellval(k, :crosscor, () -> isnothing(stats) ? nothing : stats.apriori.crosscor),
            cellval(k, :e_post, () -> haskey(les_stat, k) ? mean(les_stat[k].e_post) : nothing),
        ]
        include_equi && push!(
            ds,
            k == :nomo ? "N.A." :
                cellval(k, :equi, () -> isfile(fe) ? mean(load_object(fe)) : nothing),
        )
        push!(ds, cellval(k, :diss_median, () -> isnothing(stats) ? nothing : stats.diss.median / refmed))
        push!(ds, cellval(k, :backscatter, () -> isnothing(stats) ? nothing : stats.diss.backscatter))
        ds
    end

    # Per-column uniform precision: each numeric column floors at `bases[j]` and
    # widens to the most decimals any of its fixed-point cells needs to show its
    # leading significant figure (the std's, when seeded). Sub-`maxdec` cells
    # stay scientific and don't participate. Result: every fixed cell in a column
    # prints at one shared decimal place and none collapses to `0.000`.
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
    rows = map(zip(collect(mkeys), cellrows)) do (k, cells)
        padded = join((rpad(cells[j], widths[j]) for j in eachindex(bases)), " & ")
        rstrip("    $(rpad(labels[k], 11)) & " * padded) * " \\\\"
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
    file = joinpath(plotdir, filename)
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
