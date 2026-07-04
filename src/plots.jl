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
        ref = entry(:black, :dot, :circle),
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
Short capacity-tier tag for compact labels. The grid tiers are named `pN` for a
target parameter count `N` ([`default_tiers`](@ref)); render as `~Nk` (e.g.
`p8000 → ~8k`, `p120 → ~120`). Any other symbol prints verbatim.
"""
function tierlabel(t::Symbol)
    s = string(t)
    startswith(s, "p") && all(isdigit, s[2:end]) || return s
    n = parse(Int, s[2:end])
    return n ≥ 1000 ? "~$(n ÷ 1000)k" : "~$(n)"
end

"""
Bar-chart tick label: [`plotlabel`](@ref) plus the tier tag for a learned family
(the bar plots are the one place several tiers share an axis, so color = arch and
linestyle = +Re no longer disambiguate); classical symbols are left unchanged.
"""
famlabel(m::Symbol) = plotlabel(m)
famlabel(m::NamedTuple) = "$(plotlabel(m)) $(tierlabel(m.tier))"

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
Validation-loss convergence for the learned closures — one panel per architecture,
one band per capacity tier. Within a tier the line is the per-iteration *median*
over `netseeds` and the shaded band is the seed *min–max*, so run-to-run spread is
visible without seed spaghetti. Only the `use_redelta = true` variants are shown
(the capacity sweep); the ±Re pair is training-indistinguishable, so overlaying the
base curves only added the confusing asymmetry that most tiers have no base run.
Per-batch loss is thousands of points, so curves are strided to ~`maxpoints` (the
convergence shape is preserved). Reads `losses_valid` from each [`psfile`](@ref);
classical symbols and models without a persisted artifact are skipped, so the same
call works for any trained subset.
"""
function plot_training(case, models; maxpoints = 1000)
    learned = [m for m in models if m isa NamedTuple && m.use_redelta && isfile(psfile(case, m))]
    isempty(learned) && return nothing
    archs = unique(m.arch for m in learned)
    tiers = unique(m.tier for m in learned)
    tcolor = Dict(t => c for (t, c) in zip(tiers, Makie.wong_colors()))

    function curve(m)
        v = load(psfile(case, m), "losses_valid")
        idx = 1:cld(length(v), maxpoints):length(v)
        return idx, v[idx]
    end

    fig = Figure(; size = (300 * length(archs) + 40, 360))
    for (i, a) in enumerate(archs)
        ax = Axis(
            fig[1, i];
            xlabel = "Iteration", ylabel = "Validation loss",
            ylabelvisible = i == 1, yscale = log10, title = getlabels()[a],
        )
        for t in tiers
            seeds = [m for m in learned if m.arch === a && m.tier === t]
            isempty(seeds) && continue
            curves = [curve(m) for m in seeds]
            x = first(curves[1])
            M = reduce(hcat, [c[2] for c in curves])     # iterations × seeds
            band!(ax, x, vec(minimum(M; dims = 2)), vec(maximum(M; dims = 2)); color = (tcolor[t], 0.25))
            lines!(ax, x, vec(median(M; dims = 2)); color = tcolor[t], linewidth = 1.5)
        end
    end

    # Legend: one entry per capacity tier present.
    elements = [Makie.LineElement(; color = tcolor[t]) for t in tiers]
    Legend(
        fig[0, :], elements, [string(t) for t in tiers];
        tellwidth = false, tellheight = true, framevisible = false,
        orientation = :horizontal, nbanks = 1,
    )
    rowgap!(fig.layout, 5)
    file = "$(case.plotdir)/training.pdf"
    @info "Saving training-curve plot to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Padded `(lo, hi)` axis limits from `vals` (a `pad` fractional margin on the data
span). `anchor`, if given, clamps the lower bound (e.g. `0` for a non-negative
quantity). Used to fix a TGV panel's limits from the *stable* curves only, so a
diverging closure is drawn-then-clipped at the axis box rather than flattening
every other curve into the baseline.
"""
function padded_limits(vals; pad = 0.05, anchor = nothing)
    lo, hi = extrema(vals)
    isnothing(anchor) || (lo = min(lo, anchor))
    m = pad * (hi - lo)
    m == 0 && (m = pad * abs(hi) + eps(float(hi)))
    return (lo - m, hi + m)
end

"""
Whether a decaying-TGV rollout stayed physical: an unforced flow can only lose
resolved kinetic energy, so a series that grows above its initial value (or goes
non-finite) has blown up. Lets an exploding closure stay on the TGV plots (legend
+ clipped curve) without letting it set the axis limits. The `1.05` tolerance
absorbs round-off; forced (non-decaying) runs should not use this test.
"""
tgv_rollout_stable(a) = all(isfinite, a.ke) && maximum(a.ke) <= 1.05 * first(a.ke)

"""
A-posteriori relative solution error `e_post(t) = ‖u_les − ūbar‖/‖ūbar‖` for each
closure in `models` at evaluation point (dns, Δf). The time axis is the TGV
convective time `t* = t V0 / L` for a `:tgv` run (matching
[`plot_dissipation_tgv`](@ref)) and the reference large-eddy turnover `t_int`
otherwise. On the TGV the axis is clipped to the *stable* rollouts
([`tgv_rollout_stable`](@ref)), so a diverged closure (e.g. Clark at coarse Δ)
keeps its curve but does not blow up the y-range. Reads [`apostfile`](@ref);
`:ref` (≡ 0) is skipped. (Seed bands and the cross-model inset move to the Re_Δ
trend figure.)
"""
function plot_error_post(case, dns, Δf, models)
    istgv = dns.role === :tgv
    V0 = istgv ? load(dnsmetafile(case, dns), "V0") : nothing
    t_int = istgv ? nothing : load(dnsmetafile(case, dns), "t_int")
    timeaxis(t) = istgv ? t .* V0 : (t .- t[1]) ./ t_int
    fig = Figure(; size = (450, 380))
    ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Relative error")
    inliers = Float64[]
    for m in models
        m === :ref && continue   # the reference error is identically zero
        a = load_object(apostfile(case, dns, Δf, m))
        lines!(
            ax, timeaxis(a.t), a.e_post;
            label = plotlabel(m), color = plotstyle(m).color, linestyle = plotstyle(m).linestyle,
        )
        istgv && tgv_rollout_stable(a) && append!(inliers, a.e_post)
    end
    istgv && !isempty(inliers) && ylims!(ax, padded_limits(inliers; anchor = 0.0))
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
    # X-range = the contiguous above-floor run around each KDE's mode, not its full
    # support: Smagorinsky's ε ∝ |Ā|³ heavy tail produces a lone, *gap-separated*
    # KDE bump near |Ā|³_max whose bare support stretches the axis into dead
    # whitespace. Restricting to the run containing the peak drops that artifact
    # while keeping the full backscatter shoulder + the positive tail to the floor.
    floorx = 1.0e-4
    xlo, xhi = Inf, -Inf
    for m in models
        dens = load_object(sfsstatsfile(case, dns, Δf, m)).kde.diss
        isempty(dens.x) && continue   # :nomo has degenerate (all-zero) samples
        lines!(
            ax, dens.x, max.(dens.density, 1.0e-16);
            label = plotlabel(m), color = plotstyle(m).color, linestyle = plotstyle(m).linestyle,
        )
        ip = argmax(dens.density)
        l = ip
        while l > 1 && dens.density[l - 1] > floorx
            l -= 1
        end
        r = ip
        while r < length(dens.density) && dens.density[r + 1] > floorx
            r += 1
        end
        xlo, xhi = min(xlo, dens.x[l]), max(xhi, dens.x[r])
    end
    isfinite(xlo) && xlims!(ax, xlo - 0.03 * (xhi - xlo), xhi + 0.03 * (xhi - xlo))
    # Clamp the log-y floor so the meaningless ~1e-15 KDE-tail noise is out of
    # frame; the bulk + backscatter shoulder (down to ~1e-4) stay readable.
    # dolog && ylims!(ax, 1.0e-4, nothing)
    dolog && ylims!(ax, 1.0e-4, 1.0e2)
    Legend(
        fig[0, 1], ax;
        tellwidth = false, tellheight = true, framevisible = false, horizontal = true, nbanks = 4,
    )
    rowgap!(fig.layout, 5)
    file = joinpath(figdir(case, dns, Δf), "dissipation-density.pdf")
    @info "Saving density plot to $(file)"
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Bar plot of the median pointwise SFS dissipation rate `ε_sfs = -τᵢⱼSᵢⱼ` per model,
normalized by the filtered-DNS reference median (1.0 = reference, the dashed
line). Learned `families` are aggregated over `netseeds` (whisker = ±std); the
`classical` baselines are single bars. Below 1 ⇒ under-dissipative (Clark), above
1 ⇒ over-dissipative. Same drain convention as [`compute_sfs_stats`](@ref).
"""
function plot_dissipation_bar(case, dns, Δf, families, netseeds; classical)
    agg = get_seed_statistics(case, families, dns, Δf, netseeds)
    items = [collect(classical); collect(families)]
    fig = Figure(; size = (max(520, 38 * length(items) + 150), 360))
    ax = Axis(fig[1, 1]; ylabel = "Median SFS dissipation / reference")
    vals = metric_bar!(ax, case, dns, Δf, items, agg, :diss_median)
    hlines!(ax, [1.0]; color = :black, linestyle = :dash, label = "Reference")
    ymax = maximum(v -> isnan(v.c) ? 0.0 : v.c + v.s, vals; init = 1.0)
    # Extra headroom so the top-right legend clears the value labels on the tall bars.
    ylims!(ax, 0, ymax + 0.4)
    axislegend(ax; position = :rt, framevisible = false)
    file = joinpath(figdir(case, dns, Δf), "dissipation-bar.pdf")
    @info "Saving dissipation bar plot to $(file)"
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Bar plot of the local backscatter fraction per model — the fraction of points with
`ε_sfs = -τᵢⱼSᵢⱼ < 0`. Learned `families` aggregated over `netseeds` (whisker =
±std), `classical` as single bars; the filtered-DNS reference is the dashed line
at its absolute fraction. Smagorinsky-type `τ = -2νₜS` (νₜ ≥ 0) is pinned to zero.
"""
function plot_backscatter_bar(case, dns, Δf, families, netseeds; classical)
    ref_bs = load_object(sfsstatsfile(case, dns, Δf, :ref)).diss.backscatter
    agg = get_seed_statistics(case, families, dns, Δf, netseeds)
    items = [collect(classical); collect(families)]
    fig = Figure(; size = (max(520, 38 * length(items) + 150), 360))
    ax = Axis(fig[1, 1]; ylabel = "Backscatter fraction")
    vals = metric_bar!(ax, case, dns, Δf, items, agg, :backscatter)
    hlines!(
        ax, [ref_bs];
        color = :black, linestyle = :dash, label = "Reference ($(round(ref_bs * 100; digits = 1))%)",
    )
    ymax = maximum(v -> isnan(v.c) ? 0.0 : v.c + v.s, vals; init = ref_bs)
    ylims!(ax, 0, max(ymax, ref_bs) * 1.15)
    axislegend(ax; position = :rt, framevisible = false)
    file = joinpath(figdir(case, dns, Δf), "backscatter-bar.pdf")
    @info "Saving backscatter bar plot to $(file)"
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Two-panel a-priori bar plot: relative L² error of the predicted SFS tensor (left,
lower is better) and cross-correlation with the filtered-DNS reference (right,
higher is better). Learned `families` aggregated over `netseeds` (whisker = ±std),
`classical` as single bars.
"""
function plot_apriori_bar(case, dns, Δf, families, netseeds; classical)
    agg = get_seed_statistics(case, families, dns, Δf, netseeds)
    items = [collect(classical); collect(families)]
    fig = Figure(; size = (max(900, 70 * length(items) + 120), 360))
    ax_re = Axis(fig[1, 1]; ylabel = "Relative error")
    ax_cc = Axis(fig[1, 2]; ylabel = "Cross-correlation")
    metric_bar!(ax_re, case, dns, Δf, items, agg, :relerr)
    metric_bar!(ax_cc, case, dns, Δf, items, agg, :crosscor)
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
        tellwidth = false, tellheight = true, framevisible = false,
        orientation = :horizontal, nbanks = 2,
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

    # Stable-curve accumulators: a diverging closure (Clark at coarse Δ) is still
    # drawn so the explosion is visible, but only physical rollouts set the limits.
    e_inl, eps_inl = Float64[], Float64[]

    # Full-grid DNS over the whole trajectory (gold reference).
    tdns = times .* V0
    e_dns = [s.e for s in statistics_dns] ./ V0^2
    eps_dns = [s.diss for s in statistics_dns] ./ V0^3
    lines!(ax_e, tdns, e_dns; color = :black, label = labels.dns)
    lines!(ax_eps, tdns, eps_dns; color = :black, label = labels.dns)
    append!(e_inl, e_dns)
    append!(eps_inl, eps_dns)

    # Published Re=1600 reference (already nondimensional), if available.
    # Gray dotted, so it cannot be confused with the black dashed filtered-DNS.
    ref = read_tgv_reference(reffile)
    if isnothing(ref)
        @warn "Taylor-Green reference not found at $(reffile); skipping published overlay"
    else
        lines!(ax_e, ref.t, ref.E; color = :gray, linestyle = :dot, label = "Ref. Re=1600")
        lines!(ax_eps, ref.t, ref.eps; color = :gray, linestyle = :dot, label = "Ref. Re=1600")
        append!(e_inl, ref.E)
        append!(eps_inl, ref.eps)
    end

    # LES closures: effective dissipation ε_visc + ε_sfs on each rollout.
    for m in models
        b = load_object(apostfile(case, tgv, Δf, m))
        t = b.t .* V0
        e = b.ke ./ V0^2
        eps = (b.eps_visc .+ b.eps_sfs) ./ V0^3
        kw = (; label = plotlabel(m), color = plotstyle(m).color, linestyle = plotstyle(m).linestyle)
        lines!(ax_e, t, e; kw...)
        lines!(ax_eps, t, eps; kw...)
        if tgv_rollout_stable(b)
            append!(e_inl, e)
            append!(eps_inl, eps)
        end
    end

    # Clip to the stable set: a blown-up closure runs off the box instead of
    # collapsing every physical curve to a flat line (KE ≥ 0, so anchor there).
    isempty(e_inl) || ylims!(ax_e, padded_limits(e_inl; anchor = 0.0))
    isempty(eps_inl) || ylims!(ax_eps, padded_limits(eps_inl))

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
Classical-closure metric at one eval point (no seed spread). `metric ∈`:
`:diss_median` (median SFS dissipation normalized by the `:ref` median), `:relerr`
/ `:crosscor` (a-priori tensor error / cross-correlation), `:backscatter` (local
backscatter fraction), `:e_post` (time-mean a-posteriori solution error —
`missing` if the rollout diverged, see [`apost_emean`](@ref)).
"""
function classical_metric(case, dns, Δf, c, metric)
    if metric === :diss_median
        ref = load_object(sfsstatsfile(case, dns, Δf, :ref)).diss.median
        return load_object(sfsstatsfile(case, dns, Δf, c)).diss.median / ref
    elseif metric === :relerr
        return load_object(sfsstatsfile(case, dns, Δf, c)).apriori.relerr
    elseif metric === :crosscor
        return load_object(sfsstatsfile(case, dns, Δf, c)).apriori.crosscor
    elseif metric === :backscatter
        return load_object(sfsstatsfile(case, dns, Δf, c)).diss.backscatter
    else # :e_post
        return apost_emean(case, dns, Δf, c)
    end
end

"""
`(; c, s)` for `metric` at (dns, Δf): a learned *family* `(; arch, tier,
use_redelta)` returns the seed mean/std from the `agg` aggregate (see
[`get_seed_statistics`](@ref)); a classical symbol returns its single value
([`classical_metric`](@ref), `s = 0`). `c = NaN` if a family has no surviving seed.
"""
function metric_value(case, dns, Δf, m, agg, metric)
    if m isa NamedTuple
        fam = get(agg, familyname(m), nothing)   # absent if the aggregate cached a subset
        isnothing(fam) && return (; c = NaN, s = 0.0)
        v = collect(skipmissing(getproperty(fam, metric)))
        return (; c = isempty(v) ? NaN : mean(v), s = length(v) > 1 ? std(v) : 0.0)
    end
    return (; c = classical_metric(case, dns, Δf, m, metric), s = 0.0)
end

"""
Draw `metric` as one bar per item in `items` (a mix of classical symbols and
learned families) on `ax`, learned families whiskered by their seed ±std. Sets
the x-ticks to [`famlabel`](@ref)s; returns the per-item `(; c, s)` values.
"""
function metric_bar!(ax, case, dns, Δf, items, agg, metric)
    vals = [metric_value(case, dns, Δf, m, agg, metric) for m in items]
    x = collect(1:length(items))
    barplot!(ax, x, [v.c for v in vals]; color = [plotstyle(m).color for m in items])
    pos = findall(v -> v.s > 0, vals)
    isempty(pos) ||
        errorbars!(ax, x[pos], [vals[i].c for i in pos], [vals[i].s for i in pos]; whiskerwidth = 6, color = :black)
    # Value label above each bar (and its whisker). `fixeddecimals` keeps a fixed
    # two-decimal width so the labels read as a tidy column (`string(round(...))`
    # drops trailing zeros and looks ragged: 0.5 vs 0.47 vs 1.0).
    for (xi, v) in zip(x, vals)
        isfinite(v.c) || continue
        text!(ax, xi, v.c + v.s; text = fixeddecimals(v.c, 2), align = (:center, :bottom), offset = (0, 4), fontsize = 9)
    end
    ax.xticks = (x, [famlabel(m) for m in items])
    ax.xticklabelrotation = π / 6
    return vals
end

"""
The H2 deliverable: three trends against the global filter-scale Reynolds number
`Re_Δ` (per eval point = the series-mean `redelta_mean` stored in the light
[`lesmetafile`](@ref)) — median SFS dissipation ratio (→ 1 ideal, log axis), the
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
        netseeds, classical, trainpoints = nothing,
    )
    # Series-mean global Re_Δ, read from the light lesmetafile; fall back to the
    # heavy fieldsfile for artifacts predating `redelta_mean` (run
    # `scripts/backfill_lesmeta.jl` to avoid needing the heavy file off-cluster).
    function redelta_of(dns, Δf)
        f = lesmetafile(case, dns, Δf)
        return jldopen(f, "r") do file
            haskey(file, "redelta_mean") ? file["redelta_mean"] :
                mean(load(fieldsfile(case, dns, Δf), "redelta"))
        end
    end
    re = [redelta_of(dns, Δf) for (dns, Δf) in evalpoints]
    agg = [get_seed_statistics(case, families, dns, Δf, netseeds) for (dns, Δf) in evalpoints]
    order = sortperm(re)
    re, agg, evalpoints = re[order], agg[order], evalpoints[order]

    xlab = "Filter-scale Reynolds number"
    fig = Figure(; size = (1100, 320))
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

    # Diverged a-posteriori rollouts carry a `missing` e_post (their truncated mean
    # is meaningless — see [`apost_emean`](@ref)). Rather than plotting that value
    # and letting a blow-up spike (or a deceptively low early-bail mean) distort the
    # panel, pin such points with an ✗ just above the real data. Marker height from
    # the finite a-posteriori values actually plotted.
    postvals = Float64[]
    for fam in families
        fname = familyname(fam)
        for a in agg
            haskey(a, fname) || continue
            v = collect(skipmissing(a[fname].e_post))
            isempty(v) || push!(postvals, mean(v))
        end
    end
    for c in classical, (dns, Δf) in evalpoints
        e = classical_metric(case, dns, Δf, c, :e_post)
        ismissing(e) || push!(postvals, e)
    end
    divy = isempty(postvals) ? 1.0 : maximum(postvals) * 1.05
    diverged_any = false

    for c in classical
        st = plotstyle(c)
        for (ax, metric) in zip(axs, metrics)
            ys = [classical_metric(case, dns, Δf, c, metric) for (dns, Δf) in evalpoints]
            keep = findall(!ismissing, ys)
            isempty(keep) || scatterlines!(
                ax, re[keep], Float64[ys[i] for i in keep];
                label = plotlabel(c), color = st.color, marker = st.marker, linestyle = st.linestyle,
            )
            drop = findall(ismissing, ys)
            if !isempty(drop)
                diverged_any = true
                scatter!(ax, re[drop], fill(divy, length(drop)); color = st.color, marker = :star8, markersize = 13)
            end
        end
    end

    if diverged_any
        # Legend entry for the marker, and headroom so the ✗ row clears the data.
        scatter!(ax_diss, [NaN], [NaN]; color = :gray, marker = :star8, markersize = 13, label = "Diverged")
        ylims!(ax_post, nothing, divy * 1.05)
    end

    hlines!(ax_diss, [1.0]; color = :black, linestyle = :dash, label = "Reference")
    Legend(
        fig[0, :], ax_diss;
        tellwidth = false, tellheight = true, framevisible = false,
        orientation = :horizontal, nbanks = 2,
    )
    rowgap!(fig.layout, 5)
    file = "$(case.plotdir)/trend-vs-redelta.pdf"
    @info "Saving Re_Δ trend figure to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Companion to [`plot_trend_vs_redelta`](@ref): places the decaying TGV on the same
Re_Δ axis as the forced-HIT trend, so generalization to a genuinely different flow
(laminar → transition → decay, unforced) reads directly against the forced grid
the closures were tuned on. Two panels — a-priori relative SFS error (left),
median SFS dissipation / reference (right, log) — both vs the filter-scale
Reynolds number (log x, same definition/scale as the forced figure).

`evalpoints`/`families`/`trainpoints` are the forced-grid eval points / learned
families / training points, exactly as in `plot_trend_vs_redelta`: they draw the
forced trend curves faded (background context) plus the shaded training-Re_Δ
span. `tgvpoints` is a list of `(tgv, Δf)` TGV eval points; each lands as one
marker per family per Δ, **filled for `use_redelta = false`, open for
`use_redelta = true`** (a scatter point can't use linestyle to disambiguate the
on/off pair the way the forced curves do), in the family's architecture color.

The TGV marker's x-coordinate is its Re_Δ at the instant of **peak DNS
dissipation** ([`redelta_peak_of`](@ref)), not the series mean: unlike forced HIT
(statistically stationary), the decay sweeps a wide Re_Δ range, so the mean would
smear distinct flow regimes into one point. The y-values use the same
seed-aggregate reduction as the forced grid ([`get_seed_statistics`](@ref) /
[`classical_metric`](@ref)), so the two read on a common scale. Also prints each
TGV run's peak-instant turbulence state ([`report_tgv_peak_stats`](@ref)).
"""
function plot_tgv_vs_redelta(
        case, evalpoints, families, tgvpoints;
        netseeds, classical, trainpoints = nothing,
    )
    function redelta_of(dns, Δf)
        f = lesmetafile(case, dns, Δf)
        return jldopen(f, "r") do file
            haskey(file, "redelta_mean") ? file["redelta_mean"] :
                mean(load(fieldsfile(case, dns, Δf), "redelta"))
        end
    end

    # Forced grid (faded background curves), as in plot_trend_vs_redelta.
    re = [redelta_of(dns, Δf) for (dns, Δf) in evalpoints]
    agg = [get_seed_statistics(case, families, dns, Δf, netseeds) for (dns, Δf) in evalpoints]
    order = sortperm(re)
    re, agg, evalpoints = re[order], agg[order], evalpoints[order]

    # TGV peak-instant diagnostics (printed) and Re_Δ x-coordinates / y-values.
    for tgv in unique(first.(tgvpoints))
        report_tgv_peak_stats(case, tgv, [Δf for (t, Δf) in tgvpoints if t == tgv])
    end
    tgv_re = [redelta_peak_of(case, tgv, Δf) for (tgv, Δf) in tgvpoints]
    tgv_agg = [get_seed_statistics(case, families, tgv, Δf, netseeds) for (tgv, Δf) in tgvpoints]

    xlab = "Filter-scale Reynolds number"
    fig = Figure(; size = (760, 320))
    ax_re = Axis(fig[1, 1]; xscale = log10, xlabel = xlab, ylabel = "A-priori relative SFS error")
    ax_diss = Axis(
        fig[1, 2];
        xscale = log10, yscale = log10, xlabel = xlab,
        ylabel = "Median SFS dissipation / reference",
    )
    axs = (ax_re, ax_diss)
    metrics = (:relerr, :diss_median)

    # Shade the training Re_Δ span, as in the forced trend figure.
    if !isnothing(trainpoints)
        tre = [redelta_of(dns, Δf) for (dns, Δf) in trainpoints]
        for ax in axs
            vspan!(ax, minimum(tre), maximum(tre); color = (:gray, 0.12))
        end
    end

    # Faded forced-grid trend (background context — no legend entries of its own).
    fade(color) = (color, 0.35)
    for fam in families
        fname = familyname(fam)
        st = plotstyle(fam)
        for (ax, metric) in zip(axs, metrics)
            xs, ys = Float64[], Float64[]
            for (i, a) in enumerate(agg)
                haskey(a, fname) || continue
                v = collect(skipmissing(getproperty(a[fname], metric)))
                isempty(v) && continue
                push!(xs, re[i])
                push!(ys, mean(v))
            end
            isempty(xs) ||
                scatterlines!(ax, xs, ys; color = fade(st.color), marker = st.marker, linestyle = st.linestyle)
        end
    end
    for c in classical
        st = plotstyle(c)
        for (ax, metric) in zip(axs, metrics)
            ys = [classical_metric(case, dns, Δf, c, metric) for (dns, Δf) in evalpoints]
            keep = findall(!ismissing, ys)
            isempty(keep) || scatterlines!(
                ax, re[keep], Float64[ys[i] for i in keep];
                color = fade(st.color), marker = st.marker, linestyle = st.linestyle,
            )
        end
    end

    # TGV markers: filled = blind, open = +Re; one per Δ per model.
    for fam in families
        fname = familyname(fam)
        st = plotstyle(fam)
        open = fam.use_redelta
        for (ax, metric) in zip(axs, metrics)
            xs, ys, es = Float64[], Float64[], Float64[]
            for (i, a) in enumerate(tgv_agg)
                haskey(a, fname) || continue
                v = collect(skipmissing(getproperty(a[fname], metric)))
                isempty(v) && continue
                push!(xs, tgv_re[i])
                push!(ys, mean(v))
                push!(es, length(v) > 1 ? std(v) : 0.0)
            end
            isempty(xs) && continue
            scatter!(
                ax, xs, ys;
                color = open ? :white : st.color, strokecolor = st.color, strokewidth = 1.5,
                marker = st.marker, markersize = 13, label = plotlabel(fam),
            )
            isp = findall(>(0), es)
            isempty(isp) ||
                errorbars!(ax, xs[isp], ys[isp], es[isp]; whiskerwidth = 6, color = st.color)
        end
    end
    for c in classical
        st = plotstyle(c)
        for (ax, metric) in zip(axs, metrics)
            ys = [classical_metric(case, tgv, Δf, c, metric) for (tgv, Δf) in tgvpoints]
            keep = findall(!ismissing, ys)
            isempty(keep) || scatter!(
                ax, tgv_re[keep], Float64[ys[i] for i in keep];
                color = st.color, marker = st.marker, markersize = 13, label = plotlabel(c),
            )
        end
    end

    hlines!(ax_diss, [1.0]; color = :black, linestyle = :dash, label = "Reference")
    Legend(
        fig[0, :], ax_diss;
        tellwidth = false, tellheight = true, framevisible = false,
        orientation = :horizontal, nbanks = 2,
    )
    rowgap!(fig.layout, 5)
    file = "$(case.plotdir)/tgv-vs-redelta.pdf"
    @info "Saving TGV-vs-Re_Δ figure to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
The saturation / parameter-efficiency figure: a-priori relative SFS error (left)
and time-mean a-posteriori solution error (right) vs **parameter count** (log-x),
one series per architecture, at evaluation point (dns, Δf). `families` is the size
grid for each architecture (same Re_Δ setting); each `(arch, size)` is
seed-aggregated over `netseeds` (marker = mean, whisker = ±std), read **straight
from the per-model artifacts** so adding sizes later needs no aggregate rebuild.
The story (Langford–Moser optimal closure): every architecture saturates to the
same error floor, but the ones with more inductive bias reach it at far fewer
parameters. `classical` closures are horizontal reference lines. Reads
[`sfsstatsfile`](@ref) and [`apostfile`](@ref); `paramcount` sets the x-position.
"""
function plot_saturation(case, dns, Δf, families, netseeds; classical)
    archs = unique(f.arch for f in families)
    relerr(m) = isfile(sfsstatsfile(case, dns, Δf, m)) ?
        load_object(sfsstatsfile(case, dns, Δf, m)).apriori.relerr : missing
    epost(m) = isfile(apostfile(case, dns, Δf, m)) ?
        mean(load_object(apostfile(case, dns, Δf, m)).e_post) : missing

    # (params, seed-mean, seed-std) for family `fam`, or `nothing` if no seed landed.
    function point(fam, read)
        v = collect(skipmissing(read((; fam..., netseed = s)) for s in netseeds))
        isempty(v) && return nothing
        return (paramcount(case, fam), mean(v), length(v) > 1 ? std(v) : 0.0)
    end

    fig = Figure(; size = (820, 380))
    ax_re = Axis(fig[1, 1]; xscale = log10, xlabel = "Parameters", ylabel = "A-priori relative SFS error")
    ax_post = Axis(fig[1, 2]; xscale = log10, xlabel = "Parameters", ylabel = "A-posteriori solution error")

    for (ax, read, metric, withlabel) in
        ((ax_re, relerr, :relerr, true), (ax_post, epost, :e_post, false))
        for arch in archs
            style = getstyles()[arch]
            pts = sort!(
                filter(!isnothing, [point(f, read) for f in families if f.arch === arch]); by = first,
            )
            isempty(pts) && continue
            xs, ys, es = first.(pts), getindex.(pts, 2), getindex.(pts, 3)
            scatterlines!(
                ax, xs, ys;
                color = style.color, marker = style.marker, label = withlabel ? getlabels()[arch] : nothing,
            )
            pos = findall(>(0), es)
            isempty(pos) ||
                errorbars!(ax, xs[pos], ys[pos], es[pos]; whiskerwidth = 6, color = style.color)
        end
        for c in classical
            hlines!(
                ax, [classical_metric(case, dns, Δf, c, metric)];
                color = plotstyle(c).color, linestyle = :dash, label = withlabel ? plotlabel(c) : nothing,
            )
        end
    end
    Legend(
        fig[0, :], ax_re;
        tellwidth = false, tellheight = true, framevisible = false, orientation = :horizontal,
    )
    rowgap!(fig.layout, 5)
    file = "$(case.plotdir)/saturation.pdf"
    @info "Saving saturation figure to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Conditional-mean (one-point optimal closure) a-priori error vs bin resolution —
the figure behind the "the floor *is* the optimal closure" claim
(Notes/ExperimentFollowups.md item 1). One panel per eval point in `points`
(each a `(dns, Δf)` with a [`condmeanfile`](@ref)); per panel, one curve per fit
(`train` = fitted on the training pool, `self` = fitted in-sample on that
dataset) — the level hierarchy doubles as the binning-sensitivity check.
Each family in `families` contributes a seed-mean horizontal line ± std band
from its per-seed [`sfsstatsfile`](@ref)s (house style: arch color, +Re dashed).
The no-Re nets see exactly the estimator's feature space, so their floor should
sit *on* the curve; a +Re net dipping below it shows the Re_Δ input carrying
information beyond the invariant manifold. Cross-point figure → `case.plotdir`.
"""
function plot_condmean(case, points, families, netseeds)
    titles = (; test_indist = "In-distribution", test_ood = "Out-of-distribution (higher Re)")
    fig = Figure(; size = (270 + 280 * length(points), 380))
    local ax
    for (i, (dns, Δf)) in enumerate(points)
        res = load_object(condmeanfile(case, dns, Δf))
        ax = Axis(
            fig[1, i];
            xscale = log2, xlabel = "Bins per invariant",
            ylabel = i == 1 ? "A-priori relative SFS error" : "",
            title = get(titles, dns.role, string(dns.role)),
        )

        for fam in families
            vals = collect(
                skipmissing(
                    isfile(sfsstatsfile(case, dns, Δf, (; fam..., netseed = s))) ?
                        load_object(sfsstatsfile(case, dns, Δf, (; fam..., netseed = s))).apriori.relerr :
                        missing
                        for s in netseeds
                ),
            )
            isempty(vals) && continue
            m, s = mean(vals), length(vals) > 1 ? std(vals) : 0.0
            style = plotstyle(fam)   # house style: arch color, +Re dashed
            iszero(s) || hspan!(ax, m - s, m + s; color = (style.color, 0.15))
            hlines!(ax, [m]; style.color, style.linestyle, label = plotlabel(fam))
        end

        for (r, label, marker, linestyle) in (
                (res.train, "E[τ|λ] (train-pool fit)", :circle, :solid),
                (res.self, "E[τ|λ] (in-sample fit)", :utriangle, :dot),
            )
            isnothing(r) && continue
            scatterlines!(
                ax, [l.nq for l in r.levels], [l.relerr for l in r.levels];
                color = :black, marker, linestyle, label,
            )
        end
    end

    Legend(
        fig[0, :], ax;
        tellwidth = false, tellheight = true, framevisible = false,
        horizontal = true, nbanks = 2,
    )
    rowgap!(fig.layout, 5)
    file = joinpath(case.plotdir, "condmean.pdf")
    @info "Saving conditional-mean figure to $(file)"
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
    plot_vorticity_tgv(case, tgv; ntime = 7, clip_quantile = 0.99)

Full-DNS-resolution z-vorticity montage for TGV run `tgv`: a row of horizontal
`ω_z` sections at the top z-plane, read straight from the precomputed
[`tgvvorticityfile`](@ref) (no field reconstruction). Sampled at the IC, the
peak-dissipation snapshot (the transition roll-up), then evenly through the decay;
a shared zero-centered color range keeps the amplitude decay legible. The slices
are the raw DNS field (810²), so the small-scale roll-up is resolved. Δ-independent
— one per run, saved under [`dnsfigdir`](@ref).

`ω_z` is concentrated in thin vortex sheets: a handful of cores reach the peak
amplitude while the bulk of the field is near zero, so scaling the range to the
global max washes the plot out (RdBu's center is white). The shared range is
clipped to the `clip_quantile` quantile of `|ω_z|` instead — the cores saturate
to the colormap ends but the sheets stay legible.
"""
function plot_vorticity_tgv(case, tgv; ntime = 7, clip_quantile = 0.99)
    slices, times = load(tgvvorticityfile(case, tgv), "slices", "times")
    statistics_dns, V0 = load(dnsmetafile(case, tgv), "statistics_dns", "V0")
    # Dimensionless convective time t* = t·V₀/L (L = 1), matching the TGV pipeline.
    tstar = (times .- times[1]) .* V0
    nt = length(times)

    # IC, peak-dissipation snapshot (transition), then evenly through the decay.
    diss = [s.diss for s in statistics_dns]
    ipk = argmax(diss)
    inds = round.(Int, [1; ipk; range(ipk, nt; length = ntime - 1)[2:end]])
    inds = sort(unique(clamp.(inds, 1, nt)))

    # Per-slice high quantile, maxed over the displayed snapshots: one shared
    # zero-centered range, set by the strongest sheets rather than the few
    # extreme cores (see the docstring).
    amp = maximum(i -> quantile(abs.(vec(slices[i])), clip_quantile), inds)
    colorrange = (-amp, amp)

    fig = Figure(; size = (180 * length(inds) + 80, 230))
    local hm
    for (col, i) in enumerate(inds)
        ax = Axis(
            fig[1, col];
            title = "t = $(round(tstar[i]; sigdigits = 2))",
            xticksvisible = false,
            xticklabelsvisible = false,
            yticksvisible = false,
            yticklabelsvisible = false,
            aspect = DataAspect(),
        )
        hm = image!(ax, slices[i]; colormap = :RdBu, colorrange, interpolate = false)
    end
    Colorbar(fig[1, length(inds) + 1], hm; label = "Vorticity")

    rowgap!(fig.layout, 10)
    colgap!(fig.layout, 10)

    save(joinpath(dnsfigdir(case, tgv), "vorticity-tgv.png"), fig; backend = CairoMakie)

    return fig
end

"""
    animate_vorticity_tgv(case, tgv; framerate = 12, format = "mp4", clip_quantile = 0.99)

Animate the full z-vorticity slice series for TGV run `tgv`: every horizontal `ω_z`
section from [`tgvvorticityfile`](@ref), in time order, as one evolving heatmap. A
shared zero-centered color range keeps the rise through transition and the decay on
the same scale, so the animation shows the true amplitude evolution. The range is
clipped to the `clip_quantile` quantile of `|ω_z|` (maxed over frames) rather than
the global peak amplitude, so the thin vortex sheets stay legible instead of washing
out against the near-zero background (see [`plot_vorticity_tgv`](@ref)). Writes
`vorticity-tgv.<format>` (`format ∈ ("mp4", "gif", "webm")`) under
[`dnsfigdir`](@ref) and returns the path. The static counterpart is
[`plot_vorticity_tgv`](@ref).
"""
function animate_vorticity_tgv(case, tgv; framerate = 12, format = "mp4", clip_quantile = 0.99)
    slices, times = load(tgvvorticityfile(case, tgv), "slices", "times")
    V0 = load(dnsmetafile(case, tgv), "V0")
    # Dimensionless convective time t* = t·V₀/L (L = 1), matching the TGV pipeline.
    tstar = (times .- times[1]) .* V0

    amp = maximum(s -> quantile(abs.(vec(s)), clip_quantile), slices)
    colorrange = (-amp, amp)

    frame = Observable(slices[1])
    title = Observable("t = $(round(tstar[1]; sigdigits = 2))")

    fig = Figure(; size = (560, 600))
    ax = Axis(
        fig[1, 1];
        title,
        xticksvisible = false,
        xticklabelsvisible = false,
        yticksvisible = false,
        yticklabelsvisible = false,
        aspect = DataAspect(),
    )
    hm = image!(ax, frame; colormap = :RdBu, colorrange, interpolate = false)
    Colorbar(fig[1, 2], hm; label = "Vorticity")

    file = joinpath(dnsfigdir(case, tgv), "vorticity-tgv.$(format)")
    # `Makie.record` — qualified because CUDA also exports `record` (CUDA events).
    Makie.record(fig, file, eachindex(slices); framerate) do i
        frame[] = slices[i]
        title[] = "t = $(round(tstar[i]; sigdigits = 2))"
    end
    return file
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
LES energy-spectrum comparison: the time-averaged spectra *divided by the
reference spectrum*, on a linear ordinate — this is where the per-model
deviations (Smagorinsky's intermediate-wavenumber excess, the learned models'
high-wavenumber deficit) become visible (the raw log-log spectra are nearly
indistinguishable, so that panel was dropped). The under-dissipative models
(No-model, Clark) leave the frame through their high-wavenumber pile-up; the
axis is clamped so the ±10% deviations of the well-behaved closures stay readable.
"""
function plot_spectrum_les(case, dns, Δf, models)
    # Reference LES spectrum = filtered-DNS spectra averaged over the test series.
    s_ref = mean(load(lesmetafile(case, dns, Δf), "spectra_les"))
    r = spectrum_reference(case, mean_of_named_tuple_series(load(dnsmetafile(case, dns), "statistics_dns")))
    k = 2π / case.l * eachindex(s_ref)
    styles = getstyles()

    fig = Figure(; size = (520, 360))
    ax_ratio = Axis(
        fig[1, 1];
        xscale = log10, xlabel = "Wavenumber", ylabel = "Energy relative to reference",
    )
    for m in models
        m === :ref && continue
        s = mean(load_object(apostfile(case, dns, Δf, m)).spectra_les)
        lines!(
            ax_ratio, r.kscale * k, s ./ s_ref;
            label = plotlabel(m), color = plotstyle(m).color, linestyle = plotstyle(m).linestyle,
        )
    end
    hlines!(ax_ratio, [1.0]; color = styles.ref.color, linestyle = styles.ref.linestyle, label = "Reference")
    ylims!(ax_ratio, 0, 2)
    Legend(
        fig[0, :], ax_ratio;
        tellwidth = false, tellheight = true, framevisible = false,
        orientation = :horizontal, nbanks = 2,
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
evaluation point (dns, Δf) to `<plotdir>/<filename>`. One row per learned *family*
`(; arch, tier, use_redelta)` in `families` — summarized as mean ± std over
`netseeds` (from the cached [`get_seed_statistics`](@ref) aggregate), the seed
sweep collapsed — plus one row per `classical` symbol (single values). A `Tier`
column carries the capacity tier (`--` for classical).

Columns: a-priori SFS tensor error, a-priori cross-correlation (skipped with
`include_crosscor = false` — it ranks the models the same as the closure error),
time-mean a-posteriori solution error, mean a-priori equivariance error (skipped
with `include_equi = false`; the machine-zero rows drop their seed spread, since
its ± std is meaningless there, while the non-equivariant MLPs keep theirs),
median pointwise SFS dissipation
normalized by the reference, and the local backscatter fraction. Cells whose
artifact is missing print `--`.

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
        case, dns, Δf, families, netseeds;
        classical, include_equi = true, include_crosscor = true, include_tier = true,
        filename = "errors.tex",
    )
    refstats = load_object(sfsstatsfile(case, dns, Δf, :ref))
    refmed = refstats.diss.median
    seed = get_seed_statistics(case, families, dns, Δf, netseeds)
    items = [collect(classical); collect(families)]

    # Numeric columns, in order: closure (relerr), optional cross-corr, solution
    # (e_post), optional equivariance, median diss., backscatter.
    ncol = 4 + include_crosscor + include_equi

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
        (isnothing(x) || ismissing(x)) && return nothing
        isnan(x) && return "--"
        return (; c = float(x), s = nothing)
    end

    # Drop the seed spread only from a machine-zero equivariance cell (one that
    # renders in `\e{}` scientific notation): its ± std is noise on noise. A
    # meaningful, fixed-point equivariance error — the non-equivariant MLPs —
    # keeps its spread.
    dropspread(d) = d
    # d isa NamedTuple && !(d.c == 0 || sigdecimals(d.c) <= maxdec) ?
    # (; d.c, s = nothing) : d

    # One descriptor per cell, in column order, for a learned family or classical.
    function rowdescs(m)
        if m isa NamedTuple
            s = get(seed, familyname(m), nothing)   # absent if the aggregate cached a subset
            isnothing(s) && return Any[nothing for _ in 1:ncol]
            ds = Any[aggcell(s.relerr)]
            include_crosscor && push!(ds, aggcell(s.crosscor))
            push!(ds, aggcell(s.e_post))
            include_equi && push!(ds, dropspread(aggcell(s.equi)))
            push!(ds, aggcell(s.diss_median))
            push!(ds, aggcell(s.backscatter))
            return ds
        end
        sf = sfsstatsfile(case, dns, Δf, m)
        stats = isfile(sf) ? load_object(sf) : nothing
        ds = Any[onecell(isnothing(stats) ? nothing : stats.apriori.relerr)]
        include_crosscor &&
            push!(ds, onecell(isnothing(stats) ? nothing : stats.apriori.crosscor))
        # Divergence-aware (→ "--" if the rollout blew up), matching the learned
        # rows; a bare mean over a truncated prefix would read artificially low.
        push!(ds, onecell(apost_emean(case, dns, Δf, m)))
        include_equi && push!(ds, m === :nomo ? "N.A." : nothing)   # classical: not equivariance-tested
        push!(ds, onecell(isnothing(stats) ? nothing : stats.diss.median / refmed))
        push!(ds, onecell(isnothing(stats) ? nothing : stats.diss.backscatter))
        return ds
    end
    descs = map(rowdescs, items)

    # Per-column uniform precision: each numeric column floors at `bases[j]` and
    # widens to the most decimals any of its fixed-point cells needs to show its
    # leading significant figure (the std's, when seeded). Sub-`maxdec` cells stay
    # scientific. Result: every fixed cell in a column shares one decimal place.
    bases = Int[4]                            # closure (relerr)
    include_crosscor && push!(bases, 4)       # cross-corr
    push!(bases, 4)                           # solution (e_post)
    include_equi && push!(bases, 4)           # equivariance
    append!(bases, [3, 3])                    # median diss., backscatter
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
    tiercell(m) = m isa NamedTuple ? string(m.tier) : "--"
    cellrows = [[fmt(ds[j], coldigits[j]) for j in eachindex(bases)] for ds in descs]
    widths = [maximum(length(cr[j]) for cr in cellrows) for j in eachindex(bases)]
    twidth = include_tier ? maximum(length(tiercell(m)) for m in items) : 0
    rows = map(zip(items, cellrows)) do (m, cells)
        padded = join((rpad(cells[j], widths[j]) for j in eachindex(bases)), " & ")
        prefix = "    $(rpad(plotlabel(m), 11)) & "
        include_tier && (prefix *= "$(rpad(tiercell(m), twidth)) & ")
        rstrip(prefix * padded) * " \\\\"
    end

    header = [
        "Model",
        "            & Closure \\eqref{eq:tensor-error-prior}",
    ]
    include_crosscor && push!(header, "            & Cross-corr.")
    push!(header, "            & Solution \\eqref{eq:tensor-error-post}")
    include_tier && insert!(header, 2, "            & Tier")
    include_equi &&
        push!(header, "            & Equivariance \\eqref{eq:equi-error-prior}")
    append!(
        header, [
            "            & Median diss.",
            "            & Backscatter \\\\",
        ]
    )

    file = joinpath(case.plotdir, filename)
    open(file, "w") do io
        println(io, "% Generated by SymmetryCode write_errors_table; do not edit by hand.")
        println(io, "% Reference backscatter fraction: $(round(refstats.diss.backscatter; digits = 3)).")
        println(io, "\\begin{tabular}{l $(include_tier ? "l " : "")$(join(fill("l", ncol), " "))}")
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

"Group an integer's digits in threes with commas: `commaify(12544) == \"12,544\"`."
function commaify(n::Integer)
    s = string(abs(n))
    parts = String[]
    while length(s) > 3
        pushfirst!(parts, s[(end - 2):end])
        s = s[1:(end - 3)]
    end
    pushfirst!(parts, s)
    return (n < 0 ? "-" : "") * join(parts, ",")
end

"""
Write a paper-ready LaTeX `tabular` of the computational cost per model at
evaluation point (dns, Δf) to `<plotdir>/<filename>` — the timing story for the
parameter-matched architectures across capacity tiers. One row per learned family
in `families` (training and inference aggregated as mean ± std over `netseeds`)
plus one per `classical` symbol (0 parameters and training; inference from its own
rollout). A `Tier` column carries the capacity tier.

Columns: the closed-form [`paramcount`](@ref), the training wall-time
([`psfile`](@ref) `timing`, a sweep-global one-off), and the a-posteriori
inference wall-time ([`apostfile`](@ref) `timing` at *this* eval point). At matched
parameters within a tier the columns isolate the cost of each inductive bias
(equivariant group convolution vs the tensor basis vs the plain MLP); the two
tiers show how that cost scales, and the adjacent ±Re rows show the Re_Δ feature
adds one input channel at negligible cost. A cell whose artifact is missing prints
`--`. The `:ref` row is dropped: its rollout no longer integrates (`timing ≡ 0`).
"""
function write_timing_table(
        case, dns, Δf, families, netseeds;
        classical, filename = "timing.tex",
    )
    items = [collect(classical); collect(families)]

    # Per-family seed series (a missing artifact is skipped, not an error).
    train_times(m) = [
        load(psfile(case, (; m..., netseed = s)), "timing")
            for s in netseeds if isfile(psfile(case, (; m..., netseed = s)))
    ]
    apost(file) = isfile(file) ? [load_object(file).timing] : Float64[]
    infer_times(m) =
        m isa NamedTuple ?
        reduce(vcat, (apost(apostfile(case, dns, Δf, (; m..., netseed = s))) for s in netseeds); init = Float64[]) :
        apost(apostfile(case, dns, Δf, m))

    # Math-mode mean ± std cell at one decimal; "--" when no seed survived.
    function timecell(v)
        isempty(v) && return "--"
        c = mean(v)
        s = length(v) > 1 && std(v) > 0 ? std(v) : nothing
        return "\$$(format_fixed(c, s, 1))\$"
    end

    function rowcells(m)
        learned = m isa NamedTuple
        return (;
            label = plotlabel(m),
            tier = learned ? string(m.tier) : "--",
            params = learned ? commaify(paramcount(case, m)) : "0",
            train = learned ? timecell(train_times(m)) : "\$0\$",
            infer = timecell(infer_times(m)),
        )
    end
    rows = map(rowcells, items)

    # Column-align the source (labels/tier left, the numeric columns right).
    w(f) = maximum(length(getproperty(r, f)) for r in rows)
    wl, wt, wp, wtr, wi = w(:label), w(:tier), w(:params), w(:train), w(:infer)

    file = joinpath(case.plotdir, filename)
    open(file, "w") do io
        println(io, "% Generated by SymmetryCode write_timing_table; do not edit by hand.")
        println(io, "% Inference timing at eval point: visc=$(dns.visc), Δ=$(Δf).")
        println(io, "\\begin{tabular}{l l r r r}")
        println(io, "    \\toprule")
        println(io, "    Model")
        println(io, "                & Tier")
        println(io, "                & Parameters")
        println(io, "                & Training \$[s]\$")
        println(io, "                & Inference \$[s]\$ \\\\")
        println(io, "    \\midrule")
        for r in rows
            println(
                io,
                "    $(rpad(r.label, wl)) & $(rpad(r.tier, wt)) & " *
                    "$(lpad(r.params, wp)) & $(lpad(r.train, wtr)) & $(lpad(r.infer, wi)) \\\\",
            )
        end
        println(io, "    \\bottomrule")
        println(io, "\\end{tabular}")
    end
    @info "Saving LaTeX timing table to $(file)"
    flush(stderr)
    return file
end

"""
Write a paper-ready LaTeX `tabular` characterizing each forced-HIT DNS dataset in
`runs` (e.g. `dns_runs().all`) to `<plotdir>/<filename>` — one row per `(ν, seed)`
run. Columns: viscosity `ν`, the Taylor- and integral-scale Reynolds numbers
`Re_λ` / `Re_L`, the eddy-turnover time `L/u'`, the Kolmogorov scale `η`, the
resolution `k_max·η` (≳ 1.5 is well resolved), and the dissipation `ε`.

Statistics are **time-averaged over the data window** (mean of `statistics_dns`
from [`dnsmetafile`](@ref)). For stationary forced HIT this is the converged
characterization of the regime — more robust than the single post-warmup snapshot,
which is one noisy realization. (The plot time axis instead normalizes by the
*post-warmup* turnover `t_int` stored in `dnsmetafile`, so that "n turnovers" spans
exactly `[0, n]`; when stationary, that value sits within the window fluctuation of
the `L/u'` reported here — the per-run text tables carry both for comparison.) A
run whose `dnsmetafile` is missing is skipped with a warning; decaying TGV runs
have no stationary state and are not characterized here.
"""
function write_dns_table(case, runs; filename = "dns-stats.tex")
    label(dns) =
        dns.role === :train ? "Train" :
        dns.role === :test_indist ? "Test (in-dist)" :
        dns.role === :test_ood ? "Test (OOD)" : string(dns.role)

    # 3-significant-figure LaTeX: a plain decimal in a sensible exponent range,
    # else the paper's `\e{n}` ×10ⁿ macro (e.g. ν = 1.50\e{-4}).
    function num(x; sig = 3)
        x == 0 && return "0"
        e = floor(Int, log10(abs(x)))
        -2 ≤ e ≤ 3 && return fixeddecimals(x, max(0, sig - 1 - e))
        return "$(fixeddecimals(x / exp10(e), sig - 1))\\e{$(e)}"
    end

    headers = [
        "Dataset", "\$\\nu\$", "\$\\mathrm{Re}_\\lambda\$", "\$\\mathrm{Re}_L\$",
        "\$L/u'\$", "\$\\eta\$", "\$k_{\\max}\\eta\$", "\$\\varepsilon\$",
    ]
    rows = NamedTuple[]
    for dns in runs
        f = dnsmetafile(case, dns)
        if !isfile(f)
            @warn "Missing $(f); skipping $(label(dns)) (visc=$(dns.visc)) in the DNS table"
            continue
        end
        s = mean_of_named_tuple_series(load(f, "statistics_dns"))
        push!(
            rows, (;
                name = label(dns),
                cells = [
                    "\$$(num(dns.visc))\$", "\$$(num(s.Re_tay))\$", "\$$(num(s.Re_int))\$",
                    "\$$(num(s.t_int))\$", "\$$(num(s.l_kol))\$", "\$$(num(s.kmax_eta))\$",
                    "\$$(num(s.diss))\$",
                ],
            ),
        )
    end
    if isempty(rows)
        @warn "No DNS datasets available; DNS table not written"
        flush(stderr)
        return nothing
    end

    ncell = length(headers) - 1
    wname = maximum(length(r.name) for r in rows)
    wcol = [maximum(length(r.cells[j]) for r in rows) for j in 1:ncell]

    file = joinpath(case.plotdir, filename)
    open(file, "w") do io
        println(io, "% Generated by SymmetryCode write_dns_table; do not edit by hand.")
        println(io, "% Window-time-averaged forced-HIT DNS statistics.")
        println(io, "\\begin{tabular}{l $(join(fill("l", ncell), " "))}")
        println(io, "    \\toprule")
        println(io, "    $(headers[1])")
        for (j, h) in enumerate(headers[2:end])
            println(io, "                & $(h)$(j == ncell ? " \\\\" : "")")
        end
        println(io, "    \\midrule")
        for r in rows
            cells = join((lpad(r.cells[j], wcol[j]) for j in 1:ncell), " & ")
            println(io, "    $(rpad(r.name, wname)) & $(cells) \\\\")
        end
        println(io, "    \\bottomrule")
        println(io, "\\end{tabular}")
    end
    @info "Saving LaTeX DNS-stats table to $(file)"
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
