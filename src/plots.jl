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
    equi = "G-Conv",
    conv = "MLP",
)

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
Plot validation-loss curves for every learned closure in `mkeys` whose
`ps-<key>.jld2` is on disk. Keys not in `(:tbnn, :equi, :conv)` and keys
without a persisted artifact are silently skipped, so the same call works
regardless of which subset of learned models was actually trained.
"""
function plot_training(setup, mkeys)
    labels = getlabels()
    learned = filter(in([:tbnn, :equi, :conv]), collect(mkeys))
    curves = NamedTuple(
        k => load_object(joinpath(setup.outdir, "ps-$(k).jld2")).losses_valid
            for k in learned if isfile(joinpath(setup.outdir, "ps-$(k).jld2"))
    )
    isempty(curves) && return nothing

    fig = Figure(; size = (400, 340))
    ax = Axis(fig[1, 1]; xlabel = "Iteration", ylabel = "Loss")
    for k in keys(curves)
        lines!(ax, curves[k]; label = labels[k])
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
    eps = 0.1
    # ylims!(ax, -eps, 1 + eps)
    rowgap!(fig.layout, 5)
    save("$(setup.plotdir)/training.pdf", fig; backend = CairoMakie)
    return fig
end

function plot_error_post(setup, les_stat)
    data = joinpath(setup.outdir, "data.jld2") |> load_object
    fig = Figure(; size = (400, 360))
    ax = Axis(
        fig[1, 1];
        xlabel = "Time",
        ylabel = "Relative error",
    )
    # e_post is eval-window-aligned; pull the matching times.
    t = data.times[data_ranges(setup).eval]
    t .-= t[1] # Mark start time as zero
    labels = getlabels()
    for k in keys(les_stat)
        e = les_stat[k].e_post
        ntime = length(e)
        lines!(ax, t[1:ntime], e; label = labels[k])
    end
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
    save("$(setup.plotdir)/error-post.pdf", fig; backend = CairoMakie)
    return fig
end

function plot_densities(setup, mkeys; dolog)
    (; outdir, plotdir, name) = setup
    yscale = dolog ? log10 : identity

    # t_kol = mean(x -> x.t_kol, data.statistics_les)

    fig = Figure(; size = (900, 300))
    labels = getlabels()

    # Axes
    ax = (;
        xx = Axis(
            fig[1, 1];
            # xlabel = L"\tau_{1 1}", xlabelsize = 20,
            xlabel = "SFS (xx)",
            ylabel = "Density",
            yscale,
        ),
        xy = Axis(
            fig[1, 2];
            # xlabel = L"\tau_{1 2}", xlabelsize = 20,
            xlabel = "SFS (xy)",
            yscale,
            yticksvisible = false,
            yticklabelsvisible = false,
        ),
        diss = Axis(
            fig[1, 3];
            # xlabel = L"-\tau_{i j} S_{i j}", xlabelsize = 20,
            xlabel = "SFS dissipation rate",
            yscale,
            yticksvisible = false,
            yticklabelsvisible = false,
        ),
    )

    for mkey in mkeys
        stats = load_object("$(outdir)/sfs_stats_$(mkey).jld2")
        for fkey in [:xx, :xy, :diss]
            dens = stats.kde[fkey]
            isempty(dens.x) && continue   # :nomo has degenerate (all-zero) samples
            lines!(ax[fkey], dens.x, max.(dens.density, 1.0e-16); label = labels[mkey])
        end
    end

    if contains(name, "laptop")
        xlims!(ax.xx, -0.1, 0.3)
        ylims!(ax.xx, 2.0e-2, 3.0e2)
    elseif contains(name, "turbulator")
        xlims!(ax.xx, -0.2, 0.2)
        ylims!(ax.xx, 2.0e-4, 3.0e2)
    elseif contains(name, "snellius")
        xlims!(ax.xx, -0.1, 0.15)
        ylims!(ax.xx, 4.0e-4, 4.0e2)
    end

    # XY-component
    if contains(name, "laptop")
        xlims!(ax.xy, -0.1, 0.1)
        ylims!(ax.xy, 1.0e-1, 5.0e2)
    elseif contains(name, "turbulator")
        xlims!(ax.xy, -0.15, 0.15)
        ylims!(ax.xy, 1.0e-3, 3.0e2)
    elseif contains(name, "snellius")
        xlims!(ax.xy, -0.12, 0.12)
        ylims!(ax.xy, 4.0e-4, 4.0e2)
    end

    # SFS dissipation rate ε_sfs = -τᵢⱼSᵢⱼ: positive = drain (the bulk of
    # the distribution for any sane closure); negative = local backscatter.
    if contains(name, "laptop")
        xlims!(ax.diss, -0.3, 0.3)
        ylims!(ax.diss, 1.0e-1, 1.0e2)
    elseif contains(name, "turbulator")
        xlims!(ax.diss, -0.15, 0.5)
        ylims!(ax.diss, 1.0e-3, 1.0e2)
    elseif contains(name, "snellius")
        xlims!(ax.diss, -0.17, 0.6)
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

"""
Bar plot of median pointwise SFS dissipation rate `ε_sfs = -τᵢⱼSᵢⱼ` per
model, normalized by the filtered-DNS reference median. Bars above 1.0 are
over-dissipative (smag/dynsmag); bars below 1.0 are under-dissipative
(clark). The reference is shown as a horizontal dashed line at 1.0.

Drain convention (positive = drain); same as [`compute_sfs_stats`](@ref).
`keys` must include `:ref` (used as the normalization baseline).
"""
function plot_dissipation_bar(setup, keys)
    (; outdir, plotdir) = setup
    labels = getlabels()
    for (momkey, momlabel) in [(:median, "Median"), (:mean, "Mean")]
        moments = NamedTuple(
            k => load_object("$(outdir)/sfs_stats_$(k).jld2").diss[momkey] for k in keys
        )
        @assert haskey(moments, :ref) "plot_dissipation_bar requires :ref in keys"
        ref_med = moments.ref
        plot_keys = filter(!=(:ref), collect(keys))
        normalized = [moments[k] / ref_med for k in plot_keys]

        fig = Figure(; size = (450, 340))
        ax = Axis(
            fig[1, 1];
            xticks = (1:length(plot_keys), [labels[k] for k in plot_keys]),
            # ylabel = L"\mathrm{median}(\tau_{ij} S_{ij}) / \mathrm{median}_{\mathrm{ref}}",
            ylabel = "$(momlabel) dissipation",
            xticklabelrotation = π / 6,
        )
        barplot!(ax, 1:length(plot_keys), normalized; bar_labels = :y)
        hlines!(ax, [1.0]; color = :red, linestyle = :dash, label = "Reference")

        # Reference
        # axislegend(ax; position = :rt, framevisible = false)
        Legend(
            fig[0, :], ax;
            tellwidth = false, tellheight = true, framevisible = false,
            horizontal = true,
        )

        # Adjust upper limit to make space for bar label
        ylims!(ax, 0, maximum(normalized) + 0.15)
        rowgap!(fig.layout, 5)

        file = "$(plotdir)/dissipation-$(momkey)-bar.pdf"
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

`keys` must include `:ref`.
"""
function plot_backscatter_bar(setup, keys)
    (; outdir, plotdir) = setup
    labels = getlabels()
    backscatter = NamedTuple(
        k => load_object("$(outdir)/sfs_stats_$(k).jld2").diss.backscatter for k in keys
    )
    @assert haskey(backscatter, :ref) "plot_backscatter_bar requires :ref in keys"
    ref_bs = backscatter.ref
    plot_keys = filter(!=(:ref), collect(keys))
    bars = [backscatter[k] for k in plot_keys]

    fig = Figure(; size = (520, 340))
    ax = Axis(
        fig[1, 1];
        xticks = (1:length(plot_keys), [labels[k] for k in plot_keys]),
        # ylabel = L"\mathrm{fraction\ with\ } \tau_{ij} S_{ij} > 0",
        ylabel = "Backscatter fraction",
        xticklabelrotation = π / 6,
    )
    barplot!(ax, 1:length(plot_keys), bars; bar_labels = :y)
    hlines!(
        ax, [ref_bs];
        color = :red, linestyle = :dash,
        label = "Reference ($(round(ref_bs * 100; digits = 1))%)",
    )
    ylims!(ax, 0, max(maximum(bars; init = 0.0), ref_bs) * 1.15)
    axislegend(ax; position = :rt, framevisible = false)

    file = "$(plotdir)/backscatter-bar.pdf"
    @info "Saving backscatter bar plot to $(file)"
    save(file, fig; backend = CairoMakie)
    return fig
end

"""
Two-panel bar plot of a-priori predictive metrics per model:
relative L² error of the predicted SFS tensor (left, lower is better) and
cross-correlation with the filtered-DNS reference (right, higher is better).

`keys` should not include `:ref` (self-comparison is trivial).
"""
function plot_apriori_bar(setup, keys)
    (; outdir, plotdir) = setup
    labels = getlabels()
    stats = NamedTuple(
        k => load_object("$(outdir)/sfs_stats_$(k).jld2").apriori for k in keys
    )
    plot_keys = collect(keys)
    relerrs = [stats[k].relerr for k in plot_keys]
    crosscors = [stats[k].crosscor for k in plot_keys]

    fig = Figure(; size = (820, 340))
    xtks = (1:length(plot_keys), [labels[k] for k in plot_keys])
    ax_re = Axis(
        fig[1, 1];
        xticks = xtks,
        # ylabel = "Relative L² error",
        ylabel = "Relative error",
        xticklabelrotation = π / 6,
    )
    barplot!(ax_re, 1:length(plot_keys), relerrs; bar_labels = :y)
    ax_cc = Axis(
        fig[1, 2];
        xticks = xtks,
        ylabel = "Cross-correlation",
        xticklabelrotation = π / 6,
    )
    barplot!(ax_cc, 1:length(plot_keys), crosscors; bar_labels = :y)

    # Adjust upper limit to make space for bar label
    ylims!(ax_re, 0.0, 1.1)
    ylims!(ax_cc, 0.0, 1.1)

    file = "$(plotdir)/apriori-bar.pdf"
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

`keys` should include `:ref` and any closure keys. Reads `budget_<k>.jld2`.
"""
function plot_budget(setup, keys)
    (; outdir, plotdir) = setup
    labels = getlabels()
    fig = Figure(; size = (820, 360))
    ax_ke = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Kinetic energy")
    ax_eps = Axis(fig[1, 2]; xlabel = "Time", ylabel = "SFS dissipation rate")
    for k in keys
        b = load_object("$(outdir)/budget_$(k).jld2")
        lines!(ax_ke, b.t .- b.t[1], b.ke; label = labels[k])
        lines!(ax_eps, b.t .- b.t[1], b.eps_sfs; label = labels[k])
    end
    Legend(
        fig[0, :], ax_ke;
        tellwidth = false, tellheight = true, framevisible = false,
        horizontal = true, nbanks = 4,
    )
    rowgap!(fig.layout, 5)
    file = "$(plotdir)/budget.pdf"
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
    data = joinpath(outdir, "data.jld2") |> load_object

    fig = Figure(; size = (820, 360))
    ax_e = Axis(fig[1, 1]; xlabel = L"t^* = t V_0 / L", ylabel = L"E_k / V_0^2")
    ax_eps = Axis(fig[1, 2]; xlabel = L"t^* = t V_0 / L", ylabel = L"\varepsilon\, L / V_0^3")

    # Full-grid DNS over the whole trajectory (gold reference).
    tdns = data.times .* V0
    lines!(ax_e, tdns, [s.e for s in data.statistics_dns] ./ V0^2; color = :black, label = labels.dns)
    lines!(ax_eps, tdns, [s.diss for s in data.statistics_dns] ./ V0^3; color = :black, label = labels.dns)

    # Published Re=1600 reference (already nondimensional), if available.
    ref = read_tgv_reference(reffile)
    if isnothing(ref)
        @warn "Taylor-Green reference not found at $(reffile); skipping published overlay"
    else
        lines!(ax_e, ref.t, ref.E; color = :gray, linestyle = :dash, label = "Ref. Re=1600")
        lines!(ax_eps, ref.t, ref.eps; color = :gray, linestyle = :dash, label = "Ref. Re=1600")
    end

    # LES closures: effective dissipation ε_visc + ε_sfs on each rollout.
    for k in keys
        b = load_object("$(outdir)/budget_$(k).jld2")
        t = b.t .* V0
        lines!(ax_e, t, b.ke ./ V0^2; label = labels[k])
        lines!(ax_eps, t, (b.eps_visc .+ b.eps_sfs) ./ V0^3; label = labels[k])
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
"""
function plot_dissipation_vs_re(
        setups, keys;
        train_anchor = nothing,
        plotdir = last(setups).plotdir,
        Re_key = :Re_int,
    )
    labels = getlabels()
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
    ax_d = Axis(fig[1, 1]; xlabel = Re_labels[Re_key], ylabel = "Median SFS dissipation / reference")
    ax_e = Axis(fig[1, 2]; xlabel = Re_labels[Re_key], ylabel = "A-priori relative SFS error")

    re_all = [peak_re_int(s.outdir) for s in setups]
    order = sortperm(re_all)
    sorted = setups[order]
    res = re_all[order]
    for k in plot_keys
        m = [load_metrics(s.outdir, k) for s in sorted]
        keep = .!isnothing.(m)
        any(keep) || continue
        scatterlines!(ax_d, res[keep], [x.ratio for x in m[keep]]; label = labels[k])
        scatterlines!(ax_e, res[keep], [x.relerr for x in m[keep]]; label = labels[k])
    end

    # Forced-training anchor: the regime the learned closures were fit to, drawn
    # as star markers (one shared legend entry) colored to match each model — the
    # point the per-model trends should extrapolate toward. All stars sit at the
    # forced case's eval-window mean Re_int (its stationary integral Reynolds
    # number), the like-for-like counterpart of the TGV peak Re_int.
    if !isnothing(train_anchor)
        anchor = filter(!isnothing, [load_metrics(train_anchor.outdir, k) for k in plot_keys])
        if !isempty(anchor)
            d = load_object("$(train_anchor.outdir)/data.jld2")
            anchor_re = mean(s -> s[Re_key], d.statistics_dns[data_ranges(train_anchor).eval])
            d = nothing
            GC.gc()
            re = fill(anchor_re, length(anchor))
            label = "Training (Re=$(round(Int, anchor_re)))"
            color = Makie.wong_colors()[1:length(anchor)]
            scatter!(ax_d, re, [x.ratio for x in anchor]; marker = :star5, markersize = 14, color, label)
            scatter!(ax_e, re, [x.relerr for x in anchor]; marker = :star5, markersize = 14, color)
        end
    end

    hlines!(ax_d, [1.0]; color = :red, linestyle = :dash, label = "Reference")
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
convention as [`plot_budget`](@ref) and [`plot_densities`](@ref). Reads
`transfer_<k>.jld2`.

The axes match the energy-spectrum plots: wavenumber rescaled by the
small-scale length (Kolmogorov η in 3D, Kraichnan η_K in 2D); dissipation
rescaled by the eval-window-mean total energy dissipation rate `ε`, so
`Σ_k ε_sfs(k)/ε ≈ 1` for a balanced closure.
"""
function plot_spectral_transfer(setup, keys)
    (; outdir, plotdir) = setup
    labels = getlabels()
    data = joinpath(outdir, "data.jld2") |> load_object
    eval_range = data_ranges(setup).eval
    stats = mean_of_named_tuple_series(data.statistics_dns[eval_range])
    r = spectrum_reference(setup, stats)
    ε = stats.diss

    fig = Figure(; size = (520, 360))
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        xlabel = "Wavenumber",
        ylabel = "SFS dissipation rate",
    )
    for k in keys
        k == :nomo && continue
        t = load_object("$(outdir)/transfer_$(k).jld2")
        lines!(ax, r.kscale * t.k, t.eps_sfs ./ ε; label = labels[k])
    end
    hlines!(ax, [0.0]; color = :gray, linestyle = :dash)
    Legend(
        fig[0, 1], ax;
        tellwidth = false, tellheight = true, framevisible = false,
        horizontal = true, nbanks = 3,
    )
    rowgap!(fig.layout, 5)
    file = "$(plotdir)/spectral-transfer.pdf"
    @info "Saving spectral transfer plot to $(file)"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    return fig
end

function plot_velocities(setup, comp, modelkeys)
    (; D, l, n_les, backend) = setup

    upostfiles = get_upostfiles(setup)
    data = joinpath(setup.outdir, "data.jld2") |> load_object

    # Both reference (filtered DNS) and model rollouts live on the eval window.
    eval_range = data_ranges(setup).eval
    times_eval = data.times[eval_range]
    inputs_eval = data.inputs[eval_range]

    fig = Figure(; size = (800, 470))
    g = Grid{D}(; l, n = n_les, backend)
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
            upost = load_object(upostfiles[key])
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

            ui = useries[t][comp] |> adapt(backend)
            apply!(twothirds!, g, (ui, g))
            to_phys!(ui_space, ui, plan, g)
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
    gaussian_blur(A, σ)

Separable Gaussian blur of matrix `A` with standard deviation `σ` measured in
**grid cells**, using clamped (replicate) boundaries so the decaying tails of a
density are not pulled toward zero at the grid edge. Returns `A` unchanged when
`σ` is `nothing` or non-positive.

Blurring a precomputed kernel density estimate is equivalent to having used a
wider KDE bandwidth (Gaussian ∗ Gaussian = Gaussian), and is used here only to
calm the noisy outer isocontours in `plot_qr`. Keep `σ` small so the sharp
near-origin / Vieillefosse-tail structure is not washed out.
"""
function gaussian_blur(A::AbstractMatrix, σ)
    (σ === nothing || σ <= 0) && return A
    rad = ceil(Int, 3σ)
    k = [exp(-(t^2) / (2σ^2)) for t in (-rad):rad]
    k ./= sum(k)
    n1, n2 = size(A)
    B = similar(A) # blur along dim 1
    for j in 1:n2, i in 1:n1
        acc = zero(eltype(A))
        for (t, w) in zip((-rad):rad, k)
            acc += w * A[clamp(i + t, 1, n1), j]
        end
        B[i, j] = acc
    end
    C = similar(A) # blur along dim 2
    for j in 1:n2, i in 1:n1
        acc = zero(eltype(A))
        for (t, w) in zip((-rad):rad, k)
            acc += w * B[i, clamp(j + t, 1, n2)]
        end
        C[i, j] = acc
    end
    return C
end

function plot_qr(setup, modelkeys; smooth_σ = nothing)
    (; name) = setup
    qr = map(key -> key => load_object("$(setup.outdir)/qr_$(key).jld2"), modelkeys)
    qr = NamedTuple(qr)

    fig = Figure(; size = (650, 440))
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
        bard = colorvec[lescolor],
        tbnn = colorvec[lescolor],
        conv = colorvec[lescolor],
        equi = colorvec[lescolor],
    )

    plotkeys = filter(!=(:ref), modelkeys)

    # Reference density is the same in every panel — blur it once.
    refdens = max.(gaussian_blur(qr.ref.density, smooth_σ), 1.0e-20)

    for (k, key) in plotkeys |> enumerate
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
            # xlabel = L"r", xlabelsize = 20,
            # ylabel = L"q", ylabelsize = 20,
            xlabel = "r",
            ylabel = "q",
            title,
        )
        if contains(name, "turbulator")
            ran = 1.0e-3, 1.0e1
            ncat = 6
        elseif contains(name, "snellius")
            # ran = 1.0e-4, 1.0e1
            # ncat = 7
            ran = 1.0e-3, 1.0e1
            ncat = 7
        end
        # key => extrema(qr.density) |> display
        isref = key == :dns || key == :ref
        isref || contour!(
            ax,
            qr.ref.x,
            qr.ref.y,
            refdens;
            levels = logrange(ran..., ncat),
            color = colors.ref,
        )
        contour!(
            ax,
            qr[key].x,
            qr[key].y,
            max.(gaussian_blur(qr[key].density, smooth_σ), 1.0e-20);
            levels = logrange(ran..., ncat),
            color = colors[key],
        )
        qtest = range(-10, 0, 200)
        rtest1 = @. 2 / 3 / sqrt(3) * (-qtest)^(3 / 2)
        rtest2 = @. -2 / 3 / sqrt(3) * (-qtest)^(3 / 2)
        lines!(ax, rtest1, qtest; color = colors.line)
        lines!(ax, rtest2, qtest; color = colors.line)
        if contains(name, "turbulator")
            xlims!(ax, -1.5, 1.5)
            ylims!(ax, -3, 3)
        elseif contains(name, "snellius")
            # xlims!(ax, -2.0, 2.0)
            # ylims!(ax, -3, 4)
            xlims!(ax, -1.2, 1.4)
            ylims!(ax, -2.5, 3)
        end
    end
    save("$(setup.plotdir)/qr.pdf", fig; backend = CairoMakie)
    return fig
end

function plot_equivariance_errors(setup, errs; tag::Symbol)
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
        dynsmag = Cycled(2),
        clar = Cycled(3),
        bard = Cycled(3),
        tbnn = Cycled(4),
        equi = Cycled(5),
        conv = Cycled(6),
    )
    labels = getlabels()
    markers = (;
        nomo = :utriangle,
        smag = :circle,
        dynsmag = :circle,
        clar = :rect,
        bard = :star5,
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
    save("$(setup.plotdir)/equi-errors-$(tag).pdf", fig; backend = CairoMakie)
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

function plot_spectrum_les(setup, les_stat)
    (; D) = setup

    data = joinpath(setup.outdir, "data.jld2") |> load_object
    eval_range = data_ranges(setup).eval

    # Average the reference spectrum over the same eval window the LES
    # rollouts are evaluated on, so the two curves are comparable.
    s_ref = mean(data.spectra_les[eval_range])
    s_les = map(stat -> mean(stat.s), les_stat)
    r = spectrum_reference(setup, mean_of_named_tuple_series(data.statistics_dns[eval_range]))
    k = 2π / setup.l * eachindex(s_ref)
    labels = getlabels()

    fig = Figure(; size = (400, 360))
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
    lines!(ax, r.kscale * k, r.escale * s_ref; label = "Reference")
    for key in keys(s_les)
        lines!(ax, r.kscale * k, r.escale * s_les[key]; label = labels[key])
    end
    # lines!(ax, r.k_ref, r.E_ref; label = r.label)
    # vlines!(ax, r.kdiss; color = (:gray, 0.5), linestyle = :dash)
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
