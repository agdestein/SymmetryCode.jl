# DNS warm-up, paired (ubar, τ) data generation, sub-filter stress, and
# dataloaders used by the closure-model training pipelines.

function get_forcing_constant(g, u, diss, visc)
    D = dim(g)
    foreach(u -> apply!(twothirds!, g, (u, g)), u)
    u2 = sum(getenergy, u)
    d = get_dissipation!(diss, u, visc, g)
    return 4 / 5 * d / u2
end

function forced_rhs!(du, u, grid, cache; forceval, visc)
    convectiondiffusion!(du, u, grid, cache; visc)
    if isnothing(forceval)
        # Adaptive forcing strength that maintains the total energy.
        forceval = get_forcing_constant(grid, u, cache.dissfield, visc)
    end
    for (du, u) in zip(du, u)
        # Add linear forcing to du
        @. du += forceval * u
    end
    return nothing
end

"""
DNS warm-up for one run `dns = (; visc, seed, role)`: integrate forced HIT to a
statistically stationary state and save the warmed velocity field (plus the
warm-up `times`/`statistics`) to `dnsfile(case, dns)`. Coordinate-driven — ν and
seed come from `dns`, everything else from `case`.
"""
function create_dns(case, dns; force = false)
    (; D, l, n_dns, cfl, backend, totalenergy, warmup_tstop) = case
    visc = dns.visc

    file = dnsfile(case, dns)
    skip_if_cached(file; force, label = "DNS warm-up (visc=$(visc), seed=$(dns.seed))") &&
        return nothing

    rng = Xoshiro(dns.seed)
    g = Grid{D}(; l, n = n_dns, backend)

    @info "Creating initial conditions (visc=$(visc), seed=$(dns.seed))"
    flush(stderr)
    clean()
    profile = D == 2 ? linear_profile_2D : linear_profile_3D
    u = randomfield(profile, g; rng, totalenergy)
    clean()

    shells = energy_shells(g, [1, 2], u)
    cache = getcache(g)
    sc = statscache(g)

    t = 0.0
    times = [t]
    statistics = [turbulence_statistics(u, visc, g, sc)]

    @info "Running DNS warm-up to t = $(warmup_tstop)"
    flush(stderr)
    walltime = time()
    while t < warmup_tstop
        Δt = cfl * propose_timestep(u, g, visc, cache)
        Δt = min(Δt, warmup_tstop - t)
        t += Δt
        wray3!(convectiondiffusion!, u, Δt, g, cache; visc)
        maintain_shell_energy!(u, shells)
        push!(times, t)
        s = turbulence_statistics(u, visc, g, sc)
        push!(statistics, s)
        @info join(
            [
                "t = $(round(t; sigdigits = 4))",
                "Δt = $(round(Δt; sigdigits = 4))",
                "energy = $(round(s.e; sigdigits = 4))",
            ],
            ",\t",
        )
        flush(stderr)
    end
    walltime = time() - walltime

    @info "Saving warmed DNS field to $(file)"
    flush(stderr)
    jldsave_atomic(file; u = u |> cpu_device(), times, statistics, walltime)
    return nothing
end

"""
Generate the LES-resolution `(ūbar, τ)` data for one DNS run `dns`, at **every**
filter ratio in the role's list, in a *single* time-stepping pass (the DNS field
is in memory at each save time, so filtering at all Δ is near-free). Heavy fields
and light metadata go to *separate* files so plot iteration never reloads the
fields (Notes/ReExperiment.md):

- `dnsmetafile` (once): `times`, `spectra_dns`, `statistics_dns`, `t_int` — Δ-independent.
- `fieldsfile` (per Δ): `inputs` (ūbar), `outputs` (τ), `redelta` (per-snapshot
  global Re_Δ), plus `Δ`, `Δ_factor`, `visc`.
- `lesmetafile` (per Δ): `spectra_les` + `redelta_mean` (the series-mean global
  Re_Δ — the trend figure's x-coordinate, kept here so plotting never reloads the
  heavy `fieldsfile`). (Filtered-field turbulence statistics are *not* stored —
  they are meaningless for the filtered field.)

Train runs sample a few snapshots over ~2 turnovers (a-priori diversity); test
runs a denser series over ~1 turnover (a-posteriori reference). The window length
is set from the *measured* integral turnover `t_int`.
"""
function create_data(case, dns; force = false)
    (; D, l, n_dns, n_les, cfl, backend) = case
    visc = dns.visc
    filters = dns.role === :train ? case.filters_train : case.filters_test
    sampling = dns.role === :train ? case.train_sampling : case.test_sampling

    outfiles = [dnsmetafile(case, dns); [fieldsfile(case, dns, Δf) for Δf in filters]]
    if !force && all(isfile, outfiles)
        @info "data cached (visc=$(visc), seed=$(dns.seed))"
        flush(stderr)
        return nothing
    end

    @info "Creating data (visc=$(visc), seed=$(dns.seed), role=$(dns.role), filters=$(filters))"
    flush(stderr)

    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)
    u = load(dnsfile(case, dns), "u") |> adapt(backend)

    c_dns = getcache(g_dns)
    c_les = getcache(g_les)
    sc_dns = statscache(g_dns)
    stuff_dns = spectral_stuff(g_dns)
    stuff_les = spectral_stuff(g_les)

    # LES scratch buffers, reused across filters and time.
    ubar = vectorfield(g_les)
    σbar1 = tensorfield(g_les)
    σbar2 = tensorfield(g_les)
    τ = tensorfield(g_les)

    # DNS-side metadata accumulators (Δ-independent).
    stat0 = turbulence_statistics(u, visc, g_dns, sc_dns)
    times = Float64[]
    spectra_dns = Vector{Float64}[]
    statistics_dns = typeof(stat0)[]

    # Per-filter accumulators (heavy fields + light spectra).
    inputs = [typeof(map(Array, ubar))[] for _ in filters]
    outputs = [typeof(map(Array, τ))[] for _ in filters]
    redelta = [Float64[] for _ in filters]
    spectra_les = [Vector{Float64}[] for _ in filters]

    # Sampling window from the measured turnover.
    tstop = sampling.nturnover * stat0.t_int
    savetimes = range(0.0, tstop, length = sampling.nsnap)
    @info "Sampling $(sampling.nsnap) snapshots over $(sampling.nturnover) turnover(s) = " *
        "$(round(tstop; sigdigits = 4)) time units " *
        "(post-warmup t_int = $(round(stat0.t_int; sigdigits = 4)), " *
        "Δt_save ≈ $(round(tstop / (sampling.nsnap - 1); sigdigits = 3)))"
    flush(stderr)

    shells = energy_shells(g_dns, [1, 2], u)
    walltime = time()
    t = 0.0
    for (i, tnext) in enumerate(savetimes)
        # March the DNS to the save time (skip on the first, captured at t=0).
        i == 1 || while t < tnext
            Δt = cfl * propose_timestep(u, g_dns, visc, c_dns)
            Δt = min(Δt, tnext - t)
            t += Δt
            wray3!(convectiondiffusion!, u, Δt, g_dns, c_dns; visc)
            maintain_shell_energy!(u, shells)
        end

        # DNS-side metadata at this save time.
        push!(times, t)
        push!(spectra_dns, spectrum(u, g_dns, stuff_dns).s)
        push!(statistics_dns, turbulence_statistics(u, visc, g_dns, sc_dns))

        # Filter at every Δ — the DNS field is already in memory here.
        for (k, Δf) in enumerate(filters)
            Δ = Δf * l / n_les
            sfs!(; τ, σbar1, σbar2, ubar, u, c_dns, c_les, g_dns, g_les, Δ)
            push!(inputs[k], map(Array, ubar))
            push!(outputs[k], map(Array, τ))
            push!(redelta[k], filter_reynolds(ubar, g_les, visc, Δ))
            push!(spectra_les[k], spectrum(ubar, g_les, stuff_les).s)
        end

        @info "saved snapshot $(i)/$(sampling.nsnap) at t = $(round(t; sigdigits = 4))"
        flush(stderr)
    end
    walltime = time() - walltime

    jldsave_atomic(
        dnsmetafile(case, dns);
        times, spectra_dns, statistics_dns, t_int = stat0.t_int, walltime,
    )
    for (k, Δf) in enumerate(filters)
        Δ = Δf * l / n_les
        jldsave_atomic(
            fieldsfile(case, dns, Δf);
            inputs = inputs[k], outputs = outputs[k], redelta = redelta[k],
            Δ, Δ_factor = Δf, visc,
        )
        jldsave_atomic(
            lesmetafile(case, dns, Δf);
            spectra_les = spectra_les[k], redelta_mean = mean(redelta[k]),
        )
    end

    @info "Finished data generation after $(round(walltime; sigdigits = 4)) s"
    flush(stderr)
    return nothing
end

"""
    create_slices(case, dns; force = false)

Extract 2D `z = l/2` slices of the warm-up DNS snapshot ([`dnsfile`](@ref)) for
static figures (paper, graphical abstract, blog), one self-contained
[`slicefile`](@ref) per filter ratio of the run's role. Each file stores, all as
`Matrix{Float32}` (computed in Float64, cast on write):

- `u` — DNS velocity `(; x, y, z)` at full DNS resolution (`n_dns²`).
- `omega_z` — DNS out-of-plane vorticity `ω_z` (`n_dns²`).
- `ubar` — filtered-DNS velocity `(; x, y, z)` at LES resolution (`n_les²`).
- `omegabar_z` — `ω̄_z` from `ūbar` (`n_les²`).
- `grad` — resolved velocity gradient `∇ū`, non-symmetric `(; xx, yx, zx, …, zz)`.
- `tau` — deviatoric sub-filter stress `τ`, symmetric `(; xx, yy, zz, xy, yz, zx)`.

plus scalar metadata (`Δ`, `Δ_factor`, `visc`, `n_dns`, `n_les`, `l`, `z`). The plane
is `z ≈ l/2`: the cell-centered DNS and LES grids share no exact z (see the
`kslice` note below), so `u` is cut at the DNS row nearest the LES mid-plane. The
DNS-side slices are Δ-independent, so they are computed once and copied into every
filter's file (keeps each artifact standalone; the duplication is ~10 MiB/file,
negligible at this scale). Cache-guarded like [`create_data`](@ref).
"""
function create_slices(case, dns; force = false)
    (; D, l, n_dns, n_les, backend) = case
    @assert D == 3 "create_slices extracts a z = l/2 plane; assumes 3D"
    visc = dns.visc
    filters = dns.role === :train ? case.filters_train : case.filters_test

    outfiles = [slicefile(case, dns, Δf) for Δf in filters]
    if !force && all(isfile, outfiles)
        @info "slices cached (visc=$(visc), seed=$(dns.seed))"
        flush(stderr)
        return nothing
    end

    @info "Extracting slices (visc=$(visc), seed=$(dns.seed), role=$(dns.role))"
    flush(stderr)

    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)
    u = load(dnsfile(case, dns), "u") |> adapt(backend)

    # `sfs!` (via `nonlinearity!`) only ever reads `σ`, `v`, `vi_vj`, `plan` from
    # each cache; the `ustart`/`du` time-integration buffers `getcache` would also
    # allocate are two dead vector fields — ~26 GB on the 810³ DNS grid, enough to
    # OOM the H100. Build a trimmed cache with just what `sfs!` touches, and fold
    # every slice-extraction scratch buffer into these same arrays (see below).
    sfscache(g) = let vi_vj = spacescalarfield(g)
        (; σ = tensorfield(g), v = spacevectorfield(g), vi_vj, plan = plan_rfft(vi_vj))
    end
    c_dns = sfscache(g_dns)
    c_les = sfscache(g_les)

    # Which z-plane to cut. The grids are cell-centered (point k sits at
    # `(k - ½)·l/n`), so the DNS (n_dns) and LES (n_les) sample planes never
    # coincide exactly — `(i-½)/n_dns = (j-½)/n_les` reduces to `even = odd`. We
    # therefore anchor on the LES mid-plane (`z ≈ l/2`) and cut the DNS at the row
    # whose cell centre is nearest *that* height, so `u` and `ū` overlap as closely
    # as the resolutions allow (residual Δz ≈ 4e-4·l, vs ≈3e-3·l for a boundary
    # plane). `kslice(n)` returns that index for a field with `n` points per side.
    zmid = (n_les ÷ 2 + 1 - 1 // 2) / n_les          # height of the LES mid-plane
    kslice(n) = clamp(round(Int, zmid * n + 1 // 2), 1, n)

    # `to_phys!` (a c2r inverse FFT) *destroys its spectral input*, so any field
    # still needed afterwards is inverted through a spectral scratch buffer. The
    # chosen plane is gathered to the host and cast to Float32 (a copy, so the
    # phys-space buffer is free for the next component). We reuse the cache's own
    # `σ.xx` (spectral) and `vi_vj` (physical) as that scratch: on the DNS grid
    # both are free until the first `sfs!` below overwrites them, and on the LES
    # grid `sfs!` writes its nonlinearity into `σbar2` (not `c_les.σ`), so `σ.xx`
    # stays free throughout. Fields not reused (∇ū, τ) are inverted in place.
    midslice(field) = Float32.(Array(@view field[:, :, kslice(size(field, 3))]))
    slice_of!(spec, phys, plan, g) = (to_phys!(phys, spec, plan, g); midslice(phys))
    # Invert a copy so `spec_src` survives (used again downstream).
    keep_slice(spec_src, tmp, phys, plan, g) =
        (copyto!(tmp, spec_src); slice_of!(tmp, phys, plan, g))

    # --- DNS-resolution slices (Δ-independent; computed once) ---
    # `u` is reused by every sfs! below, so convert its components via a copy.
    spec_dns, phys_dns = c_dns.σ.xx, c_dns.vi_vj
    u_slice = map(ui -> keep_slice(ui, spec_dns, phys_dns, c_dns.plan, g_dns), u)
    apply!(vorticity_z!, g_dns, (spec_dns, u, g_dns))   # ω_z → spectral scratch
    omega_z = slice_of!(spec_dns, phys_dns, c_dns.plan, g_dns)

    # --- LES-resolution scratch (reused across filters) ---
    ubar = vectorfield(g_les)
    σbar1 = tensorfield(g_les)
    σbar2 = tensorfield(g_les)
    τ = tensorfield(g_les)
    G = tensorfield_nonsym(g_les)
    spec_les, phys_les = c_les.σ.xx, c_les.vi_vj

    for Δf in filters
        Δ = Δf * l / n_les
        sfs!(; τ, σbar1, σbar2, ubar, u, c_dns, c_les, g_dns, g_les, Δ)

        # ū is reused by the vorticity and gradient below — convert via a copy.
        ubar_slice = map(ui -> keep_slice(ui, spec_les, phys_les, c_les.plan, g_les), ubar)

        apply!(vorticity_z!, g_les, (spec_les, ubar, g_les))   # ω̄_z → spectral scratch
        omegabar_z = slice_of!(spec_les, phys_les, c_les.plan, g_les)

        # ∇ū: dealias each spectral component (matches the dataloader) then invert
        # in place (G is rebuilt from ū next iteration).
        apply!(vectorgradient!, g_les, (G, ubar, g_les))
        grad = map(G) do gi
            apply!(twothirds!, g_les, (gi, g_les))
            slice_of!(gi, phys_les, c_les.plan, g_les)
        end

        # τ from sfs! is already 2/3-truncated and trace-free; invert in place
        # (rebuilt next iteration).
        tau = map(ti -> slice_of!(ti, phys_les, c_les.plan, g_les), τ)

        jldsave_atomic(
            slicefile(case, dns, Δf);
            u = u_slice, omega_z, ubar = ubar_slice, omegabar_z, grad, tau,
            Δ, Δ_factor = Δf, visc, n_dns, n_les, l, z = zmid * l,
        )
        @info "saved slice (visc=$(visc), seed=$(dns.seed), Δ=$(Δf))"
        flush(stderr)
    end
    return nothing
end

"""
Time index of peak DNS dissipation for a TGV run — the reference instant shared by
every peak-instant diagnostic ([`report_tgv_redelta`](@ref),
[`compute_redelta_peak`](@ref), `S.plot_tgv_vs_redelta`). Unlike forced HIT
(statistically stationary, so the series mean is representative), the decaying TGV
sweeps a wide range of regimes, so a single physically meaningful instant — the
canonical dissipation-rate peak — is used instead of an average.
"""
redelta_peak_index(statistics_dns) = argmax([s.diss for s in statistics_dns])

"""
Post-run diagnostic for a TGV DNS pass: print (1) the resolution at peak
dissipation (`kmax_η` is most strained there) and the turbulence state there
(Re_λ, Re_L), and (2) the per-filter global Re_Δ trajectory — startup → at-peak →
final — against the forced-HIT trained band ([`hit_redelta_band`](@ref)). The
trajectory rises from a near-laminar start, so the placement check is on the
value at the peak-dissipation instant ([`redelta_peak_index`](@ref)), not the
trajectory's own (possibly later) maximum; whether it lands inside the trained
band (vs the held-out band, vs full extrapolation) is the
interpolation-generalization read for the TGV capstone. Printing only; skips the
band line if the HIT sweep is not yet on disk.
"""
function report_tgv_redelta(case, tgv, times, redelta, filters, statistics_dns)
    kme = [s.kmax_eta for s in statistics_dns]
    ipk = redelta_peak_index(statistics_dns)
    s = statistics_dns[ipk]
    @info "TGV resolution: kmax_η at peak diss (t=$(round(times[ipk]; sigdigits = 3)))" *
        " = $(round(kme[ipk]; digits = 2)), min over run = $(round(minimum(kme); digits = 2))" *
        " (target ≳ 1.5)"
    @info "TGV peak-dissipation turbulence state: Re_λ=$(round(s.Re_tay; digits = 1)), " *
        "Re_L=$(round(s.Re_int; digits = 1))"

    band = hit_redelta_band(case)
    haveband = !isempty(band.train)
    tlo, thi = haveband ? extrema(band.train) : (NaN, NaN)
    flo, fhi = haveband ? extrema(band.full) : (NaN, NaN)
    if haveband
        @info "HIT Re_Δ band (per-(ν,Δ) snapshot means): " *
            "train [$(round(Int, tlo))–$(round(Int, thi))], " *
            "full [$(round(Int, flo))–$(round(Int, fhi))]"
    else
        @info "HIT Re_Δ band unavailable (run the forced-HIT sweep first); " *
            "printing TGV trajectory only."
    end

    @info "TGV global Re_Δ trajectory (per filter):"
    for (k, Δf) in enumerate(filters)
        tr = redelta[k]
        rpk = tr[ipk]
        verdict = !haveband ? "" :
            rpk > fhi ? " — peak ABOVE full band (extrapolation)" :
            rpk > thi ? " — peak in held-out band (train < peak ≤ full)" :
            rpk ≥ tlo ? " — peak within trained band" :
            " — peak BELOW trained band (low-Re extrapolation)"
        @info "  Δf=$(Δf): start=$(round(tr[1]; digits = 1)) " *
            "at-dns-peak=$(round(Int, rpk)) traj-max=$(round(Int, maximum(tr))) " *
            "final=$(round(tr[end]; digits = 1))$(verdict)"
    end
    flush(stderr)
    return nothing
end

"""
Re_Δ at the peak-DNS-dissipation instant for one TGV eval point `(tgv, Δf)`,
computed from the heavy `fieldsfile` — call this only where it lives (the
cluster; see `scripts/backfill_redelta_peak.jl` for backfilling artifacts that
predate this). `create_data_tgv` calls this reduction inline and caches the
result as `redelta_peak` in the light `lesmetafile`, so off-cluster code should
prefer [`redelta_peak_of`](@ref).
"""
function compute_redelta_peak(case, tgv, Δf)
    statistics_dns = load(dnsmetafile(case, tgv), "statistics_dns")
    ipk = redelta_peak_index(statistics_dns)
    redelta = load(fieldsfile(case, tgv, Δf), "redelta")
    return redelta[ipk]
end

"""
Off-cluster-safe accessor for the TGV peak-instant Re_Δ: reads the cached
`redelta_peak` from the light `lesmetafile` if present, else falls back to
[`compute_redelta_peak`](@ref) (which needs the heavy `fieldsfile`). Mirrors the
`redelta_mean` fallback pattern used for the forced trend's x-coordinate.
"""
function redelta_peak_of(case, tgv, Δf)
    f = lesmetafile(case, tgv, Δf)
    return jldopen(f, "r") do file
        haskey(file, "redelta_peak") ? file["redelta_peak"] : compute_redelta_peak(case, tgv, Δf)
    end
end

"""
Light, off-cluster-safe peak-instant report for a TGV run: prints the turbulence
statistics (Re_λ, Re_L, k_max·η) at the instant of peak DNS dissipation, plus the
per-filter Re_Δ at that same instant ([`redelta_peak_of`](@ref)) — the numbers the
paper's TGV peak-Re_Δ claims are read off. Reads only the light `dnsmetafile`/
`lesmetafile`, unlike [`report_tgv_redelta`](@ref), which needs the heavy
`fieldsfile` and so only runs where that lives.
"""
function report_tgv_peak_stats(case, tgv, filters)
    times, statistics_dns = load(dnsmetafile(case, tgv), "times", "statistics_dns")
    ipk = redelta_peak_index(statistics_dns)
    s = statistics_dns[ipk]
    @info "TGV peak-dissipation instant (t=$(round(times[ipk]; sigdigits = 3))): " *
        "Re_λ=$(round(s.Re_tay; digits = 1)), Re_L=$(round(s.Re_int; digits = 1)), " *
        "k_max·η=$(round(s.kmax_eta; digits = 2))"
    for Δf in filters
        rpk = redelta_peak_of(case, tgv, Δf)
        @info "  Δf=$(Δf): Re_Δ(peak) = $(round(Int, rpk))"
    end
    flush(stderr)
    return nothing
end

"""
Generate the decaying Taylor-Green `(ūbar, τ)` data for one TGV run `tgv`
(`(; visc, seed, role=:tgv, Re_target)`), at every test filter ratio, in a single
DNS pass — the same two-file schema as [`create_data`](@ref) so the whole post-hoc
pipeline applies unchanged on the `(tgv, Δf)` eval points.

The DNS starts from the analytic [`taylorgreen`](@ref) field at amplitude
`V0 = Re_target·visc` (no warm-up, so no `dnsfile`), decays freely (no
`maintain_shell_energy!`), and the save times span `tconv` convective times
`t_c = L/V0` so the snapshots cover the laminar → transition → decay trajectory.
`dnsmetafile` additionally stores `V0`/`Re_target` for the dissipation benchmark
([`plot_dissipation_tgv`](@ref)). `lesmetafile` additionally stores `redelta_peak`
— the Re_Δ at the peak-DNS-dissipation instant ([`redelta_peak_index`](@ref)),
the TGV's representative point on the Re_Δ axis (the decay sweeps too wide a
range for the forced-HIT series mean to be meaningful here).
"""
function create_data_tgv(case, tgv; force = false)
    (; D, l, n_dns, n_les, cfl, backend) = case
    @assert D == 3 "create_data_tgv expects a 3D case"
    @assert tgv.role === :tgv "create_data_tgv expects a :tgv run"
    visc = tgv.visc
    V0 = tgv.Re_target * visc
    filters = case.filters_test
    (; nsnap, tconv) = case.tgv_sampling
    tstop = tconv / V0

    outfiles = [
        dnsmetafile(case, tgv); tgvvorticityfile(case, tgv);
        [fieldsfile(case, tgv, Δf) for Δf in filters]
    ]
    if !force && all(isfile, outfiles)
        @info "TGV data cached (Re=$(tgv.Re_target))"
        flush(stderr)
        return nothing
    end

    @info "Creating Taylor-Green data (Re=$(tgv.Re_target), V0=$(V0), filters=$(filters))"
    flush(stderr)

    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)
    c_dns = getcache(g_dns)
    c_les = getcache(g_les)
    sc_dns = statscache(g_dns)
    stuff_dns = spectral_stuff(g_dns)
    stuff_les = spectral_stuff(g_les)

    # Analytic Taylor-Green initial condition (no warm-up, no forcing).
    u = taylorgreen(g_dns, c_dns.plan; V0)

    # LES scratch, reused across filters and time.
    ubar = vectorfield(g_les)
    σbar1 = tensorfield(g_les)
    σbar2 = tensorfield(g_les)
    τ = tensorfield(g_les)

    # DNS-side metadata accumulators (Δ-independent).
    stat0 = turbulence_statistics(u, visc, g_dns, sc_dns)
    times = Float64[]
    spectra_dns = Vector{Float64}[]
    statistics_dns = typeof(stat0)[]

    # z-vorticity slice series — a horizontal x-y plane at max z, full DNS
    # resolution — for the transition roll-up visualization. ω_z = ∂x u_y − ∂y u_x
    # is formed in spectral space, transformed to a physical slab, and the top
    # plane is stored as Float32 to keep the file light. Both working fields alias
    # time-stepping scratch (`σ.xx` spectral, `vi_vj` physical): the slice is taken
    # before `sfs!` and the next march, which overwrite them anyway — so no extra
    # 810³ allocation (memory is tight at this DNS size).
    ω_spec = c_dns.σ.xx
    ω_phys = c_dns.vi_vj
    vort_slices = Matrix{Float32}[]

    # Per-filter accumulators (heavy fields + light spectra).
    inputs = [typeof(map(Array, ubar))[] for _ in filters]
    outputs = [typeof(map(Array, τ))[] for _ in filters]
    redelta = [Float64[] for _ in filters]
    spectra_les = [Vector{Float64}[] for _ in filters]

    # Span the full transition→decay trajectory (no measured turnover here).
    savetimes = range(0.0, tstop, length = nsnap)
    @info "Sampling $(nsnap) snapshots over $(tconv) convective times = " *
        "$(round(tstop; sigdigits = 4)) time units (t_c = L/V0 = $(round(1 / V0; sigdigits = 4)))"
    flush(stderr)
    walltime = time()
    t = 0.0
    for (i, tnext) in enumerate(savetimes)
        # March the decaying DNS to the save time (skip on the first, t = 0).
        i == 1 || while t < tnext
            Δt = cfl * propose_timestep(u, g_dns, visc, c_dns)
            Δt = min(Δt, tnext - t)
            t += Δt
            wray3!(convectiondiffusion!, u, Δt, g_dns, c_dns; visc)   # decaying, no forcing
        end

        push!(times, t)
        push!(spectra_dns, spectrum(u, g_dns, stuff_dns).s)
        push!(statistics_dns, turbulence_statistics(u, visc, g_dns, sc_dns))

        # Horizontal z-vorticity slice at the top z-plane (full DNS resolution).
        apply!(vorticity_z!, g_dns, (ω_spec, u, g_dns))
        to_phys!(ω_phys, ω_spec, c_dns.plan, g_dns)
        push!(vort_slices, Float32.(Array(@view ω_phys[:, :, end])))

        for (k, Δf) in enumerate(filters)
            Δ = Δf * l / n_les
            sfs!(; τ, σbar1, σbar2, ubar, u, c_dns, c_les, g_dns, g_les, Δ)
            push!(inputs[k], map(Array, ubar))
            push!(outputs[k], map(Array, τ))
            push!(redelta[k], filter_reynolds(ubar, g_les, visc, Δ))
            push!(spectra_les[k], spectrum(ubar, g_les, stuff_les).s)
        end

        @info "saved TGV snapshot $(i)/$(nsnap) at t = $(round(t; sigdigits = 4))"
        flush(stderr)
    end
    walltime = time() - walltime

    report_tgv_redelta(case, tgv, times, redelta, filters, statistics_dns)
    ipk = redelta_peak_index(statistics_dns)

    jldsave_atomic(
        dnsmetafile(case, tgv);
        times, spectra_dns, statistics_dns,
        t_int = stat0.t_int, V0, Re_target = tgv.Re_target, walltime,
    )
    jldsave_atomic(tgvvorticityfile(case, tgv); slices = vort_slices, times)
    for (k, Δf) in enumerate(filters)
        Δ = Δf * l / n_les
        jldsave_atomic(
            fieldsfile(case, tgv, Δf);
            inputs = inputs[k], outputs = outputs[k], redelta = redelta[k],
            Δ, Δ_factor = Δf, visc,
        )
        jldsave_atomic(
            lesmetafile(case, tgv, Δf);
            spectra_les = spectra_les[k], redelta_mean = mean(redelta[k]),
            redelta_peak = redelta[k][ipk],
        )
    end

    @info "Finished TGV data generation after $(round(walltime; sigdigits = 4)) s"
    flush(stderr)
    return nothing
end

"""
Compute the deviatoric sub-filter stress for one DNS velocity snapshot.

This is the data-generation definition of the target:
`τ = overline(u_i u_j) - ubar_i ubar_j`, optionally Gaussian-filtered, then
made trace-free to match the learned closures' output space.
"""
function sfs(u, g_dns, g_les, Δ)
    c_dns = getcache(g_dns)
    c_les = getcache(g_les)
    ubar = vectorfield(g_les)
    σbar1 = tensorfield(g_les)
    σbar2 = tensorfield(g_les)
    τ = tensorfield(g_les)
    sfs!(; τ, σbar1, σbar2, ubar, u, c_dns, c_les, g_dns, g_les, Δ)
    return τ
end

function sfs!(; τ, σbar1, σbar2, ubar, u, c_dns, c_les, g_dns, g_les, Δ)
    nonlinearity!(c_dns.σ, c_dns.vi_vj, c_dns.v, u, c_dns.plan, g_dns)
    for (ubar, u) in zip(ubar, u)
        apply!(cutoff!, g_les, (ubar, u))
        isnothing(Δ) || apply!(gaussianfilter!, g_les, (ubar, Δ, g_les))
        apply!(twothirds!, g_les, (ubar, g_les))
    end
    for (σbar1, σ) in zip(σbar1, c_dns.σ)
        apply!(cutoff!, g_les, (σbar1, σ))
        isnothing(Δ) || apply!(gaussianfilter!, g_les, (σbar1, Δ, g_les))
    end
    nonlinearity!(σbar2, c_les.vi_vj, c_les.v, ubar, c_les.plan, g_les)
    foreach(i -> (τ[i] .= σbar1[i] .- σbar2[i]), 1:tensordim(g_dns))
    foreach(τ -> apply!(twothirds!, g_les, (τ, g_les)), τ)
    # The isotropic part of SFS stress can be absorbed into pressure in the
    # incompressible equations, so training and inference use deviatoric τ.
    return make_tracefree!(τ, g_les)
end

"""
Split per-snapshot `(x, y)` pairs into a training and a held-out validation
`DataLoader`. The split is time-based: the last `val_fraction` of the
(time-ordered) snapshots become the validation set. The last spatial axis is
folded into the batch dimension (the pointwise 1x1 models use no neighbor
information) so each snapshot is cheap and 2D gets as many samples as 3D; a
singleton keeps the `Conv` seeing `D` spatial dims. The caller stores snapshots
already in the training precision (so nothing here is ever the Float64 solver
type — the TBNN dataset, ~5× the channels of the others, would otherwise blow up
host RAM). Used by every model's dataloader so all closures see the same
train/val partition and float type.
"""
function split_loaders(snaps, D, batchsize; rng = nothing, val_fraction = 0.2)
    n = length(snaps)
    @assert n >= 2 "need at least 2 snapshots to form a train/val split"
    nval = clamp(round(Int, val_fraction * n), 1, n - 1)
    function build(s)
        x = stack(first, s)
        y = stack(last, s)
        cax = D + 1                                    # channel axis
        perm = (ntuple(identity, D - 1)..., cax, D, D + 2)
        x = permutedims(x, perm)
        y = permutedims(y, perm)
        x = reshape(x, size(x)[1:(D - 1)]..., 1, size(x, D), :)
        y = reshape(y, size(y)[1:(D - 1)]..., 1, size(y, D), :)
        return (x, y)
    end
    xt, yt = build(snaps[1:(n - nval)])
    xv, yv = build(snaps[(n - nval + 1):n])
    trainloader = if isnothing(rng)
        DataLoader((xt, yt); batchsize, shuffle = true, partial = false)
    else
        DataLoader((xt, yt); batchsize, shuffle = true, partial = false, rng)
    end
    valloader = DataLoader((xv, yv); batchsize, shuffle = false, partial = true)
    return trainloader, valloader
end

"""
Append a constant standardized `log Re_Δ` channel along the channel axis `D+1`
(the group-invariant Re_Δ input feature); a no-op unless `use_redelta`.
`redelta_norm` is `(; μ, σ)` from the trainpool. Works for both the dataloader
layout `(spatial…, channel)` and the inference layout `(spatial…, channel, batch)`
— the new channel matches every non-channel axis.
"""
function append_redelta(x, re, use_redelta, redelta_norm, D)
    use_redelta || return x
    z = (log(re) - redelta_norm.μ) / redelta_norm.σ
    chansize = (size(x)[1:D]..., 1, size(x)[(D + 2):end]...)
    chan = fill!(similar(x, chansize), z)
    return cat(x, chan; dims = D + 1)
end

"""
(train, val) `DataLoader`s for the G-CNN / MLP closures from a `trainpool` — a
list of `(dns, Δ_factor)` coordinates. Each dataset's heavy fields are read from
its `fieldsfile`, the normalized pair `(∇ū/|∇ū|, τ/(Δ²|∇ū|²))` is formed using
*that dataset's* Δ, and all snapshots are concatenated. With `use_redelta`, each
snapshot's standardized `log Re_Δ` is appended as an extra input channel. The val
set is a holdout of the pool (monitoring only — see [`train`](@ref)).
"""
function create_dataloader(
        case, trainpool;
        batchsize, rng = nothing, use_redelta = false, redelta_norm = nothing,
    )
    (; D) = case
    g = Grid{D}(; case.l, n = case.n_les, case.backend)
    G = tensorfield_nonsym(g)
    u = vectorfield(g)
    τ = scalarfield(g)
    GG = spacetensorfield_nonsym(g)
    ττ = spacetensorfield(g)
    plan = plan_rfft(GG.xx)
    T = typeof(case.l)
    fac = get_fft_fac(g)
    snaps = mapreduce(vcat, trainpool) do (dns, Δf)
        inputs, outputs, Δ, redelta =
            load(fieldsfile(case, dns, Δf), "inputs", "outputs", "Δ", "redelta")
        map(eachindex(inputs)) do j
            foreach(copyto!, u, inputs[j])
            apply!(vectorgradient!, g, (G, u, g))
            for (GG, G) in zip(GG, G)
                apply!(twothirds!, g, (G, g))
                ldiv!(GG, plan, G) # Inverse RFFT
                GG .*= fac
            end
            for (ττ, τcpu) in zip(ττ, outputs[j])
                copyto!(τ, τcpu)
                apply!(twothirds!, g, (τ, g))
                ldiv!(ττ, plan, τ) # Inverse RFFT
                ττ .*= fac
            end
            x = stack(GG)
            y = stack(ττ)
            A2 = sum(abs2, x; dims = D + 1) # VGT squared norm
            @. x ./= (sqrt(A2) + eps(T)) # Normalize input gradient
            @. y ./= (Δ^2 * A2 + eps(T)) # Normalize output stress (this dataset's Δ)
            x = append_redelta(x, redelta[j], use_redelta, redelta_norm, D)
            # Downcast to the training precision *before* storing on the host, so
            # the concatenated dataset never holds the Float64 solver type.
            P = case.schedule.precision
            (P.(x), P.(y)) |> cpu_device()
        end
    end
    return split_loaders(snaps, D, batchsize; rng, val_fraction = case.schedule.val_fraction)
end

"""
(train, val) `DataLoader`s for the TBNN from a `trainpool`. Same per-dataset Δ
normalization as [`create_dataloader`](@ref); the input packs the gradient
invariants (with the standardized `log Re_Δ` appended as an extra invariant when
`use_redelta`) followed by the (O(1)) basis tensors. The loss splits the two
blocks from the *end* (the basis is the last `tensordim·nbasis` channels), so the
extra invariant needs no change there.
"""
function create_dataloader_tbnn(
        case, trainpool;
        batchsize, rng = nothing, use_redelta = false, redelta_norm = nothing,
    )
    (; D) = case
    g = Grid{D}(; case.l, n = case.n_les, case.backend)
    nx = space_ndrange(g)
    u = vectorfield(g)
    τ = scalarfield(g)
    ττ = spacetensorfield(g)
    plan = plan_rfft(ττ.xx)
    fac = get_fft_fac(g)
    T = typeof(case.l)
    snaps = mapreduce(vcat, trainpool) do (dns, Δf)
        inputs, outputs, Δ, redelta =
            load(fieldsfile(case, dns, Δf), "inputs", "outputs", "Δ", "redelta")
        map(eachindex(inputs)) do j
            foreach(copyto!, u, inputs[j])
            G = getgradient(u, g)
            for (ττ, τcpu) in zip(ττ, outputs[j])
                copyto!(τ, τcpu)
                apply!(twothirds!, g, (τ, g))
                ldiv!(ττ, plan, τ) # Inverse RFFT
                ττ .*= fac
            end
            i, b, a2 = build_tensorbasis(G, g)
            y = reshape(stack(ττ), nx..., tensordim(g))
            a2 = reshape(a2, nx..., 1)
            @. y = y / (Δ^2 * a2 + eps(T))
            i = i |> cpu_device()
            i = append_redelta(i, redelta[j], use_redelta, redelta_norm, D) # extra invariant
            b = reshape(b, nx..., :) |> cpu_device()
            x = cat(i, b; dims = D + 1)
            # Downcast to the training precision before storing — the TBNN dataset
            # carries the (O(1)) basis tensors (~5× the channels of the others), so
            # keeping it Float64 on the host is what OOMs the node.
            P = case.schedule.precision
            (P.(x), P.(y)) |> cpu_device()
        end
    end
    return split_loaders(
        snaps, D, batchsize;
        rng, val_fraction = case.schedule.val_fraction,
    )
end
