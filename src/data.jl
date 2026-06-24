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
- `lesmetafile` (per Δ): `spectra_les`. (Filtered-field turbulence statistics are
  *not* stored — they are meaningless for the filtered field.)

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
        jldsave_atomic(lesmetafile(case, dns, Δf); spectra_les = spectra_les[k])
    end

    @info "Finished data generation after $(round(walltime; sigdigits = 4)) s"
    flush(stderr)
    return nothing
end

"""
Generate the `(ubar, τ)` data for a decaying Taylor-Green vortex.

Same `data.jld2` schema as [`create_data`](@ref) — `inputs` (filtered DNS),
`outputs` (reference deviatoric SFS stress `τ`), `times`, `spectra_dns/les`,
`statistics_dns/les` — so the entire post-hoc pipeline runs unchanged. The
differences from the forced case: the DNS is initialized from the analytic
Taylor-Green field [`taylorgreen`](@ref) at amplitude `setup.V0` (no warm-up,
so no `dns.jld2` is read), there is **no** `maintain_shell_energy!` (the flow
decays freely), and `savetimes` spans the full `[0, tstop]` so the snapshots
cover the laminar → transitional → turbulent-decay trajectory.
"""
function create_data_tgv(setup)
    (; l, visc, D, n_dns, n_les, cfl, backend, outdir, datagen, Δ, V0) = setup
    (; nstep, tstop) = datagen

    @assert D == 3 "create_data_tgv expects a 3D setup"

    filename = joinpath(outdir, "data.jld2")
    skip_if_cached(filename; label = "Taylor-Green data") && return nothing

    @info "Creating Taylor-Green data (V0 = $(V0), Re = $(V0 / visc))"
    flush(stderr)

    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)

    c_dns = getcache(g_dns)
    c_les = getcache(g_les)

    # Analytic Taylor-Green initial condition (no warm-up, no forcing).
    u = taylorgreen(g_dns, c_dns.plan; V0)

    # Allocate arrays
    ubar = vectorfield(g_les)
    σbar1 = tensorfield(g_les)
    σbar2 = tensorfield(g_les)
    τ = tensorfield(g_les)
    inputs = fill(map(Array, ubar), 0)
    outputs = fill(map(Array, τ), 0)
    sc_dns = statscache(g_dns)
    sc_les = statscache(g_les)

    # Spectra
    stuff_dns = spectral_stuff(g_dns)
    stuff_les = spectral_stuff(g_les)
    spectra_dns = fill(zeros(0), 0)
    spectra_les = fill(zeros(0), 0)

    # Compute turbulence statistics
    statistics_dns = fill(turbulence_statistics(u, visc, g_dns, sc_dns), 0)
    statistics_les = fill(turbulence_statistics(ubar, visc, g_les, sc_les), 0)

    # Keep track of adaptive time stepping
    times = zeros(0)

    @info "Starting time stepping"
    flush(stderr)

    # Time stepping
    savetimes = range(0.0, tstop, length = nstep)
    t = savetimes[1]
    timing = time()
    for (i, tnext) in enumerate(savetimes)
        # Step until the next save point.
        # Skip the first step to capture the initial statistics.
        i == 1 || while t < tnext
            # Time step
            Δt = cfl * propose_timestep(u, g_dns, visc, c_dns)
            Δt = min(Δt, tnext - t)
            t += Δt

            # Evolve DNS (decaying, no forcing)
            wray3!(convectiondiffusion!, u, Δt, g_dns, c_dns; visc)

            # Log
            if i % 1 == 0
                e = energy(u)
                @info join(
                    [
                        "i = $i",
                        "t = $(round(t; sigdigits = 4))",
                        "Δt = $(round(Δt; sigdigits = 4))",
                        "energy = $(round(e; sigdigits = 4))",
                    ],
                    ",\t",
                )
                flush(stderr)
            end
        end

        # Compute ubar and sub-filter stress
        sfs!(; τ, σbar1, σbar2, ubar, u, c_dns, c_les, g_dns, g_les, Δ)

        # Save current (ubar,tau)-pair
        push!(inputs, map(Array, ubar))
        push!(outputs, map(Array, τ))

        # Compute spectra
        s_dns = spectrum(u, g_dns, stuff_dns)
        s_les = spectrum(ubar, g_les, stuff_les)
        push!(spectra_dns, s_dns.s)
        push!(spectra_les, s_les.s)

        # Compute turbulence statistics
        stat_dns = turbulence_statistics(u, visc, g_dns, sc_dns)
        stat_les = turbulence_statistics(ubar, visc, g_les, sc_les)
        push!(statistics_dns, stat_dns)
        push!(statistics_les, stat_les)

        # Keep track of times
        push!(times, t)
    end

    timing = time() - timing

    # Save results
    save_object_atomic(
        filename,
        (;
            inputs,
            outputs,
            times,
            spectra_dns,
            spectra_les,
            statistics_dns,
            statistics_les,
            timing,
        ),
    )

    @info "Finished Taylor-Green data generation after $(timing) seconds"
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
singleton keeps the `Conv` seeing `D` spatial dims. Arrays are converted to
`precision`. Used by every model's dataloader so all closures see the same
train/val partition and float type.
"""
function split_loaders(snaps, D, batchsize; rng = nothing, val_fraction = 0.2, precision = Float32)
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
        return (precision.(x), precision.(y))
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

function create_dataloader(setup, data; nsample, batchsize, rng = nothing)
    (; D, Δ) = setup
    g = Grid{D}(; setup.l, n = setup.n_les, setup.backend)
    G = tensorfield_nonsym(g)
    u = vectorfield(g)
    τ = scalarfield(g)
    GG = spacetensorfield_nonsym(g)
    ττ = spacetensorfield(g)
    plan = plan_rfft(GG.xx)
    T = typeof(setup.l)
    train_range = data_ranges(setup).train
    nsample_use = min(nsample, length(train_range))
    fac = get_fft_fac(g)
    snaps = map(train_range[1:nsample_use]) do j
        ucpu, τcpu = data.inputs[j], data.outputs[j]
        foreach(copyto!, u, ucpu)
        apply!(vectorgradient!, g, (G, u, g))
        for (GG, G) in zip(GG, G)
            apply!(twothirds!, g, (G, g))
            ldiv!(GG, plan, G) # Inverse RFFT
            GG .*= fac
        end
        for (ττ, τcpu) in zip(ττ, τcpu)
            copyto!(τ, τcpu)
            apply!(twothirds!, g, (τ, g))
            ldiv!(ττ, plan, τ) # Inverse RFFT
            ττ .*= fac
        end
        x = stack(GG)
        y = stack(ττ)
        A2 = sum(abs2, x; dims = D + 1) # VGT squared norm
        # The nets learn an O(1), dimensionless mapping; inference in
        # `fullchain` multiplies by the same Δ^2 * |∇u|^2 factor.
        @. x ./= (sqrt(A2) + eps(T)) # Normalize input gradient
        @. y ./= (Δ^2 * A2 + eps(T)) # Normalize output stress
        (x, y) |> cpu_device()
    end
    return split_loaders(
        snaps, D, batchsize;
        rng, setup.train_setup.val_fraction, setup.train_setup.precision,
    )
end

function create_dataloader_tbnn(setup, data; nsample, batchsize, rng)
    (; D, Δ) = setup
    g = Grid{D}(; setup.l, n = setup.n_les, setup.backend)
    nx = space_ndrange(g)
    u = vectorfield(g)
    τ = scalarfield(g)
    ττ = spacetensorfield(g)
    plan = plan_rfft(ττ.xx)
    train_range = data_ranges(setup).train
    nsample_use = min(nsample, length(train_range))
    fac = get_fft_fac(g)
    T = typeof(setup.l)
    snaps = map(train_range[1:nsample_use]) do j
        ucpu, τcpu = data.inputs[j], data.outputs[j]
        foreach(copyto!, u, ucpu)
        G = getgradient(u, g)
        for (ττ, τcpu) in zip(ττ, τcpu)
            copyto!(τ, τcpu)
            apply!(twothirds!, g, (τ, g))
            ldiv!(ττ, plan, τ) # Inverse RFFT
            ττ .*= fac
        end
        i, b, a2 = build_tensorbasis(G, g)
        y = reshape(stack(ττ), nx..., tensordim(g))
        a2 = reshape(a2, nx..., 1)
        # Normalize output stress by Δ^2 * |A|^2, exactly as create_dataloader
        # does, so TBNN regresses the same normalized target as equi/conv.
        @. y = y / (Δ^2 * a2 + eps(T))
        i = i |> cpu_device()
        b = reshape(b, nx..., :) |> cpu_device()
        x = cat(i, b; dims = D + 1)
        y = y |> cpu_device()
        x, y
    end
    return split_loaders(
        snaps, D, batchsize;
        rng, setup.train_setup.val_fraction, setup.train_setup.precision,
    )
end
