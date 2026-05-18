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

function create_dns(setup)
    (; outdir, l, visc, D, n_dns, cfl, backend, warmup) = setup
    (; totalenergy, tstop, seed) = warmup
    rng = Xoshiro(seed)

    g = Grid{D}(; l, n = n_dns, backend)

    @info "Creating initial conditions"
    flush(stderr)
    clean()
    # profile = D == 2 ? peak_profile : linear_profile_3D
    # args = D == 2 ? (; kpeak = 10) : (;)
    profile = D == 2 ? linear_profile_2D : linear_profile_3D
    args = (;)
    u = randomfield(profile, g; rng, totalenergy, args...)
    clean()

    # # Load previous DNS state
    # u = load("$(outdir)/dns.jld2", "u") |> adapt(backend)
    # times, energies, dissipations =
    #     load("$(outdir)/dns.jld2", "times", "energies", "dissipations")
    # t = times[end]

    # # Band stuff
    # b = getband(g, 3)
    # band = (;
    #     inds = b.inds |> adapt(backend),
    #     energyinds = vcat(b.inds, b.conjinds) |> adapt(backend),
    # )
    # eband_ref = sum(u -> sum(abs2, view(u, band.energyinds)), u) / 2

    shells = energy_shells(g, [1, 2], u)

    # Allocate arrays
    cache = getcache(g)
    sc = statscache(g)

    t = 0.0
    times = [t]
    statistics = [turbulence_statistics(u, visc, g, sc)]

    @info "Running DNS simulation"
    flush(stderr)
    k = 0
    walltime = time()
    while t < tstop
        Δt = cfl * propose_timestep(u, g, visc, cache)
        Δt = min(Δt, tstop - t)
        t += Δt
        k += 1

        # Step
        wray3!(convectiondiffusion!, u, Δt, g, cache; visc)
        # wray3!(forced_rhs!, u, Δt, g, cache; forceval = nothing, visc)

        # # Maintain energy
        # eband = sum(u -> sum(abs2, view(u, band.energyinds)), u) / 2
        # foreach(u -> (view(u, band.inds) .*= sqrt(eband_ref / eband)), u)

        maintain_shell_energy!(u, shells)

        if k % 1 == 0
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
    end
    walltime = time() - walltime

    # Save results
    file = joinpath(outdir, "dns.jld2")
    @info "Saving final DNS snapshot to $(file)"
    flush(stderr)
    jldsave(
        file;
        u = u |> cpu_device(),
        times,
        statistics,
        walltime,
    )

    return nothing
end

function create_data(setup)
    (; l, visc, D, n_dns, n_les, cfl, backend, outdir, datagen, Δ) = setup
    (; nstep, tstop) = datagen

    @info "Creating data"
    flush(stderr)

    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)

    # Load DNS state from warm-up simulation
    u = load(joinpath(outdir, "dns.jld2"), "u") |> adapt(backend)

    # Allocate arrays
    ubar = vectorfield(g_les)
    σbar1 = tensorfield(g_les)
    σbar2 = tensorfield(g_les)
    τ = tensorfield(g_les)
    inputs = fill(map(Array, ubar), 0)
    outputs = fill(map(Array, τ), 0)
    c_dns = getcache(g_dns)
    c_les = getcache(g_les)
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

    shells = energy_shells(g_dns, [1, 2], u)

    # # Compute force factor from DNS
    # forceval = get_forcing_constant(g_dns, u, dissfield_dns, visc)

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

            # Evolve DNS
            wray3!(convectiondiffusion!, u, Δt, g_dns, c_dns; visc)
            # wray3!(forced_rhs!, u, Δt, g_dns, c_dns; forceval, visc)

            maintain_shell_energy!(u, shells)

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
    filename = joinpath(setup.outdir, "data.jld2")
    save_object(
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

    @info "Finished data generation after $(timing) seconds"
    flush(stderr)

    return nothing
end

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

function create_dataloader(setup, data; nsample, batchsize)
    (; D, Δ) = setup
    g = Grid{D}(; setup.l, n = setup.n_les, setup.backend)
    G = tensorfield_nonsym(g)
    u = vectorfield(g)
    τ = scalarfield(g)
    GG = spacetensorfield_nonsym(g)
    ττ = spacetensorfield(g)
    plan = plan_rfft(GG.xx)
    T = typeof(setup.l)
    nsample_use = min(nsample, length(data.inputs))
    fac = get_fft_fac(g)
    snaps = map(1:nsample_use) do j
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
        @. x ./= (sqrt(A2) + eps(T)) # Normalize input gradient
        @. y ./= (Δ^2 * A2 + eps(T)) # Normalize output stress
        (x, y) |> cpu_device()
    end
    return split_loaders(
        snaps, D, batchsize;
        setup.train_setup.val_fraction, setup.train_setup.precision,
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
    nsample_use = min(nsample, length(data.inputs))
    fac = get_fft_fac(g)
    T = typeof(setup.l)
    snaps = map(1:nsample_use) do j
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
