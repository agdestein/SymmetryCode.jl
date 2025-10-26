module Spectral

using Adapt
using CairoMakie
using CUDA
using FFTW
using JLD2
using KernelAbstractions
using KernelDensity
using LaTeXStrings
using LinearAlgebra
using Lux
using Makie
using MLUtils
using Optimisers
using Random
using Seneca
using StaticArrays
using Statistics
using ..SymmetryCode
using Zygote

getlabels() = (;
    dns = "DNS",
    ref = "Filtered DNS",
    nomo = "No-model",
    smag = "Smagorinsky",
    vers = "Verstappen",
    clar = "Clark",
    tbnn = "TBNN",
    equi = "G-Conv",
    conv = "Conv",
)

@inline function cutoff_index(nbar, n, i, is1)
    imax = div(nbar, 2) + is1
    isneg = i > imax
    ifelse(isneg, n - nbar + i, i) # Negative wavenumbers count backwards
end
@inline cutoff_index(nbar, n, I::CartesianIndex{2}) = CartesianIndex((
    cutoff_index(nbar, n, I.I[1], true),
    cutoff_index(nbar, n, I.I[2], false),
))
@inline cutoff_index(nbar, n, I::CartesianIndex{3}) = CartesianIndex((
    cutoff_index(nbar, n, I.I[1], true),
    cutoff_index(nbar, n, I.I[2], false),
    cutoff_index(nbar, n, I.I[3], false),
))

export cutoff!
@kernel function cutoff!(ubar, u)
    nbar = size(ubar, 2)
    n = size(u, 2)
    I = @index(Global, Cartesian)
    J = cutoff_index(nbar, n, I)
    ubar[I] = u[J]
end

export inverse_cutoff!
@kernel function inverse_cutoff!(u, ubar)
    nbar = size(ubar, 2)
    n = size(u, 2)
    I = @index(Global, Cartesian)
    J = cutoff_index(nbar, n, I)
    u[J] = ubar[I]
end

export gaussianfilter!
@kernel function gaussianfilter!(u, Δ, kstart, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = wavenumber_full(g, I)
    k2 = kx^2 + ky^2
    kstart2 = (π / g.l * 2 * kstart)^2
    # Don't filter the forced wavenumbers (where k < kstart)
    # Otherwise, the fixed-energy forcing will not commute with the filter.
    # Since we only force the low wavenumbers, the Gaussian is close to 1 anyway.
    w = ifelse(k2 < kstart2, one(Δ), exp(-Δ^2 * k2 / 24))
    u[I] *= w
end
@kernel function gaussianfilter!(u, Δ, kstart, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumber_full(g, I)
    k2 = kx^2 + ky^2 + kz^2
    kstart2 = (π / g.l * 2 * kstart)^2
    # Don't filter the forced wavenumbers (where k < kstart)
    # Otherwise, the fixed-energy forcing will not commute with the filter.
    # Since we only force the low wavenumbers, the Gaussian is close to 1 anyway.
    w = ifelse(k2 < kstart2, one(Δ), exp(-Δ^2 * k2 / 24))
    u[I] *= w
end

"Verification with DNS-aided LES."
function dns_aid()
    visc = 4e-4
    t = 0.0
    cfl = 0.85
    tstop = 1e-1
    D = 3
    g = Grid{D}(; l = 1.0, n = 16)
    gbar = Grid{D}(; l = 1.0, n = 8)
    u = randomfield(g; kpeak = 5)
    foreach(randn!, u)
    apply!(project!, g, (u, g))
    ubar = vectorfield(gbar)
    foreach(i -> apply!(cutoff!, gbar, (ubar[i], u[i])), 1:D)
    v = map(copy, ubar)
    fσ = tensorfield(gbar)
    σf = tensorfield(gbar)
    c = getcache(g)
    cbar = getcache(gbar)
    i = 0
    while t < tstop
        i += 1
        Δt = cfl * propose_timestep(u, c, visc, g)
        Δt = min(Δt, tstop - t)
        t += Δt
        @info "t = $t, Δt = $Δt"
        # DNS
        stress!(c.σ, c.vi_vj, c.v, u, c.plan, visc, g)
        apply!(tensordivergence!, g, (c.du, c.σ, g))
        # LES
        foreach(i -> apply!(cutoff!, gbar, (ubar[i], u[i])), 1:D)
        stress!(σf, cbar.vi_vj, cbar.v, ubar, cbar.plan, visc, gbar)
        stress!(cbar.σ, cbar.vi_vj, cbar.v, v, cbar.plan, visc, gbar)
        foreach(i -> apply!(cutoff!, gbar, (fσ[i], c.σ[i])), 1:tensordim(g))
        foreach(i -> (cbar.σ[i] .+= fσ[i] .- σf[i]), 1:tensordim(g))
        apply!(tensordivergence!, gbar, (cbar.du, cbar.σ, gbar))
        # Step
        for i = 1:dim(g)
            axpy!(Δt, c.du[i], u[i])
            axpy!(Δt, cbar.du[i], v[i])
        end
        apply!(project!, g, (u, g))
        apply!(project!, gbar, (v, gbar))
    end
    foreach(i -> apply!(cutoff!, gbar, (ubar[i], u[i])), 1:D)
    sum(i -> sum(abs2, v[i] - ubar[i]) / sum(abs2, ubar[i]), 1:D)
end

export create_dns
function create_dns(setup; tstop, cfl, rng)
    (; outdir, l, visc, D, n_dns, backend, force) = setup
    g = Grid{D}(; l, n = n_dns, backend)

    @info "Creating initial conditions"
    flush(stderr)
    u = randomfield(
        g;
        rng,
        # totalenergy = 5,
        kpeak = 5,
    )
    # u = load("$(outdir)/dns.jld2", "u") |> adapt(backend)
    GC.gc();
    CUDA.reclaim()
    cache = getcache(g)

    # Forcing stuff
    b = getband(g, force[1])

    # Indices for computing energy (with conjugate indices)
    energyinds = vcat(b...) |> adapt(backend)

    # Components to be forced (without conjugate indices)
    inds = b[1] |> adapt(backend)

    band = (; inds, energyinds)

    eref = force[2]

    @info "Running DNS simulation"
    flush(stderr)
    t = 0.0
    k = 0
    times = [t]
    energies = [energy(u)]
    walltime = time()
    while t < tstop
        Δt = cfl * propose_timestep(u, g, visc, cache)
        Δt = min(Δt, tstop - t)
        t += Δt
        k += 1

        # Step
        wray3!(convectiondiffusion!, u, Δt, g, cache; visc)

        # Reinject energy in forced band
        # Current energy in band
        e = sum(u -> sum(abs2, view(u, band.energyinds)) / 2, u)

        # Scaling factor that enforces energy
        fac = sqrt(eref / e)

        # Scale components in band
        for u in u
            uband = view(u, band.inds)
            uband .*= fac
        end

        if k % 1 == 0
            e = energy(u)
            push!(times, t)
            push!(energies, e)
            @info join(
                [
                    "k = $k",
                    "t = $(round(t; sigdigits = 4))",
                    "Δt = $(round(Δt; sigdigits = 4))",
                    # "umax = $(round(maximum(u -> maximum(abs, u), u); sigdigits = 4))",
                    "energy = $(round(e; sigdigits = 4))",
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
    jldsave(file; u = u |> cpu_device(), times, energies, walltime)
end

function create_data(setup; cfl, nstep, nsubstep)
    (; visc, D, n_dns, n_les, backend, force, outdir, Δ) = setup
    g_dns = Grid{D}(; setup.l, n = n_dns, backend)
    g_les = Grid{D}(; setup.l, n = n_les, backend)
    c_dns = getcache(g_dns)
    c_les = getcache(g_les)

    # Allocate arrays
    ubar = vectorfield(g_les)
    σbar1 = tensorfield(g_les)
    σbar2 = tensorfield(g_les)
    trace = scalarfield(g_les)
    τ = tensorfield(g_les)
    inputs = fill(map(Array, ubar), 0)
    outputs = fill(map(Array, τ), 0)

    # Load DNS state from warm-up simulation
    u = load(joinpath(outdir, "dns.jld2"), "u") |> adapt(backend)

    # Forcing stuff
    b = getband(g_dns, force[1])

    # Indices for computing energy (with conjugate indices)
    energyinds = vcat(b...) |> adapt(backend)

    # Components to be forced (without conjugate indices)
    inds = b[1] |> adapt(backend)

    band = (; inds, energyinds)

    # These energies will be maintained throughout the simulation.
    eref = force[2]

    # Spectra
    stuff_dns = Seneca.spectral_stuff(g_dns)
    stuff_les = Seneca.spectral_stuff(g_les)
    spectra_dns = fill(zeros(0), 0)
    spectra_les = fill(zeros(0), 0)

    # Compute turbulence statistics (use σ as temporary tensor storage)
    stat = turbulence_statistics(u, visc, g_dns, c_dns.σ)
    statistics = fill(stat, 0)

    times = zeros(0)

    # Time stepping
    t = 0.0
    for i = 1:nstep
        # Do multiple substeps before storing data
        # Skip first step to get initial statistics
        i == 1 || for j = 1:nsubstep
            # Time step
            Δt = cfl * propose_timestep(u, g_dns, visc, c_dns)
            t += Δt

            # Evolve DNS
            wray3!(convectiondiffusion!, u, Δt, g_dns, c_dns; visc)

            # Reinject energy in forced band
            # Current energy in band
            e = sum(u -> sum(abs2, view(u, band.energyinds)) / 2, u)

            # Scaling factor that enforces energy
            fac = sqrt(eref / e)

            # Scale components in band
            for u in u
                uband = view(u, band.inds)
                uband .*= fac
            end

            # Log
            if j == nsubstep && i % 1 == 0
                e = energy(u)
                @info join(
                    [
                        "i = $i",
                        "t = $(round(t; sigdigits = 4))",
                        "Δt = $(round(Δt; sigdigits = 4))",
                        # "umax = $(round(maximum(u -> maximum(abs, u), u); sigdigits = 4))",
                        "energy = $(round(e; sigdigits = 4))",
                    ],
                    ",\t",
                )
            end
        end

        # Compute ubar and sub-filter stress
        sfs!(;
            τ,
            trace,
            σbar1,
            σbar2,
            ubar,
            u,
            c_dns,
            c_les,
            g_dns,
            g_les,
            Δ,
            kforce = force[1],
        )

        # Save current (ubar,tau)-pair
        push!(inputs, map(Array, ubar))
        push!(outputs, map(Array, τ))

        # Compute spectra
        s_dns = spectrum(u, g_dns, stuff_dns)
        s_les = spectrum(ubar, g_les, stuff_les)
        push!(spectra_dns, s_dns.s)
        push!(spectra_les, s_les.s)

        # Compute turbulence statistics (use σ as temporary tensor storage)
        stat = turbulence_statistics(u, visc, g_dns, c_dns.σ)
        push!(statistics, stat)

        # Keep track of times
        push!(times, t)
    end

    (; inputs, outputs, times, spectra_dns, spectra_les, statistics)
end

export sfs
function sfs(u, g_dns, g_les, Δ, kforce)
    c_dns = getcache(g_dns)
    c_les = getcache(g_les)
    ubar = vectorfield(g_les)
    σbar1 = tensorfield(g_les)
    σbar2 = tensorfield(g_les)
    trace = scalarfield(g_les)
    τ = tensorfield(g_les)
    sfs!(; τ, trace, σbar1, σbar2, ubar, u, c_dns, c_les, g_dns, g_les, Δ, kforce)
    τ
end

export sfs!
function sfs!(; τ, trace, σbar1, σbar2, ubar, u, c_dns, c_les, g_dns, g_les, Δ, kforce)
    D = dim(g_dns)
    nonlinearity!(c_dns.σ, c_dns.vi_vj, c_dns.v, u, c_dns.plan, g_dns)
    for (ubar, u) in zip(ubar, u)
        apply!(cutoff!, g_les, (ubar, u))
        isnothing(Δ) || apply!(gaussianfilter!, g_les, (ubar, Δ, kforce, g_les))
    end
    for (σbar1, σ) in zip(σbar1, c_dns.σ)
        apply!(cutoff!, g_les, (σbar1, σ))
        isnothing(Δ) || apply!(gaussianfilter!, g_les, (σbar1, Δ, kforce, g_les))
    end
    nonlinearity!(σbar2, c_les.vi_vj, c_les.v, ubar, c_les.plan, g_les)
    foreach(i -> (τ[i] .= σbar1[i] .- σbar2[i]), 1:tensordim(g_dns))
    foreach(τ -> apply!(twothirds!, g_les, (τ, g_les)), τ)

    # Make tensor trace-free
    if D == 2
        @. trace = (τ[1] + τ[2]) / 2
        τ[1] .-= trace
        τ[2] .-= trace
    elseif D == 3
        @. trace = (τ[1] + τ[2] + τ[3]) / 3
        τ[1] .-= trace
        τ[2] .-= trace
        τ[3] .-= trace
    end
end

function create_dataloader(setup, data; batchsize)
    (; D, Δ) = setup
    g = Grid{D}(; setup.l, n = setup.n_les, setup.backend)
    G = tensorfield_nonsym(g)
    u = vectorfield(g)
    τ = scalarfield(g)
    GG = spacetensorfield_nonsym(g)
    ττ = spacetensorfield(g)
    plan = plan_rfft(GG.xx)
    T = typeof(setup.l)
    snaps = map(zip(data...)) do (ucpu, τcpu)
        foreach(copyto!, u, ucpu)
        apply!(vectorgradient!, g, (G, u, g))
        for (GG, G) in zip(GG, G)
            apply!(twothirds!, g, (G, g))
            ldiv!(GG, plan, G) # Inverse RFFT
            GG .*= g.n^D # FFT factor
        end
        for (ττ, τcpu) in zip(ττ, τcpu)
            copyto!(τ, τcpu)
            apply!(twothirds!, g, (τ, g))
            ldiv!(ττ, plan, τ) # Inverse RFFT
            ττ .*= g.n^D # FFT factor
        end
        x = stack(GG)
        y = stack(ττ)
        A2 = sum(abs2, x; dims = D + 1) # VGT squared norm
        @. x ./= (sqrt(A2) + eps(T)) # Normalize input gradient
        @. y ./= (Δ^2 * A2 + eps(T)) # Normalize output stress
        (x, y) |> cpu_device()
    end
    x = stack(first, snaps)
    y = stack(last, snaps)
    if D == 3
        # Put one of the spatial dimensions into the batch dimension,
        # so that each snapshot is smaller. The models do not use
        # neighboring information anyway.
        x = permutedims(x, (1, 2, 4, 3, 5))
        y = permutedims(y, (1, 2, 4, 3, 5))
        nx = size(x, 1)
        x = reshape(x, nx, nx, 1, size(x, 3), :)
        y = reshape(y, nx, nx, 1, size(y, 3), :)
    end
    DataLoader((x, y); batchsize, shuffle = true, partial = false)
end

function vectorfield_to_svector(u)
    D = ndims(u[1])
    T = eltype(u[1])
    M = SVector{D,T}
    M.(u...)
end
function svector_to_vectorfield(u)
    V = eltype(u)
    z = zero(V)
    D = size(z, 1)
    if D == 2
        (; x = getindex.(u, 1), y = getindex.(u, 2))
    elseif D == 3
        (; x = getindex.(u, 1), y = getindex.(u, 2), z = getindex.(u, 3))
    end
end

function tensorfield_to_smatrix(t)
    D = ndims(t[1])
    T = eltype(t[1])
    M = SMatrix{D,D,T,D^2}
    M.(t...)
end
function smatrix_to_tensorfield(t)
    M = eltype(t)
    z = zero(M)
    D = size(z, 1)
    if D == 2
        (;
            xx = getindex.(t, 1, 1),
            yx = getindex.(t, 2, 1),
            xy = getindex.(t, 1, 2),
            yy = getindex.(t, 2, 2),
        )
    elseif D == 3
        (;
            xx = getindex.(t, 1, 1),
            yx = getindex.(t, 2, 1),
            zx = getindex.(t, 3, 1),
            xy = getindex.(t, 1, 2),
            yy = getindex.(t, 2, 2),
            zy = getindex.(t, 3, 2),
            xz = getindex.(t, 1, 3),
            yz = getindex.(t, 2, 3),
            zz = getindex.(t, 3, 3),
        )
    end
end

export inverse_vector_fourier
function inverse_vector_fourier(u, g)
    uu = spacevectorfield(g)
    temp = scalarfield(g)
    plan = plan_rfft(uu.x)
    for (uu, u) in zip(uu, u)
        copyto!(temp, u)
        temp .*= g.n^dim(g) # FFT factor
        apply!(twothirds!, g, (temp, g))
        ldiv!(uu, plan, temp)
    end
    uu
end

export forward_vector_fourier
function forward_vector_fourier(uu, g)
    u = vectorfield(g)
    plan = plan_rfft(uu.x)
    for (uu, u) in zip(uu, u)
        mul!(u, plan, uu)
        u ./= g.n^dim(g) # FFT factor
    end
    u
end

function transform_vector(u, g, (p, s))
    T, D = typeof(g.l), dim(g)
    u_sa = SVector{D,T}.(u...)
    u_sa = permutedims(u_sa, p)
    dims = (findall(==(-1), s)...,)
    u_sa = reverse(u_sa; dims)
    m = roto_reflection_matrix(p, s)
    ru_sa = map(u -> m * u, u_sa)
    if D == 2
        (; x = getindex.(ru_sa, 1), y = getindex.(ru_sa, 2))
    elseif D == 3
        (; x = getindex.(ru_sa, 1), y = getindex.(ru_sa, 2), z = getindex.(ru_sa, 3))
    end
end

function transform_tensor(t, g, (p, s))
    T, D = typeof(g.l), dim(g)
    SM = SMatrix{D,D,T,D^2}
    t = if D == 2
        SM.(t.xx, t.xy, t.xy, t.yy)
    else
        SM.(t.xx, t.xy, t.zx, t.xy, t.yy, t.yz, t.zx, t.yz, t.zz)
    end
    t = permutedims(t, p)
    dims = (findall(==(-1), s)...,)
    t = reverse(t; dims)
    m = roto_reflection_matrix(p, s)
    t = map(t -> m * t * m', t)
    if D == 2
        (; xx = getindex.(t, 1, 1), yy = getindex.(t, 2, 2), xy = getindex.(t, 1, 2))
    elseif D == 3
        (;
            xx = getindex.(t, 1, 1),
            yy = getindex.(t, 2, 2),
            zz = getindex.(t, 3, 3),
            xy = getindex.(t, 1, 2),
            yz = getindex.(t, 2, 3),
            zx = getindex.(t, 3, 1),
        )
    end
end

function transform_tensor_nonsym(t, g, (p, s))
    T, D = typeof(g.l), dim(g)
    SM = SMatrix{D,D,T,D^2}
    t = SM.(t...)
    t = permutedims(t, p)
    dims = (findall(==(-1), s)...,)
    t = reverse(t; dims)
    m = roto_reflection_matrix(p, s)
    t = map(t -> m * t * m', t)
    pairs = map(Iterators.product(1:D, 1:D)) do (i, j)
        symbols = :x, :y, :z
        s = Symbol(symbols[i], symbols[j])
        val = getindex.(t, i, j)
        s => val
    end
    NamedTuple(pairs)
end

# function transform_scalar(f, (p, s))
#     f = permutedims(f, p)
#     dims = (findall(==(-1), s)...,)
#     f = reverse(f; dims)
# end
# function transform_vector(u, (p, s))
#     u = permutedims(u, p)
#     dims = (findall(==(-1), s)...,)
#     u = reverse(u; dims)
#     m = roto_reflection_matrix(p, s)
#     u = map(u -> m * u, u)
# end
# function transform_tensor(t, (p, s))
#     t = permutedims(t, p)
#     dims = (findall(==(-1), s)...,)
#     t = reverse(t; dims)
#     m = roto_reflection_matrix(p, s)
#     map(t -> m * t * m', t)
# end

create_loss(project) = function loss(net, ps, st, (x, y))
    ps = project(ps)
    yhat = net(x, ps, st) |> first
    # l = MSELoss()(yhat, y) # (x, y) pair is already normalized
    l = sum(abs2, yhat - y) / sum(abs2, y)
    l, st, (;)
end

function train(; loss, setup, dataloader, nepoch, learning_rate, net_stuff)
    (; backend) = setup
    (; net, ps, st) = net_stuff
    ps = deepcopy(ps)
    device = adapt(backend)
    opt = AdamW(learning_rate)
    train_state = Training.TrainState(net, ps, st, opt)
    b_valid = first(dataloader) |> device
    ps_best = deepcopy(ps)
    l_best = Inf
    losses_train = zeros(0)
    losses_valid = zeros(0)
    i = 0
    for iepoch = 1:nepoch, (ibatch, batch) in enumerate(dataloader)
        i += 1
        x, y = batch |> device
        # loss(net, ps, st, (x, y)); error()
        _, l_train, _, train_state =
            Training.single_train_step!(AutoZygote(), loss, (x, y), train_state)
        if ibatch % 1 == 0
            # Check performance on validation batch
            l_valid = loss(net, ps, st, b_valid) |> first

            # Log
            @info join(
                [
                    "iepoch = $iepoch",
                    "ibatch = $ibatch",
                    "loss (valid) = $(round(l_valid; sigdigits = 4))",
                    "loss (train) = $(round(l_train; sigdigits = 4))",
                ],
                ",\t",
            )
            push!(losses_train, l_train)
            push!(losses_valid, l_valid)

            # Keep current best parameters
            if l_valid < l_best
                l_best = l_valid
                ps_best = deepcopy(train_state.parameters)
            end
        end
    end
    ps = ps_best # Retain best (not last) parameters
    st = train_state.states # Note: If st is non-empty, need to make "best"-mechanism for states
    (; ps, st, losses_train, losses_valid)
end

function fullchain(setup, net, project, ps, st, Δ)
    (; D) = setup
    ps = project(ps)
    function model(x)
        x = stack(x) # Convert named tuple to array
        T = eltype(x)
        s = size(x)
        x = reshape(x, s..., 1) # Add singleton sample dimension
        A2 = sum(abs2, x; dims = D + 1) # VGT squared norm
        @. x /= (sqrt(A2) + eps(T)) # Normalize input gradient
        y = net(x, ps, st) |> first # Apply model
        @. y *= Δ^2 * A2 # Scale output with dimensional stuff
        reshape(y, s[1:D]..., :) # Remove singleton sample dimension
    end
    model
end

"Compute LES right-hand side with closure model (put force in `du`)."
function les!(du, u, grid, cache; model, visc)
    D = dim(grid)
    (; plan, σ, vi_vj, v, G) = cache

    # Coarse DNS stress
    stress!(σ, vi_vj, v, u, plan, visc, grid)

    # Closure model stress (in physical space)
    apply!(vectorgradient!, grid, (G, u, grid))
    x = map(G) do G
        apply!(twothirds!, grid, (G, grid))
        res = plan \ G
        res .*= grid.n^D # FFT factor
        res
    end
    # x = stack(x)
    # x = reshape(x, size(x)..., 1)
    y = model(x)

    # Add closure stress to existing stress (in spectral space)
    for (i, σ) in enumerate(σ)
        # Use vi_vj and du.x as temp storage
        copyto!(vi_vj, selectdim(y, D + 1, i))
        mul!(du.x, plan, vi_vj)
        @. σ += du.x / grid.n^D # With FFT factor
    end

    # Final force
    apply!(tensordivergence!, grid, (du, σ, grid))
end

function inference_post(; u_dns, setup, models, files, cfl, tstop, dodns)
    (; D, l, n_dns, n_les, backend, visc, Δ, nshell) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)

    # Initial ubar
    @info "Filtering initial DNS"
    u_les = vectorfield(g_les)
    for (u_les, u_dns) in zip(u_les, u_dns)
        apply!(cutoff!, g_les, (u_les, u_dns))
        isnothing(Δ) || apply!(gaussianfilter!, g_les, (u_les, Δ, nshell + 1, g_les))
    end

    if dodns
        # Solve DNS
        @info "Solving DNS"
        t = time()
        solve_les!(u_dns; grid = g_dns, visc, model = nothing, tstop, cfl)
        t = time() - t

        # Compute final filtered DNS
        @info "Filtering final DNS"
        u_ref = vectorfield(g_les)
        for (u_ref, u_dns) in zip(u_ref, u_dns)
            apply!(cutoff!, g_les, (u_ref, u_dns))
            isnothing(Δ) || apply!(gaussianfilter!, g_les, (u_ref, Δ, nshell + 1, g_les))
        end

        # Save DNS results and free up memory
        jldsave(files.dns; u = u_dns |> cpu_device(), timing = t)
        jldsave(files.ref; u = u_ref |> cpu_device(), timing = t)
        u_dns = nothing
        u_ref = nothing
    end

    # Solve LES for each model
    u_model = vectorfield(g_les)
    for key in keys(models)
        @info "Solving LES with $(key)"
        model = models[key]

        # Do a short model warmup (so that compilation does not get included in timing)
        twarm = 1e-6
        foreach(copyto!, u_model, u_les)
        solve_les!(u_model; grid = g_les, visc, model, tstop = twarm, cfl)

        # Solve LES
        foreach(copyto!, u_model, u_les)
        t = time()
        solve_les!(u_model; grid = g_les, visc, model, tstop, cfl)
        t = time() - t

        # Save results
        jldsave(files[key]; u = u_model |> cpu_device(), timing = t)
    end
end

export solve_les!
function solve_les!(u; grid, visc, model, cfl, tstop)
    t = 0.0
    cache = getcache(grid)
    if !isnothing(model)
        # Allocate velocity gradient for closure
        cache = (; cache..., G = tensorfield_nonsym(grid))
    end
    t = zero(tstop)
    i = 0
    while t < tstop
        Δt = cfl * propose_timestep(u, grid, visc, cache)
        Δt = min(Δt, tstop - t)
        t += Δt
        if isnothing(model)
            # Without closure
            wray3!(convectiondiffusion!, u, Δt, grid, cache; visc)
        else
            # With closure
            wray3!(les!, u, Δt, grid, cache; model, visc)
        end
        if i % 5 == 0
            energy = Seneca.energy(u)
            @info join(
                [
                    "i = $i",
                    "t = $(round(t; sigdigits = 4))",
                    "Δt = $(round(Δt; sigdigits = 4))",
                    # "umax = $(round(maximum(u -> maximum(abs, u), u); sigdigits = 4))",
                    "energy = $(round(energy; sigdigits = 4))",
                ],
                ",\t",
            )
        end
        i += 1
    end
    foreach(u -> apply!(twothirds!, grid, (u, grid)), u)
    u
end

function get_errors(setup, u_les)
    u_les = map(stack, u_les)
    k = keys(u_les)
    k_les = filter(!=(:dns), k)
    k_les = filter(!=(:ref), k_les)
    u_ref = u_les.ref |> adapt(setup.backend)
    map(k_les) do key
        u = u_les[key] |> adapt(setup.backend)
        err = norm(u - u_ref) / norm(u_ref)
        println(key => round(err; sigdigits = 4))
        key => err
    end
end

# 2D basis
@inline nbasis(::Grid{2}) = 2 # Number of entries below
@inline getbasis(::Grid{2}, S, R) = (
    deviator(S),
    deviator(S * R - R * S),
    # deviator(S * S),
    # deviator(R * R),
)

# # Second-order only basis:
# # https://arc.aiaa.org/doi/10.2514/6.2022-0595
# @inline nbasis(::Grid{3}) = 5
# @inline getbasis(::Grid{3}, S, R) =
#     (deviator(S), deviator(S * S), deviator(R * R), deviator(S * R - R * S))

# New reduced basis:
# https://arc.aiaa.org/doi/10.2514/6.2022-0595
@inline nbasis(::Grid{3}) = 7
@inline getbasis(::Grid{3}, S, R) = (
    deviator(S),
    deviator(S * S),
    deviator(R * R),
    deviator(S * R - R * S),
    deviator(R * S * R),
    deviator(R * S * S - S * S * R),
    deviator(R * S * R * R - R * R * S * R),
)

# # Pope's basis:
# @inline nbasis(::Grid{3}) = 11
# @inline getbasis(::Grid{3}, S, R) = (
#     deviator(S),
#     deviator(S * R - R * S),
#     deviator(S * S),
#     deviator(R * R),
#     deviator(R * S * S - S * S * R),
#     deviator(S * R * R + R * R * S),
#     deviator(R * S * R * R - R * R * S * R),
#     deviator(S * R * S * S - S * S * R * S),
#     deviator(R * R * S * S + S * S * R * R),
#     deviator(R * S * S * R * R - R * R * S * S * R),
# )

# 2D invariants
@inline ninvariant(::Grid{2}) = 2
@inline getinvariants(::Grid{2}, S, R) = tr(S * S), tr(R * R)

# 3D invariants
@inline ninvariant(::Grid{3}) = 5
@inline getinvariants(::Grid{3}, S, R) =
    tr(S * S), tr(R * R), tr(S * S * S), tr(S * R * R), tr(S * S * R * R)

"Compute deviatoric part of a tensor."
@inline deviator(σ::SMatrix{2,2}) = σ - tr(σ) / 2 * one(σ)
@inline deviator(σ::SMatrix{3,3}) = σ - tr(σ) / 3 * one(σ)

@kernel function tb_kernel!(invariants, basis, grads, Δ, g::Grid{2})
    nb, ni = nbasis(g), ninvariant(g)
    I = @index(Global, Cartesian)
    Gxx, Gyx, Gxy, Gyy = grads.xx[I], grads.yx[I], grads.xy[I], grads.yy[I]
    A = @SMatrix [Gxx Gxy; Gyx Gyy]
    A2 = sum(abs2, A)
    A = A / (sqrt(A2) + eps(eltype(A))) # Normalize gradient
    S, R = (A + A') / 2, (A - A') / 2
    i, b = getinvariants(g, S, R), getbasis(g, S, R)
    for iinv in Base.OneTo(ni)
        invariants[I, iinv] = i[iinv]
    end
    for ibas in Base.OneTo(nb)
        # Convert symmetric 2x2 tensor b to
        # flattened symmetric tensor [xx, yy, xy].
        # Also premultiply by Δ^2 * |A|^2, since the output tensor is
        # Δ^2 * |A|^2 * coeffs * basis
        basis[I, 1, ibas] = b[ibas][1, 1] * A2 * Δ^2
        basis[I, 2, ibas] = b[ibas][2, 2] * A2 * Δ^2
        basis[I, 3, ibas] = b[ibas][1, 2] * A2 * Δ^2
    end
end

@kernel function tb_kernel!(invariants, basis, grads, Δ, g::Grid{3})
    ni, nb = ninvariant(g), nbasis(g)
    I = @index(Global, Cartesian)
    Axx, Axy, Axz = grads.xx[I], grads.xy[I], grads.xz[I]
    Ayx, Ayy, Ayz = grads.yx[I], grads.yy[I], grads.yz[I]
    Azx, Azy, Azz = grads.zx[I], grads.zy[I], grads.zz[I]
    A = @SMatrix [Axx Axy Axz; Ayx Ayy Ayz; Azx Azy Azz]
    A2 = sum(abs2, A)
    A = A / (sqrt(A2) + eps(eltype(A))) # Normalize gradient
    S, R = (A + A') / 2, (A - A') / 2
    i, b = getinvariants(g, S, R), getbasis(g, S, R)
    for iinv in Base.OneTo(ni)
        invariants[I, iinv] = i[iinv]
    end
    for ibas in Base.OneTo(nb)
        # Convert symmetric 3x3 tensor b to flattened symmetric tensor [xx, yy, zz, xy, yz, zx]
        # Also premultiply by Δ^2 * |A|^2, since the output tensor is
        # Δ^2 * |A|^2 * coeffs * basis
        basis[I, 1, ibas] = b[ibas][1, 1] * A2 * Δ^2
        basis[I, 2, ibas] = b[ibas][2, 2] * A2 * Δ^2
        basis[I, 3, ibas] = b[ibas][3, 3] * A2 * Δ^2
        basis[I, 4, ibas] = b[ibas][1, 2] * A2 * Δ^2
        basis[I, 5, ibas] = b[ibas][2, 3] * A2 * Δ^2
        basis[I, 6, ibas] = b[ibas][3, 1] * A2 * Δ^2
    end
end

function build_tensorbasis(grad, g, Δ)
    T = typeof(g.l)
    nx, nb, ni, nt = space_ndrange(g), nbasis(g), ninvariant(g), tensordim(g)
    basis = KernelAbstractions.zeros(g.backend, T, nx..., nt, nb)
    invariants = KernelAbstractions.zeros(g.backend, T, nx..., ni)
    apply!(tb_kernel!, g, (invariants, basis, grad, Δ, g); ndrange = nx)
    invariants, basis
end

function getgradient(u, g)
    D = dim(g)
    A = tensorfield_nonsym(g)
    AA = spacetensorfield_nonsym(g)
    apply!(vectorgradient!, g, (A, u, g))
    plan = plan_rfft(AA.xx)
    for (AA, A) in zip(AA, A)
        apply!(twothirds!, g, (A, g))
        ldiv!(AA, plan, A) # Inverse RFFT
        AA .*= g.n^D # FFT factor
    end
    AA
end

tbnn(net, ps, st, Δ, g) = function model(A)
    nx = space_ndrange(g)
    nt = tensordim(g)
    nb = nbasis(g)

    # Compute invariants and basis tensors
    invariants, basis = build_tensorbasis(A, g, Δ)

    # Compute coefficients
    invariants = reshape(invariants, size(invariants)..., 1) # One sample
    w = net(invariants, ps, st) |> first

    # Basis contraction
    b = reshape(basis, :, nt, nb)
    w = reshape(w, :, 1, nb)
    b .*= w
    m = sum(b; dims = 3)
    reshape(m, nx..., nt)
end

function create_dataloader_tbnn(setup, data; batchsize, rng)
    (; D, Δ) = setup
    g = Grid{D}(; setup.l, n = setup.n_les, setup.backend)
    nx = space_ndrange(g)
    u = vectorfield(g)
    τ = scalarfield(g)
    ττ = spacetensorfield(g)
    plan = plan_rfft(ττ.xx)
    snaps = map(zip(data...)) do (ucpu, τcpu)
        foreach(copyto!, u, ucpu)
        G = getgradient(u, g)
        for (ττ, τcpu) in zip(ττ, τcpu)
            copyto!(τ, τcpu)
            apply!(twothirds!, g, (τ, g))
            ldiv!(ττ, plan, τ) # Inverse RFFT
            ττ .*= g.n^D # FFT factor
        end
        i, b = build_tensorbasis(G, g, Δ)
        i = i |> cpu_device()
        b = reshape(b, nx..., :) |> cpu_device()
        x = cat(i, b; dims = D + 1)
        y = reshape(stack(ττ), nx..., tensordim(g)) |> cpu_device()
        x, y
    end
    x = stack(first, snaps)
    y = stack(last, snaps)
    if D == 3
        # Put one of the spatial dimensions into the batch dimension,
        # so that each snapshot is smaller. The models do not use
        # neighboring information anyway.
        x = permutedims(x, (1, 2, 4, 3, 5))
        y = permutedims(y, (1, 2, 4, 3, 5))
        nxx = nx[1]
        x = reshape(x, nxx, nxx, 1, size(x, 3), :)
        y = reshape(y, nxx, nxx, 1, size(y, 3), :)
    end
    DataLoader((x, y); batchsize, shuffle = true, partial = false, rng)
end

create_loss_tbnn(g) = function loss(net, ps, st, (x, y))
    D = dim(g)
    nx = size(x)[1:D]
    nt = tensordim(g)
    ni = ninvariant(g)
    nb = nbasis(g)

    # Destructure invariants and basis
    i = selectdim(x, D + 1, 1:ni)
    b = selectdim(x, D + 1, (ni+1):size(x, D+1))

    # Compute coefficients
    w = net(i, ps, st) |> first

    # Basis contraction
    w = reshape(w, nx..., 1, nb, :)
    b = reshape(b, nx..., nt, nb, :)
    wb = @. w * b
    m = sum(wb; dims = D + 2)
    m = reshape(m, nx..., nt, :)

    # l = MSELoss()(m, y)
    l = sum(abs2, m - y) / sum(abs2, y)
    l, st, (;)
end

export getdissipation
function getdissipation(g, u, m)
    D = dim(g)
    G = getgradient(u, g)
    τ = m(G)
    S = (; G.xx, G.yy, xy = (G.xy .+ G.yx) ./ 2)
    if D == 2
        xx, yy, xy = 1, 2, 3
        τ = (;
            xx = selectdim(τ, D + 1, xx),
            yy = selectdim(τ, D + 1, yy),
            xy = selectdim(τ, D + 1, xy),
        )
        @. τ.xx * S.xx + τ.yy * S.yy + 2 * τ.xy * S.xy
    elseif D == 3
        error()
    end
end

export test_equivariance_post
function test_equivariance_post(; ustart, setup, grid, model, groupindex, tstop, cfl, dolog)
    # Group element
    (; elements, permutations, signs) = group_stuff(setup.D)
    ip, is = elements[groupindex]
    p, s = permutations[ip], signs[is]

    # Initial conditions + rotated copy
    u = map(copy, ustart)
    space_u = inverse_vector_fourier(u, grid)
    space_ru = transform_vector(space_u, grid, (p, s))
    ru = forward_vector_fourier(space_ru, grid)
    foreach(u -> apply!(twothirds!, grid, (u, grid)), ru)

    # Time stepping
    (; visc) = setup
    cache = (; getcache(grid)..., G = tensorfield_nonsym(grid))
    t = zero(tstop)
    i = 0
    while t < tstop
        Δt_u = cfl * propose_timestep(u, grid, visc, cache)
        Δt_ru = cfl * propose_timestep(ru, grid, visc, cache)
        Δt = min(Δt_u, Δt_ru, tstop - t)
        t += Δt
        wray3!(les!, u, Δt, grid, cache; model, visc)
        wray3!(les!, ru, Δt, grid, cache; model, visc)
        dolog && if i % 1 == 0
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
        end
        i += 1
    end

    # Rotate stepped u
    space_su = inverse_vector_fourier(u, grid)
    space_rsu = transform_vector(space_su, grid, (p, s))
    rsu = forward_vector_fourier(space_rsu, grid)

    # Remove noisy ghost components
    foreach(u -> apply!(twothirds!, grid, (u, grid)), rsu)
    foreach(u -> apply!(twothirds!, grid, (u, grid)), ru)

    # Commutation error between rotation and time-stepping
    rsu = stack(rsu)
    sru = stack(ru)
    norm(rsu - sru) / norm(sru)
end

export plot_densities
function plot_densities(; u_dns, setup, models, dolog)
    (; plotdir, name, D, l, n_dns, n_les, backend, Δ, force) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)
    u_ref = vectorfield(g_les)
    for (u_ref, u_dns) in zip(u_ref, u_dns)
        apply!(cutoff!, g_les, (u_ref, u_dns))
        isnothing(Δ) || apply!(gaussianfilter!, g_les, (u_ref, Δ, nshell + 1, g_les))
    end
    τhat = sfs(u_dns, g_dns, g_les, Δ, force[1])
    τ = spacetensorfield(g_les)
    plan = Seneca.getplan(g_les)
    for (τ, τhat) in zip(τ, τhat)
        apply!(twothirds!, g_les, (τhat, g_les))
        ldiv!(τ, plan, τhat)
        τ .*= g_les.n^D
    end
    G = getgradient(u_ref, g_les)
    if D == 2
        S = (; G.xx, G.yy, xy = (G.xy .+ G.yx) ./ 2)
    elseif D == 3
        S = (;
            G.xx,
            G.yy,
            G.zz,
            xy = (G.xy .+ G.yx) ./ 2,
            yz = (G.yz .+ G.zy) ./ 2,
            zx = (G.zx .+ G.xz) ./ 2,
        )
    end
    τ_les = map(models) do m
        y = m(G)
        if D == 2
            xx, yy, xy = 1, 2, 3
            (; xx = view(y,:,:,xx), yy = view(y,:,:,yy), xy = view(y,:,:,xy))
        elseif D == 3
            xx, yy, zz = 1, 2, 3
            xy, yz, zx = 4, 5, 6
            (;
                xx = view(y,:,:,:,xx),
                yy = view(y,:,:,:,yy),
                zz = view(y,:,:,:,zz),
                xy = view(y,:,:,:,xy),
                yz = view(y,:,:,:,yz),
                zx = view(y,:,:,:,zx),
            )
        end
    end
    τ_all = (; ref = τ, τ_les...)
    for τ in τ_all
        for τ in τ
            plan = plan_rfft(τ)
            temp = plan * τ
            apply!(twothirds!, g_les, (temp, g_les))
            ldiv!(τ, plan, temp)
        end
        traces = @. (τ[1] + τ[2] + τ[3]) / 3
        τ[1] .-= traces
        τ[2] .-= traces
        τ[3] .-= traces
    end
    # for S in S
    #     plan = plan_rfft(S)
    #     temp = plan * S
    #     apply!(twothirds!, g_les, (temp, g_les))
    #     ldiv!(S, plan, temp)
    # end
    τxx = map(τ -> τ.xx, τ_all)
    τxy = map(τ -> τ.xy, τ_all)
    diss = map(τ_all) do τ
        d = if D == 2
            @. τ.xx * S.xx + τ.yy * S.yy + 2 * τ.xy * S.xy
        else
            @. τ.xx * S.xx +
               τ.yy * S.yy +
               τ.zz * S.zz +
               2 * (τ.xy * S.xy + τ.yz * S.yz + τ.zx * S.zx)
        end
        d
    end

    yscale = dolog ? log10 : identity

    fig = Figure(; size = (800, 300))
    labels = getlabels()

    # XX-component
    ax_xx = Makie.Axis(fig[1, 1]; xlabel = "xx-component", ylabel = "Density", yscale)
    for (key, val) in pairs(τxx)
        # val = key == :ref ? val : 1.4 * val .- 0.02
        k = val |> vec |> Array |> kde
        lines!(ax_xx, k.x, k.density; label = labels[key])
    end
    if name == "laptop"
        xlims!(ax_xx, -0.1, 0.3)
        ylims!(ax_xx, 2e-2, 3e2)
    elseif name == "turbulator"
        xlims!(ax_xx, -0.2, 0.2)
        ylims!(ax_xx, 2e-3, 2e1)
    elseif name == "snellius"
        xlims!(ax_xx, -0.2, 0.3)
        ylims!(ax_xx, 1e-3, 2e2)
    end
    # dx ux + dy ux

    # XY-component
    ax_xy = Makie.Axis(fig[1, 2]; xlabel = "xy-component", yscale)
    for val in τxy
        k = val |> vec |> Array |> kde
        lines!(ax_xy, k.x, k.density)
    end
    if name == "laptop"
        xlims!(ax_xy, -0.1, 0.1)
        ylims!(ax_xy, 1e-1, 5e2)
    elseif name == "turbulator"
        xlims!(ax_xy, -0.2, 0.2)
        ylims!(ax_xy, 4e-3, 2.5e1)
    elseif name == "snellius"
        xlims!(ax_xy, -0.17, 0.2)
        ylims!(ax_xy, 1e-3, 2e2)
    end

    # Dissipation
    ax_diss = Makie.Axis(fig[1, 3]; xlabel = "Dissipation", yscale)
    for val in diss
        k = val |> vec |> Array |> kde
        lines!(ax_diss, k.x, k.density)
    end
    if name == "laptop"
        xlims!(ax_diss, -0.3, 0.3)
        ylims!(ax_diss, 1e-1, 1e2)
    elseif name == "turbulator"
        xlims!(ax_diss, -6, 2)
        ylims!(ax_diss, 1e-3, 3e0)
    elseif name == "snellius"
        xlims!(ax_diss, -5.6, 1.3)
        ylims!(ax_diss, 1e-3, 7e0)
    end

    Legend(
        fig[0, :],
        ax_xx;
        tellwidth = false,
        tellheight = true,
        framevisible = false,
        orientation = :horizontal,
        # nbanks = 5,
    )
    # rowgap!(fig.layout, 5)

    # Save plot
    file = "$(plotdir)/tensor-distributions.pdf"
    @info "Saving density plot to $(file)"
    save(file, fig; backend = CairoMakie)
    fig
end

@kernel function clark_kernel!(ττ, GG, Δ, g::Grid{2})
    I = @index(Global, Cartesian)
    G = @SMatrix [GG.xx[I] GG.xy[I]; GG.yx[I] GG.yy[I]]
    τ = Δ^2 / 12 * G * G'
    xx, yy, xy = 1, 2, 3
    ττ[I, xx] = τ[1, 1]
    ττ[I, yy] = τ[2, 2]
    ττ[I, xy] = τ[1, 2]
end

@kernel function clark_kernel!(ττ, GG, Δ, g::Grid{3})
    I = @index(Global, Cartesian)
    G = @SMatrix [
        GG.xx[I] GG.xy[I] GG.xz[I]
        GG.yx[I] GG.yy[I] GG.yz[I]
        GG.zx[I] GG.zy[I] GG.zz[I]
    ]
    τ = Δ^2 / 12 * G * G'
    xx, yy, zz, xy, yz, zx = 1, 2, 3, 4, 5, 6
    ττ[I, xx] = τ[1, 1]
    ττ[I, yy] = τ[2, 2]
    ττ[I, zz] = τ[3, 3]
    ττ[I, xy] = τ[1, 2]
    ττ[I, yz] = τ[2, 3]
    ττ[I, zx] = τ[3, 1]
end

export create_clark
create_clark(Δ, g) = function clark(G)
    τ = stack(spacetensorfield(g))
    apply!(clark_kernel!, g, (τ, G, Δ, g); ndrange = space_ndrange(g))
    τ
end

@kernel function smagorinsky_kernel!(ττ, GG, CS, Δ, g::Grid{2})
    I = @index(Global, Cartesian)
    G = @SMatrix [GG.xx[I] GG.xy[I]; GG.yx[I] GG.yy[I]]
    S = (G + G') / 2
    nu = CS^2 * Δ^2 * sqrt(sum(abs2, S))
    τ = -2 * nu * S
    xx, yy, xy = 1, 2, 3
    ττ[I, xx] = τ[1, 1]
    ττ[I, yy] = τ[2, 2]
    ττ[I, xy] = τ[1, 2]
end

@kernel function smagorinsky_kernel!(ττ, GG, CS, Δ, g::Grid{3})
    I = @index(Global, Cartesian)
    G = @SMatrix [
        GG.xx[I] GG.xy[I] GG.xz[I]
        GG.yx[I] GG.yy[I] GG.yz[I]
        GG.zx[I] GG.zy[I] GG.zz[I]
    ]
    S = (G + G') / 2
    nu = CS^2 * Δ^2 * sqrt(sum(abs2, S))
    τ = -2 * nu * S
    xx, yy, zz, xy, yz, zx = 1, 2, 3, 4, 5, 6
    ττ[I, xx] = τ[1, 1]
    ττ[I, yy] = τ[2, 2]
    ττ[I, zz] = τ[3, 3]
    ττ[I, xy] = τ[1, 2]
    ττ[I, yz] = τ[2, 3]
    ττ[I, zx] = τ[3, 1]
end

@kernel function verstappen_kernel!(ττ, GG, C, Δ, g::Grid{3})
    I = @index(Global, Cartesian)
    G = @SMatrix [
        GG.xx[I] GG.xy[I] GG.xz[I]
        GG.yx[I] GG.yy[I] GG.yz[I]
        GG.zx[I] GG.zy[I] GG.zz[I]
    ]
    S = (G + G') / 2
    q = tr(S * S) / 2
    r = tr(S * S * S) / 3
    nu = C^2 * Δ^2 * abs(r) / q
    τ = -2 * nu * S
    xx, yy, zz, xy, yz, zx = 1, 2, 3, 4, 5, 6
    ττ[I, xx] = τ[1, 1]
    ττ[I, yy] = τ[2, 2]
    ττ[I, zz] = τ[3, 3]
    ττ[I, xy] = τ[1, 2]
    ττ[I, yz] = τ[2, 3]
    ττ[I, zx] = τ[3, 1]
end

export create_smagorinsky
function create_smagorinsky(CS, Δ, g)
    function smagorinsky(G)
        τ = stack(spacetensorfield(g))
        apply!(smagorinsky_kernel!, g, (τ, G, CS, Δ, g); ndrange = space_ndrange(g))
        τ
    end
    smagorinsky
end

export create_verstappen
function create_verstappen(C, Δ, g)
    D = dim(g)
    @assert D == 3 "Q-R model is only defined in 3D"
    function verstappen(G)
        τ = stack(spacetensorfield(g))
        apply!(verstappen_kernel!, g, (τ, G, C, Δ, g); ndrange = space_ndrange(g))
        τ
    end
    verstappen
end

export apriori_error
function apriori_error(; u_dns, setup, models)
    (; D, l, n_dns, n_les, backend, Δ, force) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)

    # Inputs
    u_ref = vectorfield(g_les)
    for (u_ref, u_dns) in zip(u_ref, u_dns)
        apply!(cutoff!, g_les, (u_ref, u_dns))
        isnothing(Δ) || apply!(gaussianfilter!, g_les, (u_ref, Δ, nshell + 1, g_les))
    end
    G = getgradient(u_ref, g_les)

    # Outputs
    τhat = sfs(u_dns, g_dns, g_les, Δ, force[1])
    τ = spacetensorfield(g_les)
    plan = Seneca.getplan(g_les)
    for (τ, τhat) in zip(τ, τhat)
        apply!(twothirds!, g_les, (τhat, g_les))
        ldiv!(τ, plan, τhat)
        τ .*= g_les.n^dim(g_les)
    end
    τ = stack(τ)

    errors = map(models) do m
        # Predict stress
        y = m(G)

        # Make trace free
        ydiag = selectdim(y, D + 1, 1:D)
        trace = sum(ydiag; dims = D + 1)
        @. ydiag -= trace / D

        # Metrics
        yy, ττ = y .- mean(y), τ .- mean(τ)
        relerr = norm(y - τ) / norm(τ)
        crosscor = dot(yy, ττ) / sqrt(dot(yy, yy) * dot(ττ, ττ))
        (; relerr, crosscor)
    end
    errors
end

export apriori_equivariance_error
function apriori_equivariance_error(; u, setup, models)
    (; D, l, n_les, backend) = setup
    (; elements, permutations, signs) = group_stuff(D)
    g = Grid{D}(; l, n = n_les, backend)
    u_ref = u.ref |> adapt(backend)
    G = getgradient(u_ref, g)
    errors = map(keys(models)) do key
        @info "Computing a-priori equi errors for $(key)"
        m = models[key]
        mG = m(G)
        mG_split = if D == 2
            xx, yy, xy = 1, 2, 3
            (;
                xx = selectdim(mG, D + 1, xx),
                yy = selectdim(mG, D + 1, yy),
                xy = selectdim(mG, D + 1, xy),
            )
        else
            xx, yy, zz = 1, 2, 3
            xy, yz, zx = 4, 5, 6
            (;
                xx = selectdim(mG, D + 1, xx),
                yy = selectdim(mG, D + 1, yy),
                zz = selectdim(mG, D + 1, zz),
                xy = selectdim(mG, D + 1, xy),
                yz = selectdim(mG, D + 1, yz),
                zx = selectdim(mG, D + 1, zx),
            )
        end
        err = map(elements) do e
            ip, is = e
            p, s = permutations[ip], signs[is]
            rG = transform_tensor_nonsym(G, g, (p, s))
            mrG = m(rG)
            rmG_split = transform_tensor(mG_split, g, (p, s))
            rmG = stack(rmG_split)
            norm(rmG - mrG) / norm(mrG)
        end
        key => err
    end
    errors |> NamedTuple
end

export plot_velocities
function plot_velocities(setup, u, comp)
    (; D, l, n_dns, n_les, backend) = setup
    fig = Figure(; size = (800, 440))
    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)
    ui = scalarfield(g_dns)
    ui_space = spacescalarfield(g_dns)
    plan = plan_rfft(ui_space)
    labels = getlabels()
    for (k, key) in u |> keys |> enumerate
        title = labels[key]
        j, i = CartesianIndices((4, 2))[k].I
        ax = Axis(
            fig[i, j];
            xlabelvisible = false,
            xticksvisible = false,
            xticklabelsvisible = false,
            ylabelvisible = false,
            yticksvisible = false,
            yticklabelsvisible = false,
            aspect = DataAspect(),
            title,
        )
        if key == :dns
            copyto!(ui, u[key][comp])
        else
            ubar_i = u[key][comp] |> adapt(backend)
            fill!(ui, 0)
            apply!(inverse_cutoff!, g_les, (ui, ubar_i))
        end
        ldiv!(ui_space, plan, ui) # Make copy, ldiv! overwrites...
        ui_space .*= g_dns.n^3 # FFT factor
        data = ui_space[:, :, end] |> Array
        range = (:, :)
        # range = (40:60, 40:60)
        data = ui_space[range..., end] |> Array
        # @show typeof(data); error()
        image!(ax, data; colormap = :seaborn_icefire_gradient, interpolate = false)
    end
    fig
end

export get_dissipation_errors
function get_dissipation_errors(; setup, u_dns, models)
    (; D, l, n_dns, n_les, backend, Δ, force) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)
    ubar = vectorfield(g_les)
    for (ubar, u_dns) in zip(ubar, u_dns)
        apply!(cutoff!, g_les, (ubar, u_dns))
        isnothing(Δ) || apply!(gaussianfilter!, g_les, (ubar, Δ, nshell + 1, g_les))
    end
    τhat = sfs(u_dns, g_dns, g_les, Δ, force)
    τ = spacetensorfield(g_les)
    plan = Seneca.getplan(g_les)
    for (τ, τhat) in zip(τ, τhat)
        # apply!(twothirds!, g_les, (τhat, g_les))
        ldiv!(τ, plan, τhat)
        τ .*= g_les.n^dim(g_les)
    end
    G = getgradient(ubar, g_les)
    if D == 2
        S = (; G.xx, G.yy, xy = (G.xy .+ G.yx) ./ 2)
    elseif D == 3
        S = (;
            G.xx,
            G.yy,
            G.zz,
            xy = (G.xy .+ G.yx) ./ 2,
            yz = (G.yz .+ G.zy) ./ 2,
            zx = (G.zx .+ G.xz) ./ 2,
        )
    end
    τ_les = map(models) do m
        y = m(G)
        if D == 2
            xx, yy, xy = 1, 2, 3
            (; xx = view(y,:,:,xx), yy = view(y,:,:,yy), xy = view(y,:,:,xy))
        elseif D == 3
            xx, yy, zz = 1, 2, 3
            xy, yz, zx = 4, 5, 6
            (;
                xx = view(y,:,:,:,xx),
                yy = view(y,:,:,:,yy),
                zz = view(y,:,:,:,zz),
                xy = view(y,:,:,:,xy),
                yz = view(y,:,:,:,yz),
                zx = view(y,:,:,:,zx),
            )
        end
    end
    τ_all = (; ref = τ, τ_les...)
    diss = map(τ_all) do τ
        d = if D == 2
            @. τ.xx * S.xx + τ.yy * S.yy + 2 * τ.xy * S.xy
        else
            @. τ.xx * S.xx +
               τ.yy * S.yy +
               τ.zz * S.zz +
               2 * (τ.xy * S.xy + τ.yz * S.yz + τ.zx * S.zx)
        end
        d
    end
    map(median, NamedTuple(diss))
end

export setup_laptop
function setup_laptop()
    l = 1.0
    n_les = 128
    Δ = 2 * l / n_les
    (;
        name = "laptop",
        outdir = joinpath(@__DIR__, "..", "output", "laptop") |> mkpath,
        visc = 1e-5,
        D = 2,
        l = 1.0,
        n_dns = 2048,
        n_les,
        kpeak = 5,
        Δ,
        ou_radius = 2.3,
        ou_time = 0.005,
        ou_energy = 0.01,
        backend = CUDABackend(),
    )
end

export setup_turbulator
function setup_turbulator()
    l = 1.0
    # n_dns = 256
    n_dns = 512
    n_les = 64
    Δ = 4 * l / n_les
    outdir = joinpath(@__DIR__, "..", "output", "turbulator$(n_dns)") |> mkpath
    # plotdir = "~/Projects/SymmetryPaper/figures" |> expanduser |> mkpath
    plotdir = joinpath(outdir, "plots") |> mkpath
    (;
        name = "turbulator",
        outdir,
        plotdir,
        visc = 1e-4,
        D = 3,
        l = 1.0,
        n_dns,
        n_les,
        kpeak = 5,
        Δ,
        force = 3.5 => 0.5,
        backend = CUDABackend(),
    )
end

export setup_snellius
function setup_snellius()
    l = 1.0
    n_les = 128
    Δ = 4 * l / n_les
    (;
        name = "snellius",
        # outdir = mkpath("/projects/prjs1757/SymmetryOutput"),
        outdir = mkpath("/projects/prjs1757/SymmetryOutput2"),
        plotdir = joinpath(@__DIR__, "..", "output", "snellius") |> mkpath,
        visc = 1e-4,
        D = 3,
        l = 1.0,
        n_dns = 810,
        n_les,
        kpeak = 5,
        Δ,
        ou_radius = 2.3,
        ou_time = 0.005,
        ou_energy = 0.01,
        backend = CUDABackend(),
    )
end

export qr_kernel!
@kernel function qr_kernel!(q, r, GG, g::Grid{3})
    T = eltype(q)
    I = @index(Global, Cartesian)
    G = SMatrix{3,3,T,9}(
        GG.xx[I],
        GG.yx[I],
        GG.zx[I],
        GG.xy[I],
        GG.yy[I],
        GG.zy[I],
        GG.xz[I],
        GG.yz[I],
        GG.zz[I],
    )
    q[I] = -tr(G * G) / 2
    r[I] = -tr(G * G * G) / 3
end

export compute_qr
function compute_qr(velocities, setup)
    (; D, l, n_dns, n_les, backend) = setup
    g = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)
    Ghat = scalarfield(g)
    G = spacetensorfield_nonsym(g)
    q = spacescalarfield(g)
    r = spacescalarfield(g)
    u = vectorfield(g)
    ubar = vectorfield(g_les)
    plan = plan_rfft(G.xx)

    dens = map(keys(velocities)) do k
        @info "Computing Q-R for $(k)"
        if k == :dns
            foreach(copyto!, u, velocities[k])
        else
            for i = 1:D
                copyto!(ubar[i], velocities[k][i])
                fill!(u[i], 0)
                apply!(inverse_cutoff!, g_les, (u[i], ubar[i]))
            end
        end
        for j = 1:D, i = 1:D
            s = [:x, :y, :z]
            ij = Symbol(s[i], s[j])
            apply!(derivative!, g, (Ghat, u[i], j, g))
            apply!(twothirds!, g, (Ghat, g))
            ldiv!(G[ij], plan, Ghat) # Inverse RFFT
            G[ij] .*= g.n^D # FFT factor
        end
        t_kol = 1 / sum(G -> sum(abs2, G) * (g.l / g.n)^3, G) |> sqrt
        apply!(qr_kernel!, g, (q, r, G, g); ndrange = space_ndrange(g))
        qvec = q |> cpu_device() |> vec
        rvec = r |> cpu_device() |> vec
        qvec .*= t_kol^2
        rvec .*= t_kol^3
        args = k == :dns ? (; npoints = (1000, 1000)) : (;)
        k => kde((rvec, qvec); args...)
    end
    NamedTuple(dens)
end

export plot_qr
function plot_qr(setup, qr)
    (; name) = setup
    fig = Figure(; size = (800, 440))
    labels = getlabels()
    colorvec = Makie.wong_colors()
    lescolor = 2
    colors = (;
        line = :red,
        dns = colorvec[3],
        ref = colorvec[1],
        nomo = colorvec[lescolor],
        smag = colorvec[lescolor],
        vers = colorvec[lescolor],
        clar = colorvec[lescolor],
        tbnn = colorvec[lescolor],
        conv = colorvec[lescolor],
        equi = colorvec[lescolor],
    )
    for (k, key) in qr |> keys |> enumerate
        title = labels[key]
        j, i = CartesianIndices((4, 2))[k].I
        ax = Axis(
            fig[i, j];
            xlabelvisible = i == 2,
            xticksvisible = i == 2,
            xticklabelsvisible = i == 2,
            ylabelvisible = j == 1,
            yticksvisible = j == 1,
            yticklabelsvisible = j == 1,
            xlabel = "R",
            ylabel = "Q",
            title,
        )
        if name == "turbulator"
            ran = 1e-3, 1e2
            ncat = 6
        elseif name == "snellius"
            ran = 1e-4, 1e1
            ncat = 7
        end
        # key => extrema(qr[key].density) |> display
        isref = key == :dns || key == :ref
        isref || contour!(
            ax,
            qr.ref.x,
            qr.ref.y,
            max.(qr.ref.density, 1e-20);
            levels = logrange(ran..., ncat),
            color = colors.ref,
        )
        contour!(
            ax,
            qr[key].x,
            qr[key].y,
            max.(qr[key].density, 1e-20);
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
    fig
end

export plot_equivariance_errors
function plot_equivariance_errors(errs)
    fig = Figure(; size = (400, 340))
    ax = Axis(
        fig[1, 1];
        yscale = log10,
        xlabel = "Group element",
        ylabel = "Error",
        xticks = [1, 8, 16, 24, 32, 40, 48],
    )
    ylims!(ax, 1e-17, 1)
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
        e = max.(e, 1e-30) # Encode true zeros as 1e-30
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
    fig
end

export plot_sfs
function plot_sfs(setup, u_dns, models)
    (; D, l, n_dns, n_les, backend, Δ, force) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)
    ubar = vectorfield(g_les)
    for (ubar, u_dns) in zip(ubar, u_dns)
        apply!(cutoff!, g_les, (ubar, u_dns))
        isnothing(Δ) || apply!(gaussianfilter!, g_les, (ubar, Δ, nshell + 1, g_les))
    end
    τhat = sfs(u_dns, g_dns, g_les, Δ, force)
    τ = spacetensorfield(g_les)
    plan = Seneca.getplan(g_les)
    for (τ, τhat) in zip(τ, τhat)
        # apply!(twothirds!, g_les, (τhat, g_les))
        ldiv!(τ, plan, τhat)
        τ .*= g_les.n^dim(g_les)
    end
    τ = τ |> cpu_device()
    G = getgradient(ubar, g_les)
    τ_les = map(models) do m
        # Predict SFS
        y = m(G)

        # Make tensor trace-free
        ydiag = selectdim(y, D + 1, 1:D)
        trace = sum(ydiag; dims = D + 1)
        @. ydiag -= trace / D

        # Extract components
        if D == 2
            xx, yy, xy = 1, 2, 3
            (; xx = view(y,:,:,xx), yy = view(y,:,:,yy), xy = view(y,:,:,xy))
        elseif D == 3
            xx, yy, zz = 1, 2, 3
            xy, yz, zx = 4, 5, 6
            (;
                xx = view(y,:,:,:,xx),
                yy = view(y,:,:,:,yy),
                zz = view(y,:,:,:,zz),
                xy = view(y,:,:,:,xy),
                yz = view(y,:,:,:,yz),
                zx = view(y,:,:,:,zx),
            )
        end |> cpu_device()
    end
    τ_all = (; ref = τ, τ_les...)
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
                ylabel = "$(comp)",
                title,
                titlevisible = i == 1,
            )
            data = τ_all[key][comp]
            data = data[:, :, end]
            image!(
                ax,
                data;
                colormap = :seaborn_icefire_gradient,
                colorrange = extrema(τ_all.ref[comp][:, :, end]),
                interpolate = false,
            )
        end
    end
    rowgap!(fig.layout, 10)
    colgap!(fig.layout, 10)
    fig
end

export plot_spectrum_dns
function plot_spectrum_dns(setup)
    (; outdir, plotdir, D, l, n_dns, n_les, backend, visc, force) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)
    u = load("$(outdir)/dns.jld2", "u") |> adapt(backend)
    ubar = vectorfield(g_les)
    for (ubar, u) in zip(ubar, u)
        apply!(cutoff!, g_les, (ubar, u))
        apply!(gaussianfilter!, g_les, (ubar, setup.Δ, force[1], g_les))
    end
    D = dim(g_dns)
    stuff_dns = Seneca.spectral_stuff(g_dns)
    stuff_les = Seneca.spectral_stuff(g_les)
    stat = turbulence_statistics(u, visc, g_dns)
    @show stat.Re_tay
    s_dns = spectrum(u, g_dns, stuff_dns)
    s_les = spectrum(ubar, g_les, stuff_les)
    # l_int_new = pi / 2 / stat.uavg * sum(eachindex(s_dns.s)) do i
    #     s_dns.s[i] / stuff_dns.k[i]
    # end
    # @show stat.l_int l_int_new; error()
    fig = Figure(; size = (400, 340))
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        xlabel = L"\kappa \eta",
        xlabelsize = 20,
        ylabel = "Normalized spectrum",
    )
    if D == 2
        kkolmo = [3, g_dns.n / 10]
        kolmo = @. stat.diss^(-1 / 3) * kkolmo^(-3)
        escale = stat.diss^(-1 / 3) * stat.l_kol^(-3)
    elseif D == 3
        kkolmo = [3, g_dns.n / 8]
        # C = 0.65
        C = 0.4
        # C = 1.6
        kolmo = @. C * stat.diss^(2 / 3) * kkolmo^(-5 / 3)
        escale = stat.diss^(-2 / 3) * stat.l_kol^(-5 / 3)
        # escale = 1
    end
    # escale = 1
    kscale = stat.l_kol
    band = getband(g_dns, force[1])
    k2min = minimum(band.k2)
    k2max = maximum(band.k2)
    kforce = 2π / l * [sqrt(k2min), sqrt(k2max)]
    span = kforce * kscale
    forcecolor = Makie.wong_colors()[4]
    vspan!(ax, span...; alpha = 0.3, color = forcecolor)
    b = sqrt(prod(extrema(escale * s_dns.s)))
    a = 1.1 * span[2]
    c = sqrt(prod(span))
    w = D == 2 ? 1 : 1.5
    text!(ax, a, b / w; color = forcecolor, text = "Force")
    arr = D == 2 ? 100 : 5
    arrows2d!(
        ax,
        Point2(c, b / arr),
        Point2(c, b * arr) - Point2(c, b / arr);
        color = forcecolor,
    )
    @show kscale * s_dns.k[end]
    lines!(ax, kscale * s_dns.k, escale * s_dns.s; label = "DNS")
    lines!(ax, kscale * s_les.k, escale * s_les.s; label = "Filtered DNS")
    lines!(kscale * 2π / l * kkolmo, escale * kolmo; label = "Kolmogorov")
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
    save("$(plotdir)/spectrum-dns.pdf", fig; backend = CairoMakie)
    fig
end

export plot_spectrum_les
function plot_spectrum_les(setup, u)
    (; D, l, n_dns, n_les, backend, visc) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)
    labels = getlabels()
    u_dns = u.dns
    u_les = filter(!=(u.dns), u)
    D = dim(g_dns)
    stat = turbulence_statistics(u_dns |> adapt(backend), visc, g_dns)
    stat |> pairs |> display
    # s = spectrum(u_dns |> adapt(backend), g_dns)
    s_les = map(u -> spectrum(u |> adapt(backend), g_les), u_les)
    fig = Figure(; size = (400, 360))
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        xlabel = "Normalized wavenumber",
        ylabel = "Normalized spectrum",
    )
    k = 2π / l * [2, g_dns.n / 8]
    if D == 2
        kolmo = @. 2e0 * stat.diss^(1 / 3) * k^(-3)
        escale = stat.diss^(-2 / 3) * stat.l_kol^(-3)
    elseif D == 3
        kolmo = @. 5e-1 * stat.diss^(2 / 3) * k^(-5 / 3)
        escale = stat.diss^(-2 / 3) * stat.l_kol^(-5 / 3)
    end
    kscale = stat.l_kol
    # kscale = 1
    # lines!(ax, kscale * s.k, escale * s.s; label = "DNS")
    # lines!(kscale * k, escale * kolmo)
    for (key, val) in pairs(s_les)
        key == :vers && continue
        lines!(ax, kscale * val.k, escale * val.s; label = labels[key])
    end
    # axislegend(ax; position = :lb)
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
    # ylims!(1e-7, 1)
    save("$(setup.plotdir)/spectrum-les.pdf", fig; backend = CairoMakie)
    fig
end

export vectorfield_to_svector,
    svector_to_vectorfield, tensorfield_to_smatrix, smatrix_to_tensorfield
export transform_scalar, transform_vector, transform_tensor
export getgradient
export dns_aid, create_data, create_dataloader, train, fullchain, inference_post
export tbnn, ninvariant, nbasis, create_dataloader_tbnn, create_loss, create_loss_tbnn
export get_errors

end
