module Spectral

using Adapt
using FFTW
using KernelAbstractions
using LinearAlgebra
using Lux
using MLUtils
using Optimisers
using Random
using Seneca
using StaticArrays
using ..SymmetryCode

@inline function cutoff_index(gbar, g, i, is1)
    comp = div(g.n, gbar.n)
    imax = div(gbar.n, 2) + is1
    isneg = i > imax
    ifelse(isneg, g.n - gbar.n + i, i) # Negative wavenumbers count backwards
end
@inline cutoff_index(gbar, g, I::CartesianIndex{2}) = CartesianIndex((
    cutoff_index(gbar, g, I.I[1], true),
    cutoff_index(gbar, g, I.I[2], false),
))
@inline cutoff_index(gbar, g, I::CartesianIndex{3}) = CartesianIndex((
    cutoff_index(gbar, g, I.I[1], true),
    cutoff_index(gbar, g, I.I[2], false),
    cutoff_index(gbar, g, I.I[3], false),
))

@kernel function cutoff!(ubar, u, gbar, g)
    I = @index(Global, Cartesian)
    J = cutoff_index(gbar, g, I)
    ubar[I] = u[J]
end

@kernel function gaussianfilter!(u, Δ, g::Grid{2})
    I = @index(Global, Cartesian)
    kΔ = pi / Δ * 2
    kx, ky = wavenumbers(g, I)
    k2 = kx^2 + ky^2
    w = exp(-k2 / kΔ^2 / 2)
    u[I] *= w
end
@kernel function gaussianfilter!(u, Δ, g::Grid{3})
    I = @index(Global, Cartesian)
    # kΔ = pi / Δ * 2
    kx, ky, kz = wavenumbers(g, I)
    # k2 = kx^2 + ky^2 + kz^2
    # w = exp(-k2 / kΔ^2 / 2)
    # u[I] *= w
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
    foreach(i -> apply!(cutoff!, gbar, (ubar[i], u[i], gbar, g)), 1:D)
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
        foreach(i -> apply!(cutoff!, gbar, (ubar[i], u[i], gbar, g)), 1:D)
        stress!(σf, cbar.vi_vj, cbar.v, ubar, cbar.plan, visc, gbar)
        stress!(cbar.σ, cbar.vi_vj, cbar.v, v, cbar.plan, visc, gbar)
        foreach(i -> apply!(cutoff!, gbar, (fσ[i], c.σ[i], gbar, g)), 1:tensordim(g))
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
    foreach(i -> apply!(cutoff!, gbar, (ubar[i], u[i], gbar, g)), 1:D)
    sum(i -> sum(abs2, v[i] - ubar[i]) / sum(abs2, ubar[i]), 1:D)
end

function create_dns(setup; t_warmup = 0.5, cfl = 0.35, nstep = 200, nsubstep = 10, rng)
    (; l, visc, D, n_dns, backend) = setup
    g = Grid{D}(; l, n = n_dns, backend)
    u = randomfield(g; rng, kpeak = 5)
    cache = getcache(g)

    # OU stuff
    ou = ouforcer(g, 2.3)
    t_ou, estar = 0.005, 0.01
    var = sqrt(estar / t_ou)

    t = 0.0
    k = 0
    while t < t_warmup
        Δt = cfl * propose_timestep(u, g, visc, cache)
        Δt = min(Δt, t_warmup - t)
        t += Δt
        k += 1

        # Step
        wray3!(convectiondiffusion!, u, Δt, g, cache; visc)

        # Evolve OU process and inject energy
        randn!(ou.b)
        @. ou.b *= sqrt(2 * var^2 * Δt / t_ou)
        @. ou.b += (1 - Δt / t_ou) * ou.bold
        copyto!(ou.bold, ou.b)
        @. u.x[ou.iuse] += Δt * ou.b[:, 1]
        @. u.y[ou.iuse] += Δt * ou.b[:, 2]
        D == 3 && @. u.z[ou.iuse] += Δt * ou.b[:, 3]
        apply!(project!, g, (u, g))

        if k % 10 == 0
            e = energy(u)
            @info join(
                [
                    "k = $k",
                    "t = $(round(t; sigdigits = 4))",
                    # "Δt = $(round(Δt; sigdigits = 4))",
                    # "umax = $(round(maximum(u -> maximum(abs, u), u); sigdigits = 4))",
                    "energy = $(round(e; sigdigits = 4))",
                ],
                ",\t",
            )
        end
    end
    u
end

function create_data(setup; t_warmup = 0.5, cfl = 0.35, nstep = 200, nsubstep = 10, Δ, rng)
    (; visc, D, n_dns, n_les, backend) = setup
    g_dns = Grid{D}(; setup.l, n = n_dns, backend)
    g_les = Grid{D}(; setup.l, n = n_les, backend)
    u = randomfield(g_dns; rng, kpeak = 5)
    c_dns = getcache(g_dns)
    c_les = getcache(g_les)
    fu = vectorfield(g_les)
    σfu = tensorfield(g_les)
    fσu = tensorfield(g_les)
    inputs = fill(map(Array, fu), 0)
    outputs = fill(map(Array, fσu), 0)

    # OU stuff
    ou = ouforcer(g_dns, 2.3)
    t_ou, estar = 0.005, 0.01
    var = sqrt(estar / t_ou)

    # Warwm up
    t = 0.0
    k = 0
    while t < t_warmup
        Δt = cfl * propose_timestep(u, g_dns, visc, c_dns)
        Δt = min(Δt, t_warmup - t)
        t += Δt
        k += 1

        # Step
        wray3!(convectiondiffusion!, u, Δt, g_dns, c_dns; visc)

        # Evolve OU process and inject energy
        randn!(ou.b)
        @. ou.b *= sqrt(2 * var^2 * Δt / t_ou)
        @. ou.b += (1 - Δt / t_ou) * ou.bold
        copyto!(ou.bold, ou.b)
        @. u.x[ou.iuse] += Δt * ou.b[:, 1]
        @. u.y[ou.iuse] += Δt * ou.b[:, 2]
        D == 3 && @. u.z[ou.iuse] += Δt * ou.b[:, 3]
        apply!(project!, g_dns, (u, g_dns))

        if k % 10 == 0
            e = energy(u)
            @info join(
                [
                    "k = $k",
                    "t = $(round(t; sigdigits = 4))",
                    # "Δt = $(round(Δt; sigdigits = 4))",
                    # "umax = $(round(maximum(u -> maximum(abs, u), u); sigdigits = 4))",
                    "energy = $(round(e; sigdigits = 4))",
                ],
                ",\t",
            )
        end
    end

    for i = 1:nstep
        for j = 1:nsubstep
            Δt = cfl * propose_timestep(u, g_dns, visc, c_dns)
            t += Δt
            wray3!(convectiondiffusion!, u, Δt, g_dns, c_dns; visc)

            # Evolve OU process and inject energy
            randn!(ou.b)
            @. ou.b *= sqrt(2 * var^2 * Δt / t_ou)
            @. ou.b += (1 - Δt / t_ou) * ou.bold
            copyto!(ou.bold, ou.b)
            @. u.x[ou.iuse] += Δt * ou.b[:, 1]
            @. u.y[ou.iuse] += Δt * ou.b[:, 2]
            D == 3 && @. u.z[ou.iuse] += Δt * ou.b[:, 3]
            apply!(project!, g_dns, (u, g_dns))
        end
        if i % 1 == 0
            e = energy(u)
            @info join(
                [
                    "i = $i",
                    "t = $(round(t; sigdigits = 4))",
                    # "Δt = $(round(Δt; sigdigits = 4))",
                    # "umax = $(round(maximum(u -> maximum(abs, u), u); sigdigits = 4))",
                    "energy = $(round(e; sigdigits = 4))",
                ],
                ",\t",
            )
        end
        stress!(c_dns.σ, c_dns.vi_vj, c_dns.v, u, c_dns.plan, visc, g_dns)
        for (fu, u) in zip(fu, u)
            apply!(cutoff!, g_les, (fu, u, g_les, g_dns))
            apply!(gaussianfilter!, g_les, (fu, Δ, g_les))
        end
        for (fσu, σu) in zip(fσu, c_dns.σ)
            apply!(cutoff!, g_les, (fσu, σu, g_les, g_dns))
            apply!(gaussianfilter!, g_les, (fσu, Δ, g_les))
        end
        stress!(σfu, c_les.vi_vj, c_les.v, fu, c_les.plan, visc, g_les)
        foreach(i -> (fσu[i] .-= σfu[i]), 1:tensordim(g_dns))
        push!(inputs, map(Array, fu))
        push!(outputs, map(Array, fσu))
    end
    inputs, outputs
end

function sfs!(τ, σbar1, σbar2, ubar, u, c_dns, c_les, g_dns, g_les)
    D = dim(g_dns)
    nonlinearity!(c_dns.σ, c_dns.vi_vj, c_dns.v, u, c_dns.plan, g_dns)
    foreach(i -> apply!(cutoff!, g_les, (ubar[i], u[i], g_les, g_dns)), 1:D)
    foreach(
        i -> apply!(cutoff!, g_les, (σbar1[i], c_dns.σ[i], g_les, g_dns)),
        1:tensordim(g_dns),
    )
    nonlinearity!(σbar2, c_les.vi_vj, c_les.v, ubar, c_les.plan, g_les)
    foreach(i -> (τ[i] .= σbar1[i] .- σbar2[i]), 1:tensordim(g_dns))
    foreach(τ -> apply!(twothirds!, g_les, (τ, g_les)), τ)
end

function create_dataloader(setup, data; batchsize)
    (; D) = setup
    g = Grid{D}(; setup.l, n = setup.n_les, setup.backend)
    G = tensorfield_nonsym(g)
    u = vectorfield(g)
    τ = scalarfield(g)
    GG = spacetensorfield_nonsym(g)
    ττ = spacetensorfield(g)
    plan = plan_rfft(GG.xx)
    snaps = map(zip(data...)) do (ucpu, τcpu)
        map(copyto!, u, ucpu)
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
        x = reshape(stack(GG), :, D^2) |> cpu_device()
        y = reshape(stack(ττ), :, tensordim(g)) |> cpu_device()
        x, y
    end
    x = stack(first, snaps)
    y = stack(last, snaps)
    DataLoader((x, y); batchsize, shuffle = true, partial = false)
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

function transform_scalar(f, (p, s))
    f = permutedims(f, p)
    dims = (findall(==(-1), s)...,)
    f = reverse(f; dims)
end
function transform_vector(u, (p, s))
    u = permutedims(u, p)
    dims = (findall(==(-1), s)...,)
    u = reverse(u; dims)
    m = roto_reflection_matrix(p, s)
    u = map(u -> m * u, u)
end
function transform_tensor(t, (p, s))
    t = permutedims(t, p)
    dims = (findall(==(-1), s)...,)
    m = roto_reflection_matrix(p, s)
    t = map(t -> m * t * m', t)
end

create_loss(project) = function loss(net, ps, st, (x, y))
    ps = project(ps)
    yhat = net(x, ps, st) |> first
    # l = MSELoss()(yhat, y)
    l = sum(abs2, yhat - y) / sum(abs2, y)
    l, st, (;)
end

function train(; loss, setup, dataloader, nepoch, learning_rate, net_stuff)
    (; D, backend) = setup
    g = Grid{D}(; setup.l, n = setup.n_les, backend)
    (; net, ps, st) = net_stuff
    ps = deepcopy(ps)
    device = adapt(backend)
    opt = AdamW(learning_rate)
    train_state = Training.TrainState(net, ps, st, opt)
    for iepoch = 1:nepoch, (ibatch, batch) in enumerate(dataloader)
        x, y = batch |> device
        loss(net, ps, st, (x, y)) |> display
        _, l, _, train_state =
            Training.single_train_step!(AutoZygote(), loss, (x, y), train_state)
        ibatch % 1 == 0 && @info "iepoch = $iepoch, ibatch = $ibatch, loss = $l"
    end
    ps = train_state.parameters
    st = train_state.states
    ps, st
end

function fullchain(setup, net, project, ps, st)
    (; D) = setup
    ps = project(ps)
    model(x) = net(x, ps, st) |> first
end

function les!(du, u, grid, cache; model, visc)
    D = dim(grid)
    (; plan, σ, vi_vj, v, G) = cache
    stress!(σ, vi_vj, v, u, plan, visc, grid)
    apply!(vectorgradient!, grid, (G, u, grid))
    x = map(G) do G
        apply!(twothirds!, grid, (G, grid))
        plan \ G
    end
    x = stack(x)
    x .*= grid.n^D # FFT factor
    x = reshape(x, :, D^2, 1)
    y = model(x)
    for (i, σ) in enumerate(σ)
        yi = view(y, :, i, 1)
        yi = reshape(yi, size(vi_vj))
        mul!(du.x, plan, yi) # Use du as temp storage
        @. σ += du.x / grid.n^D # With FFT factor
    end
    apply!(tensordivergence!, grid, (du, σ, grid))
end

function inference_post(; setup, models, cfl = 0.35, tstop = 1e-1, Δ)
    (; visc, D, n_dns, n_les, backend) = setup
    t = 0.0
    g_dns = Grid{D}(; setup.l, n = n_dns, backend)
    g_les = Grid{D}(; setup.l, n = n_les, backend)
    u = randomfield(g_dns; rng = Xoshiro(123), kpeak = 5)
    c_dns = getcache(g_dns)
    c_les = (; getcache(g_les)..., G = tensorfield_nonsym(g_les))
    fu = vectorfield(g_les)
    for (fu, u) in zip(fu, u)
        apply!(cutoff!, g_les, (fu, u, g_les, g_dns))
        apply!(gaussianfilter!, g_les, (fu, Δ, g_les))
    end
    u_nomo = map(copy, fu)
    u_models = map(m -> map(copy, fu), models)
    t = zero(tstop)
    i = 0
    while t < tstop
        Δt = cfl * propose_timestep(u, g_dns, visc, c_dns)
        Δt = min(Δt, tstop - t)
        t += Δt
        wray3!(convectiondiffusion!, u, Δt, g_dns, c_dns; visc)
        wray3!(convectiondiffusion!, u_nomo, Δt, g_les, c_les; visc)
        for (u, model) in zip(u_models, models)
            wray3!(les!, u, Δt, g_les, c_les; model, visc)
        end
        if i % 50 == 0
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
    end
    for (fu, u) in zip(fu, u)
        apply!(cutoff!, g_les, (fu, u, g_les, g_dns))
        apply!(gaussianfilter!, g_les, (fu, Δ, g_les))
    end
    u_models = (; nomo = u_nomo, u_models...)
    for u in (fu, u_models...)
        for u in u
            apply!(twothirds!, g_les, (u, g_les))
        end
    end
    fu = stack(fu)
    u_models = map(stack, u_models)
    for (key, u) in pairs(u_models)
        err = norm(u - fu) / norm(fu)
        @show key => err
    end
    nothing
end

nbasis(::Grid{2}) = 4
nbasis(::Grid{3}) = 11
ninvariant(::Grid{2}) = 2
ninvariant(::Grid{3}) = 2

"Compute deviatoric part of a tensor."
@inline deviator(σ) = σ - tr(σ) / 3 * one(σ)

@kernel function tb_kernel!(invariants, basis, grads, g::Grid{2})
    I = @index(Global, Cartesian)
    Gxx, Gyx, Gxy, Gyy = grads.xx[I], grads.yx[I], grads.xy[I], grads.yy[I]
    G = @SMatrix [Gxx Gxy; Gyx Gyy]
    S = (G + G') / 2
    R = (G - G') / 2
    i = tr(S * S), tr(R * R)
    b = S, S * R - R * S, deviator(S * S), deviator(R * R)
    for iinv in Base.OneTo(2)
        invariants[I, iinv] = i[iinv]
    end
    for ibas in Base.OneTo(4), y in Base.OneTo(2), x in Base.OneTo(2)
        basis[I, x, y, ibas] = b[ibas][x, y]
    end
end

@kernel function tb_kernel!(invariants, basis, grads, g::Grid{3})
    I = @index(Global, Cartesian)
    Gxx, Gxy, Gxz = grads.xx[I], grads.xy[I], grads.xz[I]
    Gyx, Gyy, Gyz = grads.yx[I], grads.yy[I], grads.yz[I]
    Gzx, Gzy, Gzz = grads.zx[I], grads.zy[I], grads.zz[I]
    G = @SMatrix [Gxx Gxy Gxz; Gyx Gyy Gyz; Gzx Gzy Gzz]
    S = (G + G') / 2
    R = (G - G') / 2
    i = tr(S * S), tr(R * R), tr(S * S * S), tr(S * R * R), tr(S * S * R * R)
    b = (
        S,
        S * R - R * S,
        deviator(S * S),
        deviator(R * R),
        R * S * S - S * S * R,
        deviator(S * R * R + R * R * S),
        R * S * R * R - R * R * S * R,
        S * R * S * S - S * S * R * S,
        deviator(R * R * S * S + S * S * R * R),
        R * S * S * R * R - R * R * S * S * R,
    )
    for iinv in Base.OneTo(5)
        invariants[I, iinv] = i[iinv]
    end
    for ibas in Base.OneTo(11), y in Base.OneTo(3), x in Base.OneTo(3)
        basis[I, x, y, ibas] = b[ibas][x, y]
    end
end

function build_tensorbasis(u, g)
    D = dim(g)
    nb, ni = nbasis(g), ninvariant(g)
    G = tensorfield_nonsym(g)
    GG = spacetensorfield_nonsym(g)
    basis = KernelAbstractions.zeros(g.backend, typeof(g.l), space_ndrange(g)..., D, D, nb)
    invariants = KernelAbstractions.zeros(g.backend, typeof(g.l), space_ndrange(g)..., ni)
    apply!(vectorgradient!, g, (G, u, g))
    plan = plan_rfft(GG.xx)
    for (GG, G) in zip(GG, G)
        apply!(twothirds!, g, (G, g))
        ldiv!(GG, plan, G) # Inverse RFFT
        GG .*= g.n^D # FFT factor
    end
    apply!(tb_kernel!, g, (invariants, basis, GG, g); ndrange = space_ndrange(g))
    invariants, basis
end

tbnn(net, ps, st, g) = function model(u)
    D = dim(g)

    # Compute invariants and basis tensors
    invariants, basis = build_tensorbasis(u, g)

    # Compute coefficients
    invariants = reshape(invariants, size(invariants)..., 1) # One sample
    w = net(invariants, ps, st) |> first

    # Basis contraction
    b = reshape(basis, :, D^2, nbasis(g))
    w = reshape(w, :, 1, nbasis(g))
    wb = @. w * b
    m = sum(wb; dims = 3)
    m = reshape(m, :, D^2)

    # Symmetrize tensor
    m = if D == 2
        hcat(m[:, 1], m[:, 4], (m[:, 2] + m[:, 3]) / 2)
    elseif D == 3
        hcat(
            m[:, 1],
            m[:, 5],
            m[:, 9],
            (m[:, 2] + m[:, 4]) / 2,
            (m[:, 6] + m[:, 8]) / 2,
            (m[:, 3] + m[:, 7]) / 2,
        )
    end

    reshape(m, space_ndrange(g)..., tensordim(g))
end

function create_dataloader_tbnn(setup, data; batchsize)
    (; D) = setup
    g = Grid{D}(; setup.l, n = setup.n_les, setup.backend)
    # G = tensorfield_nonsym(g)
    u = vectorfield(g)
    τ = scalarfield(g)
    # GG = spacetensorfield_nonsym(g)
    ττ = spacetensorfield(g)
    plan = plan_rfft(ττ.xx)
    snaps = map(zip(data...)) do (ucpu, τcpu)
        map(copyto!, u, ucpu)
        # apply!(vectorgradient!, g, (G, u, g))
        # for (GG, G) in zip(GG, G)
        #     apply!(twothirds!, g, (G, g))
        #     ldiv!(GG, plan, G) # Inverse RFFT
        #     GG .*= g.n^D # FFT factor
        # end
        for (ττ, τcpu) in zip(ττ, τcpu)
            copyto!(τ, τcpu)
            apply!(twothirds!, g, (τ, g))
            ldiv!(ττ, plan, τ) # Inverse RFFT
            ττ .*= g.n^D # FFT factor
        end
        i, b = build_tensorbasis(u, g)
        i = i |> cpu_device()
        b = reshape(b, space_ndrange(g)..., :) |> cpu_device()
        x = cat(i, b; dims = D + 1)
        y = reshape(stack(ττ), space_ndrange(g)..., tensordim(g)) |> cpu_device()
        x, y
    end
    x = stack(first, snaps)
    y = stack(last, snaps)
    DataLoader((x, y); batchsize, shuffle = true, partial = false)
end

create_loss_tbnn(g) = function loss(net, ps, st, (x, y))
    D = dim(g)
    ni = ninvariant(g)
    nb = nbasis(g)

    # Destructure invariants and basis
    i = selectdim(x, D + 1, 1:ni)
    b = selectdim(x, D + 1, ni+1:size(x, D + 1))

    # Compute coefficients
    w = net(i, ps, st) |> first

    # Basis contraction
    w = reshape(w, space_ndrange(g)..., 1, nb, :)
    b = reshape(b, space_ndrange(g)..., D^2, nb, :)
    wb = @. w * b
    m = sum(wb; dims = D + 2)
    m = reshape(m, space_ndrange(g)..., D^2, :)

    # Symmetrize tensor
    m = if D == 2
        cat(
            selectdim(m, D + 1, 1:1),
            selectdim(m, D + 1, 4:4),
            (selectdim(m, D + 1, 2:2) + selectdim(m, D + 1, 3:3)) / 2;
            dims = D + 1,
        )
    elseif D == 3
        cat(
            selectdim(m, :, 1:1),
            selectdim(m, :, 5:5),
            selectdim(m, :, 9:9),
            (selectdim(m, :, 2:2) + selectdim(m, :, 4:4)) / 2,
            (selectdim(m, :, 6:6) + selectdim(m, :, 8:8)) / 2,
            (selectdim(m, :, 3:3) + selectdim(m, :, 7:7)) / 2;
            dims = D + 1,
        )
    end

    # l = MSELoss()(m, y)
    l = sum(abs2, m - y) / sum(abs2, y)
    l, st, (;)
end

export tensorfield_to_smatrix, smatrix_to_tensorfield
export transform_scalar, transform_vector, transform_tensor
export dns_aid, create_dns, create_data, create_dataloader, train, fullchain, inference_post
export sfs!, cutoff!, gaussianfilter!
export tbnn, ninvariant, nbasis, create_dataloader_tbnn, create_loss_tbnn

end
