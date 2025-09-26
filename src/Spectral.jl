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
    kΔ = pi / Δ * 2
    kx, ky, kz = wavenumbers(g, I)
    k2 = kx^2 + ky^2 + kz^2
    w = exp(-k2 / kΔ^2 / 2)
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

function create_data(setup; t_warmup = 0.5, cfl = 0.35, nstep = 200, nsubstep = 10, kpeak, Δ, rng)
    (; visc, D, n_dns, n_les, backend) = setup
    g_dns = Grid{D}(; setup.l, n = n_dns, backend)
    g_les = Grid{D}(; setup.l, n = n_les, backend)
    u = randomfield(g_dns; rng, kpeak)
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

export sfs
function sfs(u, g_dns, g_les, Δ)
    c_dns = getcache(g_dns)
    c_les = getcache(g_les)
    ubar = vectorfield(g_les)
    σbar1 = tensorfield(g_les)
    σbar2 = tensorfield(g_les)
    τ = tensorfield(g_les)
    sfs!(τ, σbar1, σbar2, ubar, u, c_dns, c_les, g_dns, g_les, Δ)
    τ
end
function sfs!(τ, σbar1, σbar2, ubar, u, c_dns, c_les, g_dns, g_les, Δ)
    D = dim(g_dns)
    nonlinearity!(c_dns.σ, c_dns.vi_vj, c_dns.v, u, c_dns.plan, g_dns)
    for (ubar, u) in zip(ubar, u)
        apply!(cutoff!, g_les, (ubar, u, g_les, g_dns))
        apply!(gaussianfilter!, g_les, (ubar, Δ, g_les))
    end
    for (σbar1, σ) in zip(σbar1, c_dns.σ)
        apply!(cutoff!, g_les, (σbar1, σ, g_les, g_dns))
        apply!(gaussianfilter!, g_les, (σbar1, Δ, g_les))
    end
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
        x = stack(GG) |> cpu_device()
        y = stack(ττ) |> cpu_device()
        x, y
    end
    x = stack(first, snaps)
    y = stack(last, snaps)
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
    ru = if D == 2
        (; x = getindex.(ru_sa, 1), y = getindex.(ru_sa, 2))
    elseif D == 3
        (; x = getindex.(ru_sa, 1), y = getindex.(ru_sa, 2), z = getindex.(ru_sa, 3))
    end
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
    b_valid = first(dataloader) |> device
    ps_best = deepcopy(ps)
    l_best = Inf
    for iepoch = 1:nepoch, (ibatch, batch) in enumerate(dataloader)
        x, y = batch |> device
        # loss(net, ps, st, (x, y)); error()
        _, l, _, train_state =
            Training.single_train_step!(AutoZygote(), loss, (x, y), train_state)
        if ibatch % 1 == 0
            l_valid = loss(net, ps, st, b_valid) |> first
            @info join(
                [
                    "iepoch = $iepoch",
                    "ibatch = $ibatch",
                    "loss (valid) = $(round(l_valid; sigdigits = 4))",
                    "loss (train) = $(round(l; sigdigits = 4))",
                ],
                ",\t",
            )
            if l_valid < l_best
                l_best = l_valid
                ps_best = deepcopy(train_state.parameters)
            end
        end
    end
    ps = ps_best
    st = train_state.states
    ps, st
end

function fullchain(setup, net, project, ps, st)
    (; D) = setup
    ps = project(ps)
    function model(x)
        x = stack(x)
        s = size(x)
        x = reshape(x, s..., 1) # Add sample dimension
        y = net(x, ps, st) |> first
        reshape(y, s[1:D]..., :)
    end
end

function les!(du, u, grid, cache; model, visc)
    D = dim(grid)
    (; plan, σ, vi_vj, v, G) = cache
    nx = space_ndrange(grid)
    stress!(σ, vi_vj, v, u, plan, visc, grid)
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
    for (i, σ) in enumerate(σ)
        copyto!(vi_vj, selectdim(y, D + 1, i))
        mul!(du.x, plan, vi_vj) # Use du as temp storage
        @. σ += du.x / grid.n^D # With FFT factor
    end
    apply!(tensordivergence!, grid, (du, σ, grid))
end

function inference_post(; ustart, setup, models, cfl = 0.35, tstop = 1e-1, Δ)
    (; visc, D, n_dns, n_les, backend) = setup
    t = 0.0
    g_dns = Grid{D}(; setup.l, n = n_dns, backend)
    g_les = Grid{D}(; setup.l, n = n_les, backend)
    u = map(copy, ustart)
    c_dns = getcache(g_dns)
    c_les = (; getcache(g_les)..., G = tensorfield_nonsym(g_les))
    fu = vectorfield(g_les)
    for (fu, u) in zip(fu, u)
        apply!(cutoff!, g_les, (fu, u, g_les, g_dns))
        apply!(gaussianfilter!, g_les, (fu, Δ, g_les))
    end
    u_models = map(m -> map(copy, fu), models)
    t = zero(tstop)
    i = 0
    while t < tstop
        Δt = cfl * propose_timestep(u, g_dns, visc, c_dns)
        Δt = min(Δt, tstop - t)
        t += Δt
        wray3!(convectiondiffusion!, u, Δt, g_dns, c_dns; visc)
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
    u_les = (; ref = fu, u_models...)
    for u in u_models
        foreach(u -> apply!(twothirds!, g_les, (u, g_les)), u)
    end
    u, u_les
end

function get_errors(setup, u_les)
    u_les = map(stack, u_les)
    k = keys(u_les)
    k_les = filter(!=(:ref), k)
    u_ref = u_les.ref
    errs = map(k_les) do key
        u = u_les[key]
        err = norm(u - u_ref) / norm(u_ref)
        println(key => round(err; sigdigits = 4))
        key => err
    end
end

@inline nbasis(::Grid{2}) = 3 # Number of entries below
@inline getbasis(::Grid{2}, S, R) = (
    one(S),
    S,
    S * R - R * S,
    # deviator(S * S),
    # deviator(R * R),
)

@inline nbasis(::Grid{3}) = 10
@inline getbasis(::Grid{3}, S, R) = (
    # one(S),
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

@inline ninvariant(::Grid{2}) = 2
@inline getinvariants(::Grid{2}, S, R) = tr(S * S), tr(R * R)

@inline ninvariant(::Grid{3}) = 5
@inline getinvariants(::Grid{3}, S, R) =
    tr(S * S), tr(R * R), tr(S * S * S), tr(S * R * R), tr(S * S * R * R)

"Compute deviatoric part of a tensor."
@inline deviator(σ) = σ - tr(σ) / 3 * one(σ)

@kernel function tb_kernel!(invariants, basis, grads, g::Grid{2})
    nb, ni = nbasis(g), ninvariant(g)
    I = @index(Global, Cartesian)
    Gxx, Gyx, Gxy, Gyy = grads.xx[I], grads.yx[I], grads.xy[I], grads.yy[I]
    G = @SMatrix [Gxx Gxy; Gyx Gyy]
    G2 = sum(abs2, G)
    G = G / (sqrt(G2) + eps(eltype(G))) # Normalize gradient
    S, R = (G + G') / 2, (G - G') / 2
    i, b = getinvariants(g, S, R), getbasis(g, S, R)
    for iinv in Base.OneTo(ni)
        invariants[I, iinv] = i[iinv]
    end
    for ibas in Base.OneTo(nb)
        # Convert 2x2 tensor b to symmetric tensor [xx, yy, xy]
        # Also premultiply by |G|^2, since the output tensor is
        # |G|^2 * coeffs * basis
        basis[I, 1, ibas] = b[ibas][1, 1] * sqrt(G2)
        basis[I, 2, ibas] = b[ibas][2, 2] * sqrt(G2)
        basis[I, 3, ibas] = b[ibas][1, 2] * sqrt(G2)
    end
end

@kernel function tb_kernel!(invariants, basis, grads, g::Grid{3})
    ni, nb = ninvariant(g), nbasis(g)
    I = @index(Global, Cartesian)
    Gxx, Gxy, Gxz = grads.xx[I], grads.xy[I], grads.xz[I]
    Gyx, Gyy, Gyz = grads.yx[I], grads.yy[I], grads.yz[I]
    Gzx, Gzy, Gzz = grads.zx[I], grads.zy[I], grads.zz[I]
    G = @SMatrix [Gxx Gxy Gxz; Gyx Gyy Gyz; Gzx Gzy Gzz]
    G2 = sum(abs2, G)
    G = G / (sqrt(G2) + eps(eltype(G))) # Normalize gradient
    S, R = (G + G') / 2, (G - G') / 2
    i, b = getinvariants(g, S, R), getbasis(g, S, R)
    for iinv in Base.OneTo(ni)
        invariants[I, iinv] = i[iinv]
    end
    for ibas in Base.OneTo(nb)
        # Convert 3x3 tensor b to flattened symmetric tensor [xx, yy, zz, xy, yz, zx]
        basis[I, 1, ibas] = b[ibas][1, 1] * sqrt(G2)
        basis[I, 2, ibas] = b[ibas][2, 2] * sqrt(G2)
        basis[I, 3, ibas] = b[ibas][3, 3] * sqrt(G2)
        basis[I, 4, ibas] = b[ibas][1, 2] * sqrt(G2)
        basis[I, 5, ibas] = b[ibas][2, 3] * sqrt(G2)
        basis[I, 6, ibas] = b[ibas][3, 1] * sqrt(G2)
    end
end

function build_tensorbasis(grad, g)
    T = typeof(g.l)
    nx, nb, ni, nt = space_ndrange(g), nbasis(g), ninvariant(g), tensordim(g)
    basis = KernelAbstractions.zeros(g.backend, T, nx..., nt, nb)
    invariants = KernelAbstractions.zeros(g.backend, T, nx..., ni)
    apply!(tb_kernel!, g, (invariants, basis, grad, g); ndrange = nx)
    invariants, basis
end

function getgradient(u, g)
    D = dim(g)
    G = tensorfield_nonsym(g)
    GG = spacetensorfield_nonsym(g)
    apply!(vectorgradient!, g, (G, u, g))
    plan = plan_rfft(GG.xx)
    for (GG, G) in zip(GG, G)
        apply!(twothirds!, g, (G, g))
        ldiv!(GG, plan, G) # Inverse RFFT
        GG .*= g.n^D # FFT factor
    end
    GG
end

tbnn(net, ps, st, g) = function model(G)
    D = dim(g)
    nx = space_ndrange(g)
    nt = tensordim(g)
    nb = nbasis(g)

    # Compute invariants and basis tensors
    invariants, basis = build_tensorbasis(G, g)

    # Compute coefficients
    invariants = reshape(invariants, size(invariants)..., 1) # One sample
    w = net(invariants, ps, st) |> first

    # Basis contraction
    b = reshape(basis, :, nt, nb)
    w = reshape(w, :, 1, nb)
    wb = @. w * b
    m = sum(wb; dims = 3)
    reshape(m, nx..., nt)
end

function create_dataloader_tbnn(setup, data; batchsize, rng)
    (; D) = setup
    g = Grid{D}(; setup.l, n = setup.n_les, setup.backend)
    T = typeof(g.l)
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
        i, b = build_tensorbasis(G, g)
        i = i |> cpu_device()
        b = reshape(b, nx..., :) |> cpu_device()
        x = cat(i, b; dims = D + 1)
        y = reshape(stack(ττ), nx..., tensordim(g)) |> cpu_device()
        x, y
    end
    x = stack(first, snaps)
    y = stack(last, snaps)
    DataLoader((x, y); batchsize, shuffle = true, partial = false, rng)
end

create_loss_tbnn(g) = function loss(net, ps, st, (x, y))
    D = dim(g)
    nx = space_ndrange(g)
    nt = tensordim(g)
    ni = ninvariant(g)
    nb = nbasis(g)

    # Destructure invariants and basis
    i = selectdim(x, D + 1, 1:ni)
    b = selectdim(x, D + 1, ni+1:size(x, D + 1))

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
    nx = space_ndrange(g)
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
function test_equivariance_post(;
    ustart,
    setup,
    grid,
    model,
    groupindex,
    rng,
    cfl = 0.35,
    tstop = 1e-2,
)
    T, D = typeof(setup.l), setup.D

    # Group element
    (; elements, permutations, signs, mats) = group_stuff(setup.D)
    ip, is = elements[groupindex]
    p, s = permutations[ip], signs[is]
    # mats[groupindex] |> display
    # mats[groupindex] |> det |> display
    # error()

    # Initial conditions + rotated copy
    u = map(copy, ustart)
    # u = randomfield(grid; rng = Xoshiro(123), kpeak = 5)
    # foreach(u -> randn!(Xoshiro(123), u), u)
    # apply!(project!, grid, (u, grid))
    # foreach(u -> apply!(twothirds!, grid, (u, grid)), u)
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
        if i % 10 == 0
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

export vectorfield_to_svector,
    svector_to_vectorfield, tensorfield_to_smatrix, smatrix_to_tensorfield
export transform_scalar, transform_vector, transform_tensor
export getgradient
export dns_aid, create_dns, create_data, create_dataloader, train, fullchain, inference_post
export sfs!, cutoff!, gaussianfilter!
export tbnn, ninvariant, nbasis, create_dataloader_tbnn, create_loss, create_loss_tbnn
export get_errors

end
