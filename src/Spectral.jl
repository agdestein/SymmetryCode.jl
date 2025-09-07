module Spectral

using Adapt
using KernelAbstractions
using LinearAlgebra
using Lux
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
    foreach(i -> apply!(Spectral.cutoff!, gbar, (ubar[i], u[i], gbar, g)), 1:D)
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
        foreach(i -> apply!(Spectral.cutoff!, gbar, (ubar[i], u[i], gbar, g)), 1:D)
        stress!(σf, cbar.vi_vj, cbar.v, ubar, cbar.plan, visc, gbar)
        stress!(cbar.σ, cbar.vi_vj, cbar.v, v, cbar.plan, visc, gbar)
        foreach(
            i -> apply!(Spectral.cutoff!, gbar, (fσ[i], c.σ[i], gbar, g)),
            1:tensordim(g),
        )
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
    foreach(i -> apply!(Spectral.cutoff!, gbar, (ubar[i], u[i], gbar, g)), 1:D)
    sum(i -> sum(abs2, v[i] - ubar[i]) / sum(abs2, ubar[i]), 1:D)
end

function create_data(setup)
    (; visc, D, n_dns, n_les, backend) = setup
    t = 0.0
    cfl = 0.35
    tstop = 2e-1
    g_dns = Grid{D}(; l = 1.0, n = n_dns, backend)
    g_les = Grid{D}(; l = 1.0, n = n_les, backend)
    u = randomfield(g_dns; rng = Xoshiro(0), kpeak = 5)
    c_dns = getcache(g_dns)
    i = 0
    while t < tstop
        i += 1
        Δt = cfl * propose_timestep(u, c_dns, visc, g_dns)
        Δt = min(Δt, tstop - t)
        t += Δt
        Seneca.wray3!(u, c_dns, Δt, visc, g_dns)
        if i % 50 == 0
            energy = Seneca.energy(u)
            @info join(
                [
                    "t = $(round(t; sigdigits = 4))",
                    "Δt = $(round(Δt; sigdigits = 4))",
                    # "umax = $(round(maximum(u -> maximum(abs, u), u); sigdigits = 4))",
                    "energy = $(round(energy; sigdigits = 4))",
                ],
                ",\t",
            )
        end
    end
    c_les = getcache(g_les)
    fu = vectorfield(g_les)
    σfu = tensorfield(g_les)
    fσu = tensorfield(g_les)
    stress!(c_dns.σ, c_dns.vi_vj, c_dns.v, u, c_dns.plan, visc, g_dns)
    foreach(i -> apply!(Spectral.cutoff!, g_les, (fu[i], u[i], g_les, g_dns)), 1:D)
    foreach(
        i -> apply!(Spectral.cutoff!, g_les, (fσu[i], c_dns.σ[i], g_les, g_dns)),
        1:tensordim(g_dns),
    )
    stress!(σfu, c_les.vi_vj, c_les.v, fu, c_les.plan, visc, g_les)
    foreach(i -> (fσu[i] .-= σfu[i]), 1:tensordim(g_dns))
    fu, fσu
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

function train(; setup, dataloader, nepoch, learning_rate, net_stuff)
    (; D, backend) = setup
    g = Grid{D}(; setup.l, n = setup.n_les, backend)
    (; project, net, ps, st) = net_stuff
    device = adapt(backend)
    opt = Adam(learning_rate)
    train_state = Training.TrainState(net, ps, st, opt)
    function loss(model, ps, st, (x, y))
        ps = project(ps)
        yhat = net(x, ps, st) |> first
        ysym = if D == 2
            xx = selectdim(yhat, 2, 1:1)
            xy = (selectdim(yhat, 2, 2:2) + selectdim(yhat, 2, 3:3)) / 2
            yy = selectdim(yhat, 2, 4:4)
            hcat(xx, yy, xy)
        else
            xx = selectdim(yhat, 2, 1:1)
            yy = selectdim(yhat, 2, 5:5)
            zz = selectdim(yhat, 2, 9:9)
            xy = (selectdim(yhat, 2, 2:2) + selectdim(yhat, 2, 4:4)) / 2
            yz = (selectdim(yhat, 2, 6:6) + selectdim(yhat, 2, 8:8)) / 2
            zx = (selectdim(yhat, 2, 3:3) + selectdim(yhat, 2, 7:7)) / 2
            hcat(xx, yy, zz, xy, yz, zx)
        end
        l = MSELoss()(ysym, y)
        l, st, (;)
    end
    for iepoch = 1:nepoch, (ibatch, batch) in enumerate(dataloader)
        x, y = batch |> device
        # loss(net, ps, st, (x, y)) |> display
        _, l, _, train_state =
            Training.single_train_step!(AutoZygote(), loss, (x, y), train_state)
        ibatch % 1 == 0 && @info "iepoch = $iepoch, ibatch = $ibatch, loss = $l"
    end
    ps = train_state.parameters
    st = train_state.states
    net, ps, st
end

export tensorfield_to_smatrix, smatrix_to_tensorfield
export transform_scalar, transform_vector, transform_tensor

export dns_aid, create_data, train

end
