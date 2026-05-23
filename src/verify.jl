# Verification scripts. Not part of the simulation pipeline — these are
# run by hand from the REPL to check that the equivariant network actually
# respects the group and that the DNS-aided LES coupling is consistent.

function test_equivariant_dense(D)
    rng = Xoshiro(0)
    f = f64
    (; elements, mats) = group_stuff(D)
    (; r_lift, r_sink, r_mid) = get_weight_projectors(D)
    nelement = length(elements)
    proj_lift = Dense(nelement * D^2 => nelement * D^2; use_bias = false)
    proj_sink = Dense(D^2 * nelement => D^2 * nelement; use_bias = false)
    proj_mid = Dense(nelement^2 => nelement^2; use_bias = false)
    proj_lift_ps, _ = Lux.setup(rng, proj_lift) |> f
    proj_lift_ps.weight .= r_lift
    proj_sink_ps, _ = Lux.setup(rng, proj_sink) |> f
    proj_sink_ps.weight .= r_sink
    proj_mid_ps, _ = Lux.setup(rng, proj_mid) |> f
    proj_mid_ps.weight .= r_mid
    function project_lift(ps)
        w, b = ps.weight, ps.bias
        s_out, s_in = size(w)
        c_out, c_in = div(s_out, nelement), div(s_in, D^2)
        w = reshape(w, nelement, c_out, D^2, c_in)
        w = permutedims(w, (1, 3, 2, 4))
        w = reshape(w, nelement * D^2, :)
        w = proj_lift(w, proj_lift_ps, (;)) |> first
        w = reshape(w, nelement, D^2, c_out, c_in)
        w = permutedims(w, (1, 3, 2, 4))
        weight = reshape(w, s_out, s_in)
        bias = fill(b, s_out)
        return (; weight, bias)
    end
    function project_mid(ps)
        w, b = ps.weight, ps.bias
        s_out, s_in = size(w)
        c_out, c_in = div(s_out, nelement), div(s_in, nelement)
        w = reshape(w, nelement, c_out, nelement, c_in)
        w = permutedims(w, (1, 3, 2, 4))
        w = reshape(w, nelement * nelement, :)
        w = proj_mid(w, proj_mid_ps, (;)) |> first
        w = reshape(w, nelement, nelement, c_out, c_in)
        w = permutedims(w, (1, 3, 2, 4))
        weight = reshape(w, s_out, s_in)
        bias = fill(b, s_out)
        return (; weight, bias)
    end
    function project_sink(ps)
        w = ps.weight
        s_out, s_in = size(w)
        c_out, c_in = div(s_out, D^2), div(s_in, nelement)
        w = reshape(w, D^2, c_out, nelement, c_in)
        w = permutedims(w, (1, 3, 2, 4))
        w = reshape(w, D^2 * nelement, :)
        w = proj_sink(w, proj_sink_ps, (;)) |> first
        w = reshape(w, D^2, nelement, c_out, c_in)
        w = permutedims(w, (1, 3, 2, 4))
        weight = reshape(w, s_out, s_in)
        return (; weight)
    end
    project(ps) = (;
        lift = project_lift(ps.lift),
        mid_1 = project_mid(ps.mid_1),
        mid_2 = project_mid(ps.mid_2),
        mid_3 = project_mid(ps.mid_3),
        sink = project_sink(ps.sink),
    )
    net = Chain(;
        lift = Dense(D^2 => nelement * 10, gelu),
        mid_1 = Dense(nelement * 10 => nelement * 10, gelu),
        mid_2 = Dense(nelement * 10 => nelement * 20, gelu),
        mid_3 = Dense(nelement * 20 => nelement * 20, gelu),
        sink = Dense(nelement * 20 => D^2),
    )
    net |> display
    ps, st = Lux.setup(rng, net) |> f
    T = eltype(ps.lift.weight)
    a = T(0)
    ps = (;
        lift = (; ps.lift.weight, bias = a * randn(rng, T)),
        mid_1 = (; ps.mid_1.weight, bias = a * randn(rng, T)),
        mid_2 = (; ps.mid_2.weight, bias = a * randn(rng, T)),
        mid_3 = (; ps.mid_3.weight, bias = a * randn(rng, T)),
        sink = (; ps.sink.weight),
    )
    ps = project(ps)
    i = 7
    mat = mats[i]
    x = @SMatrix(randn(D, D)) |> f
    rx = mat * x * mat'
    nx =
        net(reshape(Array(x), D^2, 1), ps, st)[1] |>
        x -> reshape(x, D, D) |> SMatrix{D, D, eltype(x), D^2}
    nrx =
        net(reshape(Array(rx), D^2, 1), ps, st)[1] |>
        x -> reshape(x, D, D) |> SMatrix{D, D, eltype(x), D^2}
    rnx = mat * nx * mat'
    nrx - rnx |> display
    return nothing
end

function test_equivariant_conv(D)
    rng = Xoshiro(0)
    f = f64
    nten = D^2
    (; elements, mats) = group_stuff(D)
    (; r_lift, r_sink, r_mid) = get_weight_projectors(D)
    nreg = length(elements)
    proj_lift = Dense(nten * nreg => nten * nreg; use_bias = false)
    proj_sink = Dense(nreg * nten => nreg * nten; use_bias = false)
    proj_mid = Dense(nreg^2 => nreg^2; use_bias = false)
    proj_lift_ps, _ = Lux.setup(rng, proj_lift) |> f
    copyto!(proj_lift_ps.weight, r_lift)
    proj_sink_ps, _ = Lux.setup(rng, proj_sink) |> f
    copyto!(proj_sink_ps.weight, r_sink)
    proj_mid_ps, _ = Lux.setup(rng, proj_mid) |> f
    copyto!(proj_mid_ps.weight, r_mid)
    function project_lift(ps)
        w, b = ps.weight, ps.bias
        _, s_in, s_out = size(w)
        c_in, c_out = div(s_in, nten), div(s_out, nreg)
        w = reshape(w, nten, c_in, nreg, c_out)
        w = permutedims(w, (3, 1, 2, 4))
        w = reshape(w, nten * nreg, :)
        w = proj_lift(w, proj_lift_ps, (;)) |> first
        w = reshape(w, nreg, nten, c_in, c_out)
        w = permutedims(w, (2, 3, 1, 4))
        weight = reshape(w, 1, s_in, s_out)
        bias = fill(b, s_out)
        return (; weight, bias)
    end
    function project_mid(ps)
        w, b = ps.weight, ps.bias
        _, s_in, s_out = size(w)
        c_in, c_out = div(s_in, nreg), div(s_out, nreg)
        w = reshape(w, nreg, c_in, nreg, c_out)
        w = permutedims(w, (3, 1, 2, 4))
        w = reshape(w, nreg * nreg, :)
        w = proj_mid(w, proj_mid_ps, (;)) |> first
        w = reshape(w, nreg, nreg, c_in, c_out)
        w = permutedims(w, (2, 3, 1, 4))
        weight = reshape(w, 1, s_in, s_out)
        bias = fill(b, s_out)
        return (; weight, bias)
    end
    function project_sink(ps)
        w = ps.weight
        _, s_in, s_out = size(w)
        c_in, c_out = div(s_in, nreg), div(s_out, nten)
        w = reshape(w, nreg, c_in, nten, c_out)
        w = permutedims(w, (3, 1, 2, 4))
        w = reshape(w, nreg * nten, :)
        w = proj_sink(w, proj_sink_ps, (;)) |> first
        w = reshape(w, nten, nreg, c_in, c_out)
        w = permutedims(w, (2, 3, 1, 4))
        weight = reshape(w, 1, s_in, s_out)
        return (; weight)
    end
    project(ps) = (;
        lift = project_lift(ps.lift),
        mid_1 = project_mid(ps.mid_1),
        mid_2 = project_mid(ps.mid_2),
        mid_3 = project_mid(ps.mid_3),
        sink = project_sink(ps.sink),
    )
    net = Chain(;
        lift = Conv((1,), nten => nreg * 10, gelu),
        mid_1 = Conv((1,), nreg * 10 => nreg * 10, gelu),
        mid_2 = Conv((1,), nreg * 10 => nreg * 20, gelu),
        mid_3 = Conv((1,), nreg * 20 => nreg * 20, gelu),
        sink = Conv((1,), nreg * 20 => nten),
    )
    net |> display
    ps, st = Lux.setup(rng, net) |> f
    T = eltype(ps.lift.weight)
    a = T(0.1)
    ps = (;
        lift = (; ps.lift.weight, bias = a * randn(rng, T)),
        mid_1 = (; ps.mid_1.weight, bias = a * randn(rng, T)),
        mid_2 = (; ps.mid_2.weight, bias = a * randn(rng, T)),
        mid_3 = (; ps.mid_3.weight, bias = a * randn(rng, T)),
        sink = (; ps.sink.weight),
    )
    ps = project(ps)
    i = 7
    mat = mats[i]
    x = @SMatrix(randn(D, D)) |> f
    rx = mat * x * mat'
    nx =
        net(reshape(Array(x), 1, nten, 1), ps, st)[1] |>
        x -> reshape(x, D, D) |> SMatrix{D, D, eltype(x), nten}
    nrx =
        net(reshape(Array(rx), 1, nten, 1), ps, st)[1] |>
        x -> reshape(x, D, D) |> SMatrix{D, D, eltype(x), nten}
    rnx = mat * nx * mat'
    nrx - rnx |> display
    return nothing
end

function test_equivariant_conv_sparse(D)
    rng = Xoshiro(0)
    T, f = Float64, f64
    nten = D^2
    (; elements, mats) = group_stuff(D)
    (; r_lift, r_sink, r_mid) = get_weight_projectors(D)
    nreg = length(elements)
    e_lift = eigen(r_lift / nreg; sortby = -).vectors[:, 1:nten]
    e_mid = eigen(r_mid / nreg; sortby = -).vectors[:, 1:nreg]
    e_sink = eigen(r_sink / nreg; sortby = -).vectors[:, 1:nten]
    proj_lift = Dense(nten => nten * nreg; use_bias = false)
    proj_sink = Dense(nten => nreg * nten; use_bias = false)
    proj_mid = Dense(nreg => nreg^2; use_bias = false)
    proj_lift_ps, _ = Lux.setup(rng, proj_lift) |> f
    copyto!(proj_lift_ps.weight, e_lift)
    proj_sink_ps, _ = Lux.setup(rng, proj_sink) |> f
    copyto!(proj_sink_ps.weight, e_sink)
    proj_mid_ps, _ = Lux.setup(rng, proj_mid) |> f
    copyto!(proj_mid_ps.weight, e_mid)
    function project_lift(ps)
        w, b = ps.weight, ps.bias
        _, c_out = size(w)
        w = proj_lift(w, proj_lift_ps, (;)) |> first
        w = reshape(w, nreg, nten, c_out)
        w = permutedims(w, (2, 1, 3))
        weight = reshape(w, 1, nten, nreg * c_out)
        bias = reshape(repeat(reshape(b, 1, :), nreg), :)
        return (; weight, bias)
    end
    function project_mid(ps)
        w, b = ps.weight, ps.bias
        _, c_out, c_in = size(w)
        w = reshape(w, nreg, :)
        w = proj_mid(w, proj_mid_ps, (;)) |> first
        w = reshape(w, nreg, nreg, c_out, c_in)
        w = permutedims(w, (2, 4, 1, 3))
        weight = reshape(w, 1, nreg * c_in, nreg * c_out)
        bias = reshape(repeat(reshape(b, 1, :), nreg), :)
        return (; weight, bias)
    end
    function project_sink(ps)
        w = ps.weight
        _, c_in = size(w)
        w = proj_sink(w, proj_sink_ps, (;)) |> first
        w = reshape(w, nten, nreg, c_in)
        w = permutedims(w, (2, 3, 1))
        weight = reshape(w, 1, nreg * c_in, nten)
        return (; weight)
    end
    nchan = [10, 10, 20, 20]
    project(ps) = (;
        lift = project_lift(ps.lift),
        mid_1 = project_mid(ps.mid_1),
        mid_2 = project_mid(ps.mid_2),
        mid_3 = project_mid(ps.mid_3),
        sink = project_sink(ps.sink),
    )
    net = Chain(;
        lift = Conv((1,), nten => nreg * nchan[1], gelu),
        map(
            i ->
            Symbol(:mid_, i) =>
                Conv((1,), nreg * nchan[i] => nreg * nchan[i + 1], gelu),
            1:(length(nchan) - 1),
        )...,
        sink = Conv((1,), nreg * nchan[end] => nten),
    )
    net |> display
    ps = (;
        lift = (; weight = randn(T, nten, nchan[1]), bias = randn(T, nchan[1])),
        map(
            i ->
            Symbol(:mid_, i) => (;
                weight = randn(T, nreg, nchan[i + 1], nchan[i]),
                bias = randn(T, nchan[i + 1]),
            ),
            1:(length(nchan) - 1),
        )...,
        sink = (; weight = randn(T, nten, nchan[end])),
    )
    st = map(Returns((;)), ps)
    ps = project(ps)
    i = 6
    mat = mats[i]

    # Input
    x = @SMatrix(randn(D, D)) |> f

    # Rotated input
    rx = mat * x * mat'

    # Net on input
    nx = reshape(Array(x), 1, nten, 1)
    nx = net(nx, ps, st)[1]
    nx = reshape(nx, D, D) |> SMatrix{D, D, T, nten}

    # Net on rotated input
    nrx = reshape(Array(rx), 1, nten, 1)
    nrx = net(nrx, ps, st) |> first
    nrx = reshape(nrx, D, D) |> SMatrix{D, D, T, nten}

    # Rotated net'ed input
    rnx = mat * nx * mat'

    # Error in output tensor
    nrx - rnx |> display
    return nothing
end

"Verification with DNS-aided LES."
function dns_aid()
    visc = 4.0e-4
    t = 0.0
    cfl = 0.85
    tstop = 1.0e-1
    D = 3
    g = Grid{D}(; l = 1.0, n = 16)
    gbar = Grid{D}(; l = 1.0, n = 8)
    u = vectorfield(g)
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
        Δt = cfl * propose_timestep(u, g, visc, c)
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
        for i in 1:dim(g)
            axpy!(Δt, c.du[i], u[i])
            axpy!(Δt, cbar.du[i], v[i])
        end
        apply!(project!, g, (u, g))
        apply!(project!, gbar, (v, gbar))
    end
    foreach(i -> apply!(cutoff!, gbar, (ubar[i], u[i])), 1:D)
    return sum(i -> sum(abs2, v[i] - ubar[i]) / sum(abs2, ubar[i]), 1:D)
end

function test_equivariance_post(setup, ustart, model; groupindex, tstop, dolog)

    (; cfl) = setup
    grid = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)

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

    cache = getcache(grid)
    if !isnothing(model)
        # Allocate velocity gradient for closure
        cache = (;
            cache...,
            G = tensorfield_nonsym(grid),
            GG = spacetensorfield_nonsym(grid),
        )
    end

    # Time stepping
    (; visc) = setup
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
    return norm(rsu - sru) / norm(sru)
end
