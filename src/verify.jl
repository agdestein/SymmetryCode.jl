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

"""
Cross-check the closed-form weight synthesis (`get_weight_synthesis`, used by
`equivariant_net`) against the eigendecomposition fallback
(`get_weight_projectors`). Confirms (1) the synthesis bases span the *same*
subspace as `eigen(r_*).vectors`, (2) the synthesized weight blocks are
*bit-exactly* group-equivariant (the mid block is group-circulant to the last
bit), and (3) the resulting net's a-priori equivariance error is no larger than
the eigen path's. The net-level error is not exactly zero — it is floored by
floating-point reduction order in the matmul/gelu, *not* by the weights — so the
"structural" exactness lives at the weight level, item (2).
"""
function test_equivariant_conv_sparse(D)
    T = Float64
    nten = D^2
    (; elements, mats, cayley) = group_stuff(D)
    nreg = length(elements)

    # Eigenbasis (general fallback) vs. closed-form synthesis. Both triples have
    # identical shapes / row orderings, so they drop into the same `project_*`.
    (; r_lift, r_sink, r_mid) = get_weight_projectors(D)
    e_lift = eigen(r_lift / nreg; sortby = -).vectors[:, 1:nten]
    e_mid = eigen(r_mid / nreg; sortby = -).vectors[:, 1:nreg]
    e_sink = eigen(r_sink / nreg; sortby = -).vectors[:, 1:nten]
    (; s_lift, s_mid, s_sink) = get_weight_synthesis(D)

    # (1) Same subspace: the orthogonal projectors onto the column spaces agree.
    subspace = (;
        lift = norm(s_lift * s_lift' - e_lift * e_lift'),
        mid = norm(s_mid * s_mid' - e_mid * e_mid'),
        sink = norm(s_sink * s_sink' - e_sink * e_sink'),
    )

    nchan = [10, 10, 20, 20]
    kern = ntuple(Returns(1), D)
    function make_project((b_lift, b_mid, b_sink))
        function project_lift(ps)
            w, b = ps.weight, ps.bias
            _, c_out = size(w)
            w = b_lift * w
            w = reshape(w, nreg, nten, c_out)
            w = permutedims(w, (2, 1, 3))
            weight = reshape(w, kern..., nten, nreg * c_out)
            bias = reshape(repeat(reshape(b, 1, :), nreg), :)
            return (; weight, bias)
        end
        function project_mid(ps)
            w, b = ps.weight, ps.bias
            _, c_out, c_in = size(w)
            w = reshape(w, nreg, :)
            w = b_mid * w
            w = reshape(w, nreg, nreg, c_out, c_in)
            w = permutedims(w, (2, 4, 1, 3))
            weight = reshape(w, kern..., nreg * c_in, nreg * c_out)
            bias = reshape(repeat(reshape(b, 1, :), nreg), :)
            return (; weight, bias)
        end
        function project_sink(ps)
            w = ps.weight
            _, c_in = size(w)
            w = b_sink * w
            w = reshape(w, nten, nreg, c_in)
            w = permutedims(w, (2, 3, 1))
            weight = reshape(w, kern..., nreg * c_in, nten)
            return (; weight)
        end
        return ps -> (;
            lift = project_lift(ps.lift),
            mid_1 = project_mid(ps.mid_1),
            mid_2 = project_mid(ps.mid_2),
            mid_3 = project_mid(ps.mid_3),
            sink = project_sink(ps.sink),
        )
    end

    net = Chain(;
        lift = Conv(kern, nten => nreg * nchan[1], gelu),
        map(
            i ->
            Symbol(:mid_, i) =>
                Conv(kern, nreg * nchan[i] => nreg * nchan[i + 1], gelu),
            1:(length(nchan) - 1),
        )...,
        sink = Conv(kern, nreg * nchan[end] => nten),
    )

    # Shared random learnables (so synthesis and eigen are compared fairly).
    ps0 = (;
        lift = (;
            weight = randn(Xoshiro(0), T, nten, nchan[1]),
            bias = randn(Xoshiro(1), T, nchan[1]),
        ),
        map(
            i ->
            Symbol(:mid_, i) => (;
                weight = randn(Xoshiro(10i), T, nreg, nchan[i + 1], nchan[i]),
                bias = randn(Xoshiro(20i), T, nchan[i + 1]),
            ),
            1:(length(nchan) - 1),
        )...,
        sink = (; weight = randn(Xoshiro(2), T, nten, nchan[end])),
    )
    st = map(Returns((;)), ps0)

    # (2) Weight-level bit-exactness: the synthesized mid block is exactly
    # group-circulant (w[g·m, g·n] == w[m,n]) and the lift block is exactly the
    # Q-orbit — both to the last bit, the property the eigenbasis lacks.
    c_out, c_in = size(ps0.mid_1.weight, 2), size(ps0.mid_1.weight, 3)
    wmid = reshape(s_mid * reshape(ps0.mid_1.weight, nreg, :), nreg, nreg, c_out, c_in)
    circulant_err = maximum(eachindex(elements)) do g
        maximum(
            abs,
            wmid[cayley[g, :], cayley[g, :], :, :] .- wmid,
        )
    end

    # (3) A-priori equivariance error of the assembled net (no `symm` tail here,
    # so the output is a full D×D tensor) for each group element.
    function equiv_err(basistriple)
        project = make_project(basistriple)
        ps = project(ps0)
        rng = Xoshiro(123)
        return maximum(eachindex(mats)) do i
            mat = mats[i]
            x = SMatrix{D, D, T}(randn(rng, D, D))
            rx = mat * x * mat'
            nx =
                reshape(net(reshape(Array(x), kern..., nten, 1), ps, st)[1], D, D) |>
                SMatrix{D, D, T, nten}
            nrx =
                reshape(net(reshape(Array(rx), kern..., nten, 1), ps, st)[1], D, D) |>
                SMatrix{D, D, T, nten}
            rnx = mat * nx * mat'
            norm(nrx - rnx)
        end
    end
    err_synthesis = equiv_err((s_lift, s_mid, s_sink))
    err_eigen = equiv_err((e_lift, e_mid, e_sink))

    result = (; subspace, circulant_err, err_synthesis, err_eigen)
    @info "Equivariant weight synthesis cross-check (D = $D)" result...
    return result
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
