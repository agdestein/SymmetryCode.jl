Rx(θ) = @SMatrix [
    1 0 0
    0 cos(θ) -sin(θ)
    0 sin(θ) cos(θ)
]

Ry(θ) = @SMatrix [
    cos(θ) 0 sin(θ)
    0 1 0
    -sin(θ) 0 cos(θ)
]

Rz(θ) = @SMatrix [
    cos(θ) -sin(θ) 0
    sin(θ) cos(θ) 0
    0 0 1
]

export Rx, Ry, Rz

@inline roto_reflection_matrix(p, s) = @SMatrix [s[i] * (p[i] == j) for i = 1:3, j = 1:3]
@inline function invtransform(p, s)
    pinv = invperm(p)
    sinv = map(p -> s[p], pinv)
    pinv, sinv
end

"""
Get group element of the octohedral group and related quantities.

The octahedral group is composed of 90-degree rotations around
(and reflections along) the 3 canonical axes.
We parameterize the group elements as a permutation
of the axes followed by a sign-flip of each the axes.
"""
function octahedral_group()
    permutations = [(1, 2, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1), (2, 1, 3), (1, 3, 2)]
    signs = [
        (+1, +1, +1),
        (-1, +1, +1),
        (+1, -1, +1),
        (+1, +1, -1),
        (-1, -1, +1),
        (+1, -1, -1),
        (-1, +1, -1),
        (-1, -1, -1),
    ]
    elements =
        Iterators.product(eachindex(permutations), eachindex(signs)) |> collect |> vec
    indices = eachindex(elements)
    mats = [roto_reflection_matrix(p, s) for p in permutations, s in signs] |> vec
    unitindex = findfirst(==(I), mats) # Should be equal to 1
    dets = map(m -> m |> det |> Int, mats)
    products = reshape(mats, :) .* reshape(mats, 1, :)
    cayley = map(m -> findfirst(==(m), reshape(mats, :)), products)
    inverse_indices = map(i -> findfirst(==(unitindex), cayley[i, :]), eachindex(elements))
    inverse_elements = elements[inverse_indices]
    (;
        permutations,
        signs,
        indices,
        inverse_indices,
        elements,
        inverse_elements,
        unitindex,
        mats,
        dets,
        products,
        cayley,
    )
end

export roto_reflection_matrix, invtransform, octahedral_group

function get_weight_projectors()
    (; permutations, signs, elements, cayley) = octahedral_group()
    r = map(Iterators.product(1:48, 1:3, 1:3, 1:48, 1:3, 1:3)) do (m, x, y, n, i, j)
        sum(1:48) do g
            gp, gs = elements[g]
            p, s = permutations[gp], signs[gs]
            s[x] * s[y] * (cayley[g, n] == m) * (p[x] == i) * (p[y] == j)
        end
    end
    r_lift = reshape(r, 48 * 9, 48 * 9)
    r = map(Iterators.product(1:3, 1:3, 1:48, 1:3, 1:3, 1:48)) do (x, y, m, i, j, n)
        sum(1:48) do g
            gp, gs = elements[g]
            p, s = permutations[gp], signs[gs]
            s[x] * s[y] * (cayley[g, n] == m) * (p[x] == i) * (p[y] == j)
        end
    end
    r_sink = reshape(r, 9 * 48, 9 * 48)
    r = map(Iterators.product(1:48, 1:48, 1:48, 1:48)) do (m, n, a, b)
        sum(1:48) do g
            # a => b
            # m => n
            (cayley[g, a] == m) * (cayley[g, b] == n)
        end
    end
    r_mid = reshape(r, 48^2, 48^2)
    (; r_lift, r_sink, r_mid)
end

function test_equivariant_dense()
    (; permutations, signs, elements, mats, cayley) = octahedral_group()
    (; r_lift, r_sink, r_mid) = get_weight_projectors()
    rng = Xoshiro(0)
    f = f64
    proj_lift = Dense(48 * 9 => 48 * 9; use_bias = false)
    proj_sink = Dense(9 * 48 => 9 * 48; use_bias = false)
    proj_mid = Dense(48^2 => 48^2; use_bias = false)
    proj_lift_ps, _ = Lux.setup(rng, proj_lift) |> f
    proj_lift_ps.weight .= r_lift
    proj_sink_ps, _ = Lux.setup(rng, proj_sink) |> f
    proj_sink_ps.weight .= r_sink
    proj_mid_ps, _ = Lux.setup(rng, proj_mid) |> f
    proj_mid_ps.weight .= r_mid
    function project_lift(ps)
        w, b = ps.weight, ps.bias
        s_out, s_in = size(w)
        c_out, c_in = div(s_out, 48), div(s_in, 9)
        w = reshape(w, 48, c_out, 9, c_in)
        w = permutedims(w, (1, 3, 2, 4))
        w = reshape(w, 48 * 9, :)
        w = proj_lift(w, proj_lift_ps, (;)) |> first
        w = reshape(w, 48, 9, c_out, c_in)
        w = permutedims(w, (1, 3, 2, 4))
        weight = reshape(w, s_out, s_in)
        bias = fill(b, s_out)
        (; weight, bias)
    end
    function project_mid(ps)
        w, b = ps.weight, ps.bias
        s_out, s_in = size(w)
        c_out, c_in = div(s_out, 48), div(s_in, 48)
        w = reshape(w, 48, c_out, 48, c_in)
        w = permutedims(w, (1, 3, 2, 4))
        w = reshape(w, 48 * 48, :)
        w = proj_mid(w, proj_mid_ps, (;)) |> first
        w = reshape(w, 48, 48, c_out, c_in)
        w = permutedims(w, (1, 3, 2, 4))
        weight = reshape(w, s_out, s_in)
        bias = fill(b, s_out)
        (; weight, bias)
    end
    function project_sink(ps)
        w = ps.weight
        s_out, s_in = size(w)
        c_out, c_in = div(s_out, 9), div(s_in, 48)
        w = reshape(w, 9, c_out, 48, c_in)
        w = permutedims(w, (1, 3, 2, 4))
        w = reshape(w, 9 * 48, :)
        w = proj_sink(w, proj_sink_ps, (;)) |> first
        w = reshape(w, 9, 48, c_out, c_in)
        w = permutedims(w, (1, 3, 2, 4))
        weight = reshape(w, s_out, s_in)
        (; weight)
    end
    project(ps) = (;
        lift = project_lift(ps.lift),
        mid_1 = project_mid(ps.mid_1),
        mid_2 = project_mid(ps.mid_2),
        mid_3 = project_mid(ps.mid_3),
        sink = project_sink(ps.sink),
    )
    net = Chain(;
        lift = Dense(9 => 48 * 10, gelu),
        mid_1 = Dense(48 * 10 => 48 * 10, gelu),
        mid_2 = Dense(48 * 10 => 48 * 20, gelu),
        mid_3 = Dense(48 * 20 => 48 * 20, gelu),
        sink = Dense(48 * 20 => 9),
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
    i = 11
    mat = mats[i]
    x = @SMatrix(randn(3, 3)) |> f
    rx = mat * x * mat'
    nx =
        net(reshape(Array(x), 9, 1), ps, st)[1] |>
        x -> reshape(x, 3, 3) |> SMatrix{3,3,eltype(x),9}
    nrx =
        net(reshape(Array(rx), 9, 1), ps, st)[1] |>
        x -> reshape(x, 3, 3) |> SMatrix{3,3,eltype(x),9}
    rnx = mat * nx * mat'
    nrx - rnx |> display
    nothing
    # nx = net(reshape(Array(x), 9, 1), ps, st)[1] |> x -> reshape(x, 48)
    # nrx = net(reshape(Array(rx), 9, 1), ps, st)[1] |> x -> reshape(x, 48)
    # rnx = nx[invperm(cayley[i, :])]
    # nrx - rnx |> display
    # nothing
end

function test_equivariant_conv()
    (; permutations, signs, elements, mats, cayley) = octahedral_group()
    (; r_lift, r_sink, r_mid) = get_weight_projectors()
    rng = Xoshiro(0)
    f = f64
    D = 3
    nten = D^2
    nreg = 48
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
        (; weight, bias)
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
        (; weight, bias)
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
        (; weight)
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
    a = T(0)
    ps = (;
        lift = (; ps.lift.weight, bias = a * randn(rng, T)),
        mid_1 = (; ps.mid_1.weight, bias = a * randn(rng, T)),
        mid_2 = (; ps.mid_2.weight, bias = a * randn(rng, T)),
        mid_3 = (; ps.mid_3.weight, bias = a * randn(rng, T)),
        sink = (; ps.sink.weight),
    )
    ps = project(ps)
    i = 11
    mat = mats[i]
    x = @SMatrix(randn(3, 3)) |> f
    rx = mat * x * mat'
    nx =
        net(reshape(Array(x), 1, nten, 1), ps, st)[1] |>
        x -> reshape(x, D, D) |> SMatrix{D,D,eltype(x),nten}
    nrx =
        net(reshape(Array(rx), 1, nten, 1), ps, st)[1] |>
        x -> reshape(x, D, D) |> SMatrix{D,D,eltype(x),nten}
    rnx = mat * nx * mat'
    nrx - rnx |> display
    nothing
    # nx = net(reshape(Array(x), 1, nten, 1), ps, st)[1] |> x -> reshape(x, nreg)
    # nrx = net(reshape(Array(rx), 1, nten, 1), ps, st)[1] |> x -> reshape(x, nreg)
    # rnx = nx[cayley[i, :] |> invperm]
    # nrx - rnx |> display
    # nothing
end

export get_weight_projectors
