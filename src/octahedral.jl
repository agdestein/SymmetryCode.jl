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


"Get roto-reflection matrix from permutation and sign-flip."
@inline roto_reflection_matrix(p::NTuple{2}, s) =
    @SMatrix [s[i] * (p[i] == j) for i = 1:2, j = 1:2]
@inline roto_reflection_matrix(p::NTuple{3}, s) =
    @SMatrix [s[i] * (p[i] == j) for i = 1:3, j = 1:3]
@inline function invtransform(p, s)
    pinv = invperm(p)
    sinv = map(p -> s[p], pinv)
    pinv, sinv
end

"""
Get group elements of the octahedral group and related quantities.

The octahedral group is composed of 90-degree rotations around
(and reflections along) the 3 canonical axes.
We parameterize the group elements as a permutation
of the axes followed by a sign-flip of each the axes.
"""
function group_stuff(D)
    if D == 2
        permutations = [(1, 2), (2, 1)]
        signs = [(+1, +1), (-1, +1), (+1, -1), (-1, -1)]
    elseif D == 3
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
    end
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


function get_weight_projectors(D)
    (; permutations, signs, elements, cayley) = group_stuff(D)
    nelement = length(elements)
    r_lift = map(
        Iterators.product(1:nelement, 1:D, 1:D, 1:nelement, 1:D, 1:D),
    ) do (m, x, y, n, i, j)
        sum(1:nelement) do g
            gp, gs = elements[g]
            p, s = permutations[gp], signs[gs]
            s[x] * s[y] * (cayley[g, n] == m) * (p[x] == i) * (p[y] == j)
        end
    end
    r_lift = reshape(r_lift, nelement * D^2, nelement * D^2)
    r_mid = map(
        Iterators.product(1:nelement, 1:nelement, 1:nelement, 1:nelement),
    ) do (m, n, a, b)
        sum(1:nelement) do g
            # a => b
            # m => n
            (cayley[g, a] == m) * (cayley[g, b] == n)
        end
    end
    r_mid = reshape(r_mid, nelement^2, nelement^2)
    r_sink = map(
        Iterators.product(1:D, 1:D, 1:nelement, 1:D, 1:D, 1:nelement),
    ) do (x, y, m, i, j, n)
        sum(1:nelement) do g
            gp, gs = elements[g]
            p, s = permutations[gp], signs[gs]
            s[x] * s[y] * (cayley[g, n] == m) * (p[x] == i) * (p[y] == j)
        end
    end
    r_sink = reshape(r_sink, D^2 * nelement, D^2 * nelement)
    (; r_lift, r_mid, r_sink)
end

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
        (; weight, bias)
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
        (; weight, bias)
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
        x -> reshape(x, D, D) |> SMatrix{D,D,eltype(x),D^2}
    nrx =
        net(reshape(Array(rx), D^2, 1), ps, st)[1] |>
        x -> reshape(x, D, D) |> SMatrix{D,D,eltype(x),D^2}
    rnx = mat * nx * mat'
    nrx - rnx |> display
    nothing
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
        x -> reshape(x, D, D) |> SMatrix{D,D,eltype(x),nten}
    nrx =
        net(reshape(Array(rx), 1, nten, 1), ps, st)[1] |>
        x -> reshape(x, D, D) |> SMatrix{D,D,eltype(x),nten}
    rnx = mat * nx * mat'
    nrx - rnx |> display
    nothing
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
        (; weight, bias)
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
        (; weight, bias)
    end
    function project_sink(ps)
        w = ps.weight
        _, c_in = size(w)
        w = proj_sink(w, proj_sink_ps, (;)) |> first
        w = reshape(w, nten, nreg, c_in)
        w = permutedims(w, (2, 3, 1))
        weight = reshape(w, 1, nreg * c_in, nten)
        (; weight)
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
                    Conv((1,), nreg * nchan[i] => nreg * nchan[i+1], gelu),
            1:(length(nchan)-1),
        )...,
        sink = Conv((1,), nreg * nchan[end] => nten),
    )
    net |> display
    ps = (;
        lift = (; weight = randn(T, nten, nchan[1]), bias = randn(T, nchan[1])),
        map(
            i ->
                Symbol(:mid_, i) => (;
                    weight = randn(T, nreg, nchan[i+1], nchan[i]),
                    bias = randn(T, nchan[i+1]),
                ),
            1:(length(nchan)-1),
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
    nx = reshape(nx, D, D) |> SMatrix{D,D,T,nten}

    # Net on rotated input
    nrx = reshape(Array(rx), 1, nten, 1)
    nrx = net(nrx, ps, st) |> first
    nrx = reshape(nrx, D, D) |> SMatrix{D,D,T,nten}

    # Rotated net'ed input
    rnx = mat * nx * mat'

    # Error in output tensor
    nrx - rnx |> display
    nothing
end

function equivariant_net(setup, nchan)
    (; D, backend) = setup
    dev = adapt(backend)
    # dev = identity
    rng = Xoshiro(0)
    T, f = Float64, f64
    nten = D^2
    (; elements) = group_stuff(D)
    (; r_lift, r_sink, r_mid) = get_weight_projectors(D)
    nreg = length(elements)
    e_lift = eigen(r_lift / nreg; sortby = -).vectors[:, 1:nten]
    e_mid = eigen(r_mid / nreg; sortby = -).vectors[:, 1:nreg]
    e_sink = eigen(r_sink / nreg; sortby = -).vectors[:, 1:nten]
    proj_lift = Dense(nten => nten * nreg; use_bias = false)
    proj_sink = Dense(nten => nreg * nten; use_bias = false)
    proj_mid = Dense(nreg => nreg^2; use_bias = false)
    proj_lift_ps, _ = Lux.setup(rng, proj_lift) |> f |> dev
    copyto!(proj_lift_ps.weight, e_lift)
    proj_sink_ps, _ = Lux.setup(rng, proj_sink) |> f |> dev
    copyto!(proj_sink_ps.weight, e_sink)
    proj_mid_ps, _ = Lux.setup(rng, proj_mid) |> f |> dev
    copyto!(proj_mid_ps.weight, e_mid)
    kern = ntuple(Returns(1), D)
    function project_lift(ps)
        w, b = ps.weight, ps.bias
        _, c_out = size(w)
        w = proj_lift(w, proj_lift_ps, (;)) |> first
        w = reshape(w, nreg, nten, c_out)
        w = permutedims(w, (2, 1, 3))
        weight = reshape(w, kern..., nten, nreg * c_out)
        bias = reshape(repeat(reshape(b, 1, :), nreg), :)
        (; weight, bias)
    end
    function project_mid(ps)
        w, b = ps.weight, ps.bias
        _, c_out, c_in = size(w)
        w = reshape(w, nreg, :)
        w = proj_mid(w, proj_mid_ps, (;)) |> first
        w = reshape(w, nreg, nreg, c_out, c_in)
        w = permutedims(w, (2, 4, 1, 3))
        weight = reshape(w, kern..., nreg * c_in, nreg * c_out)
        bias = reshape(repeat(reshape(b, 1, :), nreg), :)
        (; weight, bias)
    end
    function project_sink(ps)
        w = ps.weight
        _, c_in = size(w)
        w = proj_sink(w, proj_sink_ps, (;)) |> first
        w = reshape(w, nten, nreg, c_in)
        w = permutedims(w, (2, 3, 1))
        weight = reshape(w, kern..., nreg * c_in, nten)
        (; weight)
    end
    function project(ps)
        lift, mids..., sink, symm = ps
        (;
            lift = project_lift(lift),
            map(project_mid, mids)...,
            sink = project_sink(sink),
            symm,
        )
    end
    net = Chain(;
        lift = Conv(kern, nten => nreg * nchan[1], gelu),
        map(
            i ->
                Symbol(:mid_, i) =>
                    Conv(kern, nreg * nchan[i] => nreg * nchan[i+1], gelu),
            1:(length(nchan)-1),
        )...,
        sink = Conv(kern, nreg * nchan[end] => nten; use_bias = false),
        symm = WrappedFunction() do σ
            if D == 2
                xx = selectdim(σ, 3, 1:1)
                xy = (selectdim(σ, 3, 2:2) + selectdim(σ, 3, 3:3)) / 2
                yy = selectdim(σ, 3, 4:4)
                cat(xx, yy, xy; dims = 3)
            else
                xx = selectdim(σ, 4, 1:1)
                yy = selectdim(σ, 4, 5:5)
                zz = selectdim(σ, 4, 9:9)
                xy = (selectdim(σ, 4, 2:2) + selectdim(σ, 4, 4:4)) / 2
                yz = (selectdim(σ, 4, 6:6) + selectdim(σ, 4, 8:8)) / 2
                zx = (selectdim(σ, 4, 3:3) + selectdim(σ, 4, 7:7)) / 2
                cat(xx, yy, zz, xy, yz, zx; dims = 4)
            end
        end,
    )
    net |> display
    ps =
        (;
            lift = (;
                weight = kaiming_uniform(rng, T, nten, nchan[1]),
                bias = zeros(T, nchan[1]),
            ),
            map(
                i ->
                    Symbol(:mid_, i) => (;
                        weight = kaiming_uniform(rng, T, nreg, nchan[i+1], nchan[i]),
                        bias = zeros(T, nchan[i+1]),
                    ),
                1:(length(nchan)-1),
            )...,
            sink = (; weight = kaiming_uniform(rng, T, nten, nchan[end])),
            symm = (;),
        ) |> dev
    st = map(Returns((;)), ps)
    (; project, net, ps, st)
end

"Same as `equivariant_net` but without the weight projection."
function cnn(setup, nchan; same_as_equi)
    (; D, backend) = setup
    dev = adapt(backend)
    # dev = identity
    rng = Xoshiro(0)
    f = f64
    nt_nonsym = D^2
    nt = D == 2 ? 3 : 6
    (; elements) = group_stuff(D)
    nreg = if same_as_equi
        length(elements)
    else
        1
    end
    kern = ntuple(Returns(1), D)
    net = Chain(;
        lift = Conv(kern, nt_nonsym => nreg * nchan[1], gelu),
        map(
            i ->
                Symbol(:mid_, i) =>
                    Conv(kern, nreg * nchan[i] => nreg * nchan[i+1], gelu),
            1:(length(nchan)-1),
        )...,
        sink = Conv(kern, nreg * nchan[end] => nt; use_bias = false),
        symm = WrappedFunction(identity),
    )
    net |> display
    ps, st = Lux.setup(rng, net) |> f |> dev
    project = identity # No projection
    (; project, net, ps, st)
end
