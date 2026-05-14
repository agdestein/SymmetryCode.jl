# Octahedral group: matrices, structure constants, weight-projection operators,
# and the action of the group on physical / spectral fields.

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
    @SMatrix [s[i] * (p[i] == j) for i in 1:2, j in 1:2]
@inline roto_reflection_matrix(p::NTuple{3}, s) =
    @SMatrix [s[i] * (p[i] == j) for i in 1:3, j in 1:3]
@inline function invtransform(p, s)
    pinv = invperm(p)
    sinv = map(p -> s[p], pinv)
    return pinv, sinv
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
    return (;
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
    return (; r_lift, r_mid, r_sink)
end

function vectorfield_to_svector(u)
    D = ndims(u[1])
    T = eltype(u[1])
    M = SVector{D, T}
    return M.(u...)
end
function svector_to_vectorfield(u)
    V = eltype(u)
    z = zero(V)
    D = size(z, 1)
    return if D == 2
        (; x = getindex.(u, 1), y = getindex.(u, 2))
    elseif D == 3
        (; x = getindex.(u, 1), y = getindex.(u, 2), z = getindex.(u, 3))
    end
end

function tensorfield_to_smatrix(t)
    D = ndims(t[1])
    T = eltype(t[1])
    M = SMatrix{D, D, T, D^2}
    return M.(t...)
end
function smatrix_to_tensorfield(t)
    M = eltype(t)
    z = zero(M)
    D = size(z, 1)
    return if D == 2
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

function inverse_vector_fourier(u, g)
    uu = spacevectorfield(g)
    temp = scalarfield(g)
    plan = plan_rfft(uu.x)
    fac = get_fft_fac(g)
    for (uu, u) in zip(uu, u)
        copyto!(temp, u)
        temp .*= fac
        apply!(twothirds!, g, (temp, g))
        ldiv!(uu, plan, temp)
    end
    return uu
end

function forward_vector_fourier(uu, g)
    u = vectorfield(g)
    plan = plan_rfft(uu.x)
    fac = get_fft_fac(g)
    for (uu, u) in zip(uu, u)
        mul!(u, plan, uu)
        u ./= fac
    end
    return u
end

function transform_vector(u, g, (p, s))
    T, D = typeof(g.l), dim(g)
    u_sa = SVector{D, T}.(u...)
    u_sa = permutedims(u_sa, p)
    dims = (findall(==(-1), s)...,)
    u_sa = reverse(u_sa; dims)
    m = roto_reflection_matrix(p, s)
    ru_sa = map(u -> m * u, u_sa)
    return if D == 2
        (; x = getindex.(ru_sa, 1), y = getindex.(ru_sa, 2))
    elseif D == 3
        (; x = getindex.(ru_sa, 1), y = getindex.(ru_sa, 2), z = getindex.(ru_sa, 3))
    end
end

function transform_tensor(t, g, (p, s))
    T, D = typeof(g.l), dim(g)
    SM = SMatrix{D, D, T, D^2}
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
    return if D == 2
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
    SM = SMatrix{D, D, T, D^2}
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
    return NamedTuple(pairs)
end
