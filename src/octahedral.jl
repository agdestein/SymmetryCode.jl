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

export get_weight_projectors
