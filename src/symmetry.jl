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


"""
Construct Reynolds operators for equivariant weight spaces.

Each matrix averages over the octahedral group action. Eigenvectors with
eigenvalue one span the admissible weight subspace used to lift/mix/sink
features in `equivariant_net`.
"""
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


"""
Closed-form equivariant weight bases — the direct synthesis that replaces the
eigendecomposition of [`get_weight_projectors`](@ref) (spec:
`notes/equivariant-weight-synthesis.md`).

For regular-representation hidden channels and a tensor representation at the
lift/sink boundaries, the equivariant subspace has a closed form, so the three
returned bases play exactly the role of `eigen(r_*).vectors[:, 1:k]` — same
shape, same row ordering — but with entries in `{0, ±1/√|G|}`:

  - `s_mid`  (`|G|² × |G|`): a **group-circulant gather**. `s_mid * vec(k)`
    tiles the learnable `k(n⁻¹m)` over the regular representation. `relidx` is
    the Cayley-derived gather index (`relidx[m,n]` = flat index of `n⁻¹·m`,
    i.e. the paper's `h⁻¹g`).
  - `s_lift` (`|G|·D² × D²`) and `s_sink` (`D²·|G| × D²`): the **orbit** of the
    tensor representation `Qmat[h]` (= `R_h ⊗ R_h`), the intertwiner between the
    `D²`-dimensional tensor rep and the regular rep at the boundaries.

`Qmat[h]` is built directly from the same index expression as `r_lift`'s tensor
block, `Qmat[h][(x,y),(i,j)] = s[x] s[y] (p[x]==i)(p[y]==j)`, flattened with `x`
(then `y`) fastest. Each basis column lies in the eigenvalue-one subspace of the
matching Reynolds projector and the columns are orthonormal, so the synthesis
spans the same space as the eigenbasis while making the weights *bit-exactly*
equivariant (every `Qmat[h]` is a signed permutation; the `1/√|G|` is a single
global scale). Cross-checked against the eigen path in `verify.jl` /
`test/test_symmetry.jl`.
"""
function get_weight_synthesis(D)
    (; permutations, signs, elements, cayley, inverse_indices) = group_stuff(D)
    nreg = length(elements)
    nten = D^2
    α = 1 / sqrt(nreg)

    # Tensor-rep matrices, built from r_lift's index expression (not assumed to
    # be a kron), with (x,y) ↔ μ flattened x-fastest to match the Conv weight.
    Qmat = map(elements) do (gp, gs)
        p, s = permutations[gp], signs[gs]
        Q = zeros(Int, nten, nten)
        for j in 1:D, i in 1:D, y in 1:D, x in 1:D
            μ, ν = x + (y - 1) * D, i + (j - 1) * D
            Q[μ, ν] = s[x] * s[y] * (p[x] == i) * (p[y] == j)
        end
        Q
    end

    # Mid (regular → regular): group-circulant gather. Row index (m,n), m
    # fastest, matching `r_mid`'s reshape.
    relidx = [cayley[inverse_indices[n], m] for m in 1:nreg, n in 1:nreg]
    s_mid = zeros(nreg^2, nreg)
    for n in 1:nreg, m in 1:nreg
        s_mid[m + (n - 1) * nreg, relidx[m, n]] = α
    end

    # Lift (tensor → regular): w_lift[h,:,cout] = Qmat[h]·c. Row (h,μ), h
    # fastest, matching `r_lift`'s reshape.
    s_lift = zeros(nreg * nten, nten)
    for ν in 1:nten, μ in 1:nten, h in 1:nreg
        s_lift[h + (μ - 1) * nreg, ν] = α * Qmat[h][μ, ν]
    end

    # Sink (regular → tensor): w_sink[:,h,cin] = Qmat[h]·d. Row (μ,h), μ
    # fastest, matching `r_sink`'s reshape.
    s_sink = zeros(nten * nreg, nten)
    for ν in 1:nten, h in 1:nreg, μ in 1:nten
        s_sink[μ + (h - 1) * nten, ν] = α * Qmat[h][μ, ν]
    end

    return (; relidx, Qmat, s_lift, s_mid, s_sink)
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
    for (uu, u) in zip(uu, u)
        copyto!(temp, u)
        apply!(twothirds!, g, (temp, g))
        to_phys!(uu, temp, plan, g)
    end
    return uu
end

function forward_vector_fourier(uu, g)
    u = vectorfield(g)
    plan = plan_rfft(uu.x)
    for (uu, u) in zip(uu, u)
        to_spec!(u, uu, plan, g)
    end
    return u
end

"""
Apply one octahedral group element to a physical-space vector field.

The array is permuted/flipped first (moving the sample points), then each
stored vector is multiplied by the same roto-reflection matrix.
"""
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

"""
Apply one octahedral group element to a symmetric physical tensor field.

Uses the tensor rule `T -> R*T*R'` after moving the grid points, preserving the
packed symmetric component layout used by the solver.
"""
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
