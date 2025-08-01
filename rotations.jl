# [Rodrigues' rotation formula - Wikipedia](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)
# [Rotation matrix - Wikipedia](https://en.wikipedia.org/wiki/Rotation_matrix)
# [Training/AdvancedGPU/2-2-kernel_analysis_optimization.ipynb at master · JuliaComputing/Training](https://github.com/JuliaComputing/Training/blob/master/AdvancedGPU/2-2-kernel_analysis_optimization.ipynb)
# [FZJ-JSC/tutorial-multi-gpu: Efficient Distributed GPU Programming for Exascale, an SC/ISC Tutorial](https://github.com/FZJ-JSC/tutorial-multi-gpu)

if false
    include("src/SymmetryCode.jl")
    using .SymmetryCode
end

using LinearAlgebra
using SymmetryCode
using StaticArrays

permutations = [
    (1, 2, 3),
    (2, 3, 1),
    (3, 1, 2),
    (3, 2, 1),
    (2, 1, 3),
    (1, 3, 2),
]

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

elements = Iterators.product(eachindex(permutations), eachindex(signs))
elements |> collect

matrix(p, s) = @SMatrix [s[i] * (p[i] == j) for i in 1:3, j in 1:3]

mats = map(splat(matrix), Iterators.product(permutations, signs))

det.(mats) .|> Int

mats

mats[8]
mats[7]

t = @SMatrix [
    11 12 13
    21 22 23
    31 32 33
]

m = mats[4, 2]
det(m)

map(m -> m * t * m', mats)
map(m -> m * t * m', mats) .|> display;

m = mats[3] * mats[7]
findfirst(==(m), mats |> vec)

mats[3, 3]
mats[15]
m

map(display, mats);

products = mats .* reshape(mats, 1, :)

map(m -> findfirst(==(m), mats), products)

mats

"""
Represent all the tensor components within a control volume.
"""
struct CenterTensor{T}
    """
    The quantity `diag[k]` is equal to ``σ_{k, k}(x)`` where ``x`` is the volume center.
    """
    diag::NTuple{3,T}

    """
    The quantity
    `edge[k,a,pi,pj]` is equal to
    ``σ_{i,j}(x \\pm h/2 e_i \\pm h/2 e_j)`` if `a == 1`,
    ``σ_{j,i}(x \\pm h/2 e_i \\pm h/2 e_j)`` if `a == 2`,
    where
    ``x`` is the volume center,
    ``i \\prec j \\prec k`` are oriented positively,
    ``\\pm h/2 e_i`` uses ``+`` if `pi == 1` and ``-`` if `pi == 2`,
    and similarly for `pj`.
    """
    edge::SArray{Tuple{3,2,2,2},T,4,24}
end

diag = (1.0, 2.0, 3.0)
edge = @SArray randn(3, 2, 2, 2)
t = CenterTensor(diag, edge)
