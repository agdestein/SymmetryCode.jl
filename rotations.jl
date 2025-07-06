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


m = mats[3] * mats[7]
findfirst(==(m), mats |> vec)

mats[3, 3]
mats[15]
m

map(display, mats);

products = mats .* reshape(mats, 1, :)

map(m -> findfirst(==(m), mats), products)

mats

struct CenterTensor{T}
    diag::NTuple{3, T}
    plus::NTuple{3, SMatrix{2,2,T,4}}
    minus::NTuple{3, SMatrix{2,2,T,4}}
end

