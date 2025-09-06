# [Rodrigues' rotation formula - Wikipedia](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)
# [Rotation matrix - Wikipedia](https://en.wikipedia.org/wiki/Rotation_matrix)
# [Training/AdvancedGPU/2-2-kernel_analysis_optimization.ipynb at master · JuliaComputing/Training](https://github.com/JuliaComputing/Training/blob/master/AdvancedGPU/2-2-kernel_analysis_optimization.ipynb)
# [FZJ-JSC/tutorial-multi-gpu: Efficient Distributed GPU Programming for Exascale, an SC/ISC Tutorial](https://github.com/FZJ-JSC/tutorial-multi-gpu)
# [Cayley table - Wikipedia](https://en.wikipedia.org/wiki/Cayley_table)

if false
    include("src/SymmetryCode.jl")
    using .SymmetryCode
end

# using KernelAbstractions
using LinearAlgebra
using Lux
using Random
using SymmetryCode
# using StaticArrays
using WGLMakie


SymmetryCode.test_equivariant_dense()
SymmetryCode.test_equivariant_conv()

let
    # net = Conv((1,), 3 => 10)
    net = Dense(3 => 10)
    ps, st = Lux.setup(Xoshiro(0), net)
    ps.weight
end


(; permutations, signs, elements, mats, cayley) = octahedral_group()

(; r_lift, r_sink, r_mid) = get_weight_projectors()

e = eigen(r_lift / 48; sortby = -)
e = eigen(r_mid / 48; sortby = -)
e = eigen(r_sink / 48; sortby = -)
e = r_mid / 48 |> svd
e = r_sink / 48 |> svd

s.U[:, 1:9]

s.S

r_lift * r_lift / 48^2 - r_lift / 48 |> extrema

uu = reshape(s.U[:, 1:9], 48, 9, 9)
uu[:, :, 1] * sqrt(48)
uu[:, :, 7] * sqrt(48) .|> x -> round(x; digits = 2)

s.U * Diagonal(s.S) * s.Vt
s.U[:, 1:9] |> extrema

let
    ip, is = 2, 1
    p, s = permutations[ip], signs[is]
    g = Grid(1.0, 16)
    u = [@SVector(randn(3)) for i = 1:g.n, j = 1:g.n, k = 1:g.n]
    Gu = fill(@SMatrix(zeros(3, 3)), g.n, g.n, g.n)
    GRu = fill(@SMatrix(zeros(3, 3)), g.n, g.n, g.n)
    apply!(grad!, g, (g, Gu, u))
    Ru = transform_vector(u, (p, s))
    RGu = transform_tensor(Gu, (p, s))
    apply!(grad!, g, (g, GRu, Ru))
    GRu - RGu |> norm
end

let
    ip, is = 5, 2
    p, s = permutations[ip], signs[is]
    g = Grid(1.0, 16)
    f = randn(g.n, g.n, g.n)
    Gf = [@SVector(zeros(3)) for i = 1:g.n, j = 1:g.n, k = 1:g.n]
    GRf = [@SVector(zeros(3)) for i = 1:g.n, j = 1:g.n, k = 1:g.n]
    apply!(grad_scalar!, g, (g, Gf, f))
    Rf = transform_scalar(f, (p, s))
    RGf = transform_vector(Gf, (p, s))
    apply!(grad_scalar!, g, (g, GRf, Rf))
    GRf - RGf |> norm
end

let
    ip, is = 2, 5
    p, s = permutations[ip], signs[is]
    g = Grid(1.0, 16)
    f = randn(g.n, g.n, g.n)
    Rf = transform_scalar(f, (p, s))
    RRf = transform_scalar(Rf, invtransform(p, s))
    f - RRf |> norm
end
