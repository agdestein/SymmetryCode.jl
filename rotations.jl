# [Rodrigues' rotation formula - Wikipedia](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)
# [Rotation matrix - Wikipedia](https://en.wikipedia.org/wiki/Rotation_matrix)
# [Training/AdvancedGPU/2-2-kernel_analysis_optimization.ipynb at master · JuliaComputing/Training](https://github.com/JuliaComputing/Training/blob/master/AdvancedGPU/2-2-kernel_analysis_optimization.ipynb)
# [FZJ-JSC/tutorial-multi-gpu: Efficient Distributed GPU Programming for Exascale, an SC/ISC Tutorial](https://github.com/FZJ-JSC/tutorial-multi-gpu)
# [Cayley table - Wikipedia](https://en.wikipedia.org/wiki/Cayley_table)

# using KernelAbstractions
using LinearAlgebra
using Lux
using Random
using SymmetryCode
# using StaticArrays
using WGLMakie

SymmetryCode.test_equivariant_dense(3)
SymmetryCode.test_equivariant_conv(2)
SymmetryCode.test_equivariant_conv_sparse(2)

let
    # net = Conv((1,), 3 => 10)
    net = Dense(3 => 10)
    ps, st = Lux.setup(Xoshiro(0), net)
    ps.weight
end

(; permutations, signs, elements, mats, cayley) = group_stuff(3)

(; r_lift, r_sink, r_mid) = get_weight_projectors(3)

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
