# [Rodrigues' rotation formula - Wikipedia](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)
# [Rotation matrix - Wikipedia](https://en.wikipedia.org/wiki/Rotation_matrix)
# [Training/AdvancedGPU/2-2-kernel_analysis_optimization.ipynb at master · JuliaComputing/Training](https://github.com/JuliaComputing/Training/blob/master/AdvancedGPU/2-2-kernel_analysis_optimization.ipynb)
# [FZJ-JSC/tutorial-multi-gpu: Efficient Distributed GPU Programming for Exascale, an SC/ISC Tutorial](https://github.com/FZJ-JSC/tutorial-multi-gpu)
# [Cayley table - Wikipedia](https://en.wikipedia.org/wiki/Cayley_table)

if false
    include("src/SymmetryCode.jl")
    using .SymmetryCode
end

using KernelAbstractions
using LinearAlgebra
using Lux
using Random
using SymmetryCode
using StaticArrays
using WGLMakie

struct Grid{T}
    l::T
    n::Int
end

@inline (g::Grid)(i) = mod1(i, g.n)
@inline (g::Grid)(I::CartesianIndex) = CartesianIndex(map(g, I.I))
@inline dx(g::Grid) = g.l / g.n
@inline shift(I::CartesianIndex{N}, i, n) where {N} =
    CartesianIndex(ntuple(j -> ifelse(j == i, I[j] + n, I[j]), N))
@inline fd(g, f, I, i) = (f[shift(I, i, +1)|>g] - f[shift(I, i, -1)|>g]) / 2 / dx(g)
@inline fd(g, u, I, i, j) =
    (u[shift(I, j, +1)|>g][i] - u[shift(I, j, -1)|>g][i]) / 2 / dx(g)
@kernel function grad!(g::Grid, ∇u, u)
    I = @index(Global, Cartesian)
    ∇u[I] = @SMatrix [fd(g, u, I, i, j) for i = 1:3, j = 1:3]
end
@kernel function grad_scalar!(g::Grid, ∇f, f)
    I = @index(Global, Cartesian)
    ∇f[I] = @SVector [fd(g, f, I, i) for i = 1:3]
end
function apply!(
    kernel,
    g::Grid,
    args;
    backend = CPU(),
    workgroupsize = 64,
    ndrange = (g.n, g.n, g.n),
)
    kernel(backend, workgroupsize)(args...; ndrange)
    nothing
end

function transform_scalar(f, (p, s))
    f = permutedims(f, p)
    dims = (findall(==(-1), s)...,)
    f = reverse(f; dims)
end
function transform_vector(u, (p, s))
    u = permutedims(u, p)
    dims = (findall(==(-1), s)...,)
    u = reverse(u; dims)
    m = roto_reflection_matrix(p, s)
    u = map(u -> m * u, u)
end
function transform_tensor(t, (p, s))
    t = permutedims(t, p)
    dims = (findall(==(-1), s)...,)
    m = roto_reflection_matrix(p, s)
    t = map(t -> m * t * m', t)
end

(; permutations, signs, elements, mats, cayley) = octahedral_group()

(; r_lift, r_sink, r_mid) = get_weight_projectors()

let
    proj = Dense(48 * 9 => 48 * 9; use_bias = false)
    proj_ps, proj_st = Lux.setup(Xoshiro(0), proj)
    proj_ps.weight .= r_lift
    function project(w, b)
        w = reshape(w, 48 * 9, 1)
        w = proj(w, proj_ps, proj_st) |> first
        weight = reshape(w, 48, 9)
        bias = fill(b, 48)
        (; weight, bias)
    end
    net = Dense(9 => 48)
    ps, st = Lux.setup(Xoshiro(0), net)
    w = ps.weight
    b = 1.0f0
    ps = project(w, b)
    ps.weight
    i = 5
    m = mats[i]
    x = @SMatrix randn(Float32, 3, 3)
    rx = m * x * m'
    nx = net(reshape(Array(x), 9, 1), ps, st)[1][:]
    nrx = net(reshape(Array(rx), 9, 1), ps, st)[1][:]
    rnx = map(1:48) do n
        m = findfirst(m -> cayley[i, m] == n, 1:48)
        nx[m]
    end
    nrx - rnx
end

let
    proj = Dense(9 * 48 => 9 * 48; use_bias = false)
    proj_ps, proj_st = Lux.setup(Xoshiro(0), proj)
    proj_ps.weight .= r_sink
    function project(w)
        w = reshape(w, 9 * 48, 1)
        w = proj(w, proj_ps, proj_st) |> first
        weight = reshape(w, 9, 48)
        # No bias in last layer
        (; weight)
    end
    net = Dense(48 => 9; use_bias = false)
    ps, st = Lux.setup(Xoshiro(0), net)
    w = ps.weight
    ps = project(w)
    i = 5
    x = randn(Float32, 48)
    rx = map(1:48) do n
        m = findfirst(m -> cayley[i, m] == n, 1:48)
        x[m]
    end
    nx =
        net(reshape(x, 48, 1), ps, st)[1][:] |>
        x -> reshape(x, 3, 3) |> SMatrix{3,3,Float32,9}
    nrx =
        net(reshape(rx, 48, 1), ps, st)[1][:] |>
        x -> reshape(x, 3, 3) |> SMatrix{3,3,Float32,9}
    m = mats[i]
    rnx = m * nx * m'
    nrx - rnx
end

let
    f = f64
    proj_lift = Dense(48 * 9 => 48 * 9; use_bias = false)
    proj_sink = Dense(9 * 48 => 9 * 48; use_bias = false)
    proj_mid = Dense(48^2 => 48^2; use_bias = false)
    proj_lift_ps, _ = Lux.setup(Xoshiro(0), proj_lift) |> f
    proj_lift_ps.weight .= r_lift
    proj_sink_ps, _ = Lux.setup(Xoshiro(0), proj_sink) |> f
    proj_sink_ps.weight .= r_sink
    proj_mid_ps, _ = Lux.setup(Xoshiro(0), proj_mid) |> f
    proj_mid_ps.weight .= r_mid
    project(ps, b) = (;
        layer_1 = (;
            weight = reshape(
                proj_lift(reshape(ps.layer_1.weight, 48 * 9, 1), proj_lift_ps, (;))[1],
                48,
                9,
            ),
            bias = fill(b, 48),
        ),
        layer_2 = (;
            weight = reshape(
                proj_mid(reshape(ps.layer_2.weight, 48 * 48, 1), proj_mid_ps, (;))[1],
                48,
                48,
            ),
            bias = fill(b, 48),
        ),
        layer_3 = (;
            weight = reshape(
                proj_mid(reshape(ps.layer_3.weight, 48 * 48, 1), proj_mid_ps, (;))[1],
                48,
                48,
            ),
            bias = fill(b, 48),
        ),
        layer_4 = (;
            weight = reshape(
                proj_mid(reshape(ps.layer_4.weight, 48 * 48, 1), proj_mid_ps, (;))[1],
                48,
                48,
            ),
            bias = fill(b, 48),
        ),
        layer_5 = (;
            weight = reshape(
                proj_sink(reshape(ps.layer_5.weight, 9 * 48, 1), proj_sink_ps, (;))[1],
                9,
                48,
            ),
        ),
    )
    net = Chain(
        Dense(9 => 48, relu),
        Dense(48 => 48, relu),
        Dense(48 => 48, relu),
        Dense(48 => 48, relu),
        Dense(48 => 9),
    )
    ps, st = Lux.setup(Xoshiro(0), net) |> f
    b = 1.0 |> f
    ps = project(ps, b)
    i = 5
    mat = mats[i]
    x = @SMatrix randn(Float32, 3, 3)
    rx = mat * x * mat'
    nx = net(reshape(Array(x), 9, 1), ps, st)[1] |> x -> reshape(x, 3, 3) |> SMatrix{3,3,Float32,9}
    nrx = net(reshape(Array(rx), 9, 1), ps, st)[1] |> x -> reshape(x, 3, 3) |> SMatrix{3,3,Float32,9}
    rnx = mat * nx * mat'
    nrx - rnx
end

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
