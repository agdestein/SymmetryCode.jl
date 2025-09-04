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

rng = Xoshiro(0)

let
    f = f32
    proj_lift = Dense(48 * 9 => 48 * 9; use_bias = false)
    proj_sink = Dense(9 * 48 => 9 * 48; use_bias = false)
    proj_mid = Dense(48^2 => 48^2; use_bias = false)
    proj_lift_ps, _ = Lux.setup(rng, proj_lift) |> f
    proj_lift_ps.weight .= r_lift
    proj_sink_ps, _ = Lux.setup(rng, proj_sink) |> f
    proj_sink_ps.weight .= r_sink
    proj_mid_ps, _ = Lux.setup(rng, proj_mid) |> f
    proj_mid_ps.weight .= r_mid
    project(ps, b) = (;
        layer_1 = (;
            weight = reshape(
                proj_lift(reshape(ps.layer_1.weight, 48 * 9, 1), proj_lift_ps, (;))[1],
                48,
                9,
            ),
            bias = fill(b.layer_1, 48),
        ),
        layer_2 = (;
            weight = reshape(
                proj_mid(reshape(ps.layer_2.weight, 48 * 48, 1), proj_mid_ps, (;))[1],
                48,
                48,
            ),
            bias = fill(b.layer_2, 48),
        ),
        layer_3 = (;
            weight = reshape(
                proj_mid(reshape(ps.layer_3.weight, 48 * 48, 1), proj_mid_ps, (;))[1],
                48,
                48,
            ),
            bias = fill(b.layer_3, 48),
        ),
        layer_4 = (;
            weight = reshape(
                proj_mid(reshape(ps.layer_4.weight, 48 * 48, 1), proj_mid_ps, (;))[1],
                48,
                48,
            ),
            bias = fill(b.layer_4, 48),
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
        Dense(9 => 48, gelu),
        Dense(48 => 48, gelu),
        Dense(48 => 48, gelu),
        Dense(48 => 48, gelu),
        Dense(48 => 9),
    )
    ps, st = Lux.setup(rng, net) |> f
    T = eltype(ps.layer_1.weight)
    a = T(0)
    b = (;
        layer_1 = a * randn(rng, T),
        layer_2 = a * randn(rng, T),
        layer_3 = a * randn(rng, T),
        layer_4 = a * randn(rng, T),
    )
    ps = project(ps, b)
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
end

# With multiple channels
let
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
        layer_1 = project_lift(ps.layer_1),
        layer_2 = project_mid(ps.layer_2),
        layer_3 = project_mid(ps.layer_3),
        layer_4 = project_mid(ps.layer_4),
        layer_5 = project_sink(ps.layer_5),
    )
    net = Chain(
        Dense(9 => 48 * 10, gelu),
        Dense(48 * 10 => 48 * 10, gelu),
        Dense(48 * 10 => 48 * 20, gelu),
        Dense(48 * 20 => 48 * 20, gelu),
        Dense(48 * 20 => 9),
    )
    ps, st = Lux.setup(rng, net) |> f
    T = eltype(ps.layer_1.weight)
    a = T(0)
    ps = (;
        layer_1 = (; ps.layer_1.weight, bias = a * randn(rng, T)),
        layer_2 = (; ps.layer_2.weight, bias = a * randn(rng, T)),
        layer_3 = (; ps.layer_3.weight, bias = a * randn(rng, T)),
        layer_4 = (; ps.layer_4.weight, bias = a * randn(rng, T)),
        layer_5 = (; ps.layer_5.weight),
    )
    ps = project(ps)
    i = 11
    mat = mats[i]
    x = @SMatrix(randn(3, 3)) |> f
    rx = mat * x * mat'
    nx =
        net(reshape(Array(x), 9, 1), ps, st)[1] |> x -> reshape(x, 3, 3) |> SMatrix{3,3,eltype(x),9}
    nrx =
        net(reshape(Array(rx), 9, 1), ps, st)[1] |>
        x -> reshape(x, 3, 3) |> SMatrix{3,3,eltype(x),9}
    rnx = mat * nx * mat'
    nrx - rnx |> display
    nothing
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
