if false
    include("src/SymmetryCode.jl")
    using .SymmetryCode
end

using KernelAbstractions
using LinearAlgebra
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
    t = reverse(t; dims)
    m = roto_reflection_matrix(p, s)
    t = map(t -> m * t * m', t)
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
