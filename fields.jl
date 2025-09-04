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
    m = roto_reflection_matrix(p, s)
    t = map(t -> m * t * m', t)
end
