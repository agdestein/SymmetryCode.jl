using SymmetryCode: SymmetryCode as S
using FFTW: plan_rfft
using Random: Xoshiro, randn!
using Test

@testset "twothirds! zeros high modes" begin
    g = S.Grid{2}(; l = 2π, n = 24)
    u = S.scalarfield(g)
    randn!(Xoshiro(0), u)
    S.apply!(S.twothirds!, g, (u, g))
    kcut = div(g.n, 3)
    for I in CartesianIndices(u)
        kx, ky = S.wavenumber_int(g, I)
        in_band = (kx, ky) != (0, 0) && abs(kx) ≤ kcut && abs(ky) ≤ kcut
        @test in_band || iszero(u[I])
    end
end

@testset "project! makes vector field divergence-free" begin
    for g in (S.Grid{2}(; l = 2π, n = 16), S.Grid{3}(; l = 2π, n = 8))
        u = S.vectorfield(g)
        foreach(c -> randn!(Xoshiro(1), c), u)
        S.apply!(S.project!, g, (u, g))

        div = S.scalarfield(g)
        S.apply!(S.vectordivergence!, g, (div, u, g))
        @test maximum(abs, div) < 1.0e-10
    end
end

@testset "wavenumber_full / squared_wavenumber_full" begin
    g = S.Grid{2}(; l = 4.0, n = 8)
    I = CartesianIndex(2, 1)               # kx_int = 1, ky_int = 0
    kx, ky = S.wavenumber_full(g, I)
    @test kx ≈ 2π / 4.0
    @test ky == 0
    @test S.squared_wavenumber_full(g, I) ≈ kx^2
end
