using SymmetryCode: SymmetryCode as S
using Test

@testset "Clark closure formula" begin
    # τ = Δ²/12 * G * G^T applied componentwise; pick a simple non-symmetric G.
    g = S.Grid{2}(; l = 2π, n = 4)
    Δ = 0.5
    G = S.spacetensorfield_nonsym(g)
    fill!(G.xx, 0); fill!(G.yy, 0)
    fill!(G.xy, 2.0); fill!(G.yx, 0)
    model = S.create_clark(Δ, g)
    τ = model(nothing, G)
    # G = [0 2; 0 0]; G G' = [4 0; 0 0]; τ = Δ²/12 * [4 0; 0 0]
    @test all(τ[:, :, 1] .≈ Δ^2 / 12 * 4)  # xx
    @test all(τ[:, :, 2] .≈ 0)             # yy
    @test all(τ[:, :, 3] .≈ 0)             # xy
end

@testset "Smagorinsky closure on pure shear" begin
    # G = [0 a; 0 0]; S = (G+G')/2 = [0 a/2; a/2 0]; |S|² = a²/2;
    # √(2|S|²) = |a|; τ_ij = -2 CS² Δ² |a| S_ij.
    g = S.Grid{2}(; l = 2π, n = 4)
    Δ = 0.3
    CS = 0.17
    a = 1.7
    G = S.spacetensorfield_nonsym(g)
    fill!(G.xx, 0); fill!(G.yy, 0)
    fill!(G.xy, a); fill!(G.yx, 0)
    τ = S.create_smagorinsky(CS, Δ, g)(nothing, G)
    @test all(τ[:, :, 1] .≈ 0)
    @test all(τ[:, :, 2] .≈ 0)
    @test all(τ[:, :, 3] .≈ -2 * CS^2 * Δ^2 * abs(a) * (a / 2))
end

@testset "smagorinsky_coefficient! handles MM = 0" begin
    # When M is identically zero, the least-squares coefficient must be 0
    # rather than NaN (the bugfix we landed earlier).
    g = S.Grid{2}(; l = 2π, n = 4)
    c = S.spacescalarfield(g)
    M = S.spacetensorfield(g)
    L = S.spacetensorfield(g)
    foreach(c -> fill!(c, 0), M)
    foreach(c -> fill!(c, 1.0), L)  # arbitrary non-zero
    S.apply!(S.smagorinsky_coefficient!, g, (c, M, L, g); ndrange = S.space_ndrange(g))
    @test all(iszero, c)
end

@testset "create_verstappen rejects 2D" begin
    g = S.Grid{2}(; l = 2π, n = 4)
    @test_throws AssertionError S.create_verstappen(0.4, 0.1, g)
end
