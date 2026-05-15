using SymmetryCode: SymmetryCode as S
using Random: Xoshiro, randn!
using Test

@testset "make_tracefree!" begin
    g2 = S.Grid{2}(; l = 2π, n = 8)
    τ2 = S.spacetensorfield(g2)
    foreach(c -> randn!(Xoshiro(0), c), τ2)
    S.make_tracefree!(τ2, g2)
    @test maximum(abs, @. τ2.xx + τ2.yy) < 1.0e-12

    g3 = S.Grid{3}(; l = 1.0, n = 6)
    τ3 = S.spacetensorfield(g3)
    foreach(c -> randn!(Xoshiro(1), c), τ3)
    S.make_tracefree!(τ3, g3)
    @test maximum(abs, @. τ3.xx + τ3.yy + τ3.zz) < 1.0e-12

    # xy / yz / zx components are untouched
    raw = S.spacetensorfield(g3)
    foreach(c -> randn!(Xoshiro(2), c), raw)
    snap = map(copy, raw)
    S.make_tracefree!(raw, g3)
    @test raw.xy == snap.xy
    @test raw.yz == snap.yz
    @test raw.zx == snap.zx
end

@testset "unstack_symtensor" begin
    g2 = S.Grid{2}(; l = 2π, n = 4)
    y2 = randn(Xoshiro(0), 4, 4, S.tensordim(g2))
    t2 = S.unstack_symtensor(y2, g2)
    @test t2.xx == y2[:, :, 1]
    @test t2.yy == y2[:, :, 2]
    @test t2.xy == y2[:, :, 3]

    # Views must alias the source array
    t2.xx[1, 1] = 42
    @test y2[1, 1, 1] == 42

    g3 = S.Grid{3}(; l = 1.0, n = 4)
    y3 = randn(Xoshiro(1), 4, 4, 4, S.tensordim(g3))
    t3 = S.unstack_symtensor(y3, g3)
    @test t3.zz == y3[:, :, :, 3]
    @test t3.yz == y3[:, :, :, 5]
end

@testset "strain_from_gradient is symmetric part of G" begin
    g = S.Grid{3}(; l = 2π, n = 4)
    G = S.spacetensorfield_nonsym(g)
    foreach(c -> randn!(Xoshiro(0), c), G)
    Sg = S.strain_from_gradient(G, g)
    @test Sg.xx == G.xx
    @test Sg.yy == G.yy
    @test Sg.zz == G.zz
    @test Sg.xy ≈ (G.xy .+ G.yx) ./ 2
    @test Sg.yz ≈ (G.yz .+ G.zy) ./ 2
    @test Sg.zx ≈ (G.zx .+ G.xz) ./ 2
end

@testset "contract_dissipation matches manual sum" begin
    g = S.Grid{2}(; l = 2π, n = 8)
    τ = S.spacetensorfield(g)
    Sg = S.spacetensorfield(g)
    foreach(c -> randn!(Xoshiro(0), c), τ)
    foreach(c -> randn!(Xoshiro(1), c), Sg)
    d = S.contract_dissipation(τ, Sg, g)
    @test d ≈ @. τ.xx * Sg.xx + τ.yy * Sg.yy + 2 * τ.xy * Sg.xy

    g3 = S.Grid{3}(; l = 2π, n = 4)
    τ3 = S.spacetensorfield(g3)
    Sg3 = S.spacetensorfield(g3)
    foreach(c -> randn!(Xoshiro(2), c), τ3)
    foreach(c -> randn!(Xoshiro(3), c), Sg3)
    d3 = S.contract_dissipation(τ3, Sg3, g3)
    @test d3 ≈ @. τ3.xx * Sg3.xx + τ3.yy * Sg3.yy + τ3.zz * Sg3.zz +
        2 * (τ3.xy * Sg3.xy + τ3.yz * Sg3.yz + τ3.zx * Sg3.zx)
end
