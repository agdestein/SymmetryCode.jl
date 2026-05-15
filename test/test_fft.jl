using SymmetryCode: SymmetryCode as S
using Random: Xoshiro, randn!
using Test

@testset "to_phys! / to_spec! roundtrip" begin
    for g in (S.Grid{2}(; l = 2π, n = 16), S.Grid{3}(; l = 1.0, n = 8))
        plan = S.getplan(g)
        phys = S.spacescalarfield(g)
        spec = S.scalarfield(g)
        out = S.spacescalarfield(g)

        randn!(Xoshiro(0), phys)
        original = copy(phys)

        S.to_spec!(spec, phys, plan, g)
        S.to_phys!(out, spec, plan, g)
        @test out ≈ original

        # Constant field: the k=0 coefficient picks up the l^D scaling.
        fill!(phys, 1.5)
        S.to_spec!(spec, phys, plan, g)
        @test spec[1] ≈ 1.5
    end
end

@testset "dealias_phys! idempotent" begin
    g = S.Grid{3}(; l = 2π, n = 8)
    plan = S.getplan(g)
    phys = S.spacescalarfield(g)
    spec = S.scalarfield(g)
    randn!(Xoshiro(1), phys)

    S.dealias_phys!(phys, spec, plan, g)
    once = copy(phys)
    S.dealias_phys!(phys, spec, plan, g)
    @test phys ≈ once
end
