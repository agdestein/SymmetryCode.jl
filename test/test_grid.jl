using SymmetryCode: SymmetryCode as S
using Test

@testset "Grid" begin
    g2 = S.Grid{2}(; l = 2π, n = 16)
    g3 = S.Grid{3}(; l = 1.0, n = 8)

    @test S.dim(g2) == 2
    @test S.dim(g3) == 3

    @test S.tensordim(g2) == 3
    @test S.tensordim(g3) == 6

    @test S.spacing(g2) ≈ 2π / 16
    @test S.volume(g3) ≈ (1.0 / 8)^3

    @test S.ndrange(g2) == (9, 16)
    @test S.ndrange(g3) == (5, 8, 8)
    @test S.space_ndrange(g2) == (16, 16)
    @test S.space_ndrange(g3) == (8, 8, 8)

    @test_throws AssertionError S.Grid{2}(; l = 1.0, n = 7)
end
