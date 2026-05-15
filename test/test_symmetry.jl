using SymmetryCode: SymmetryCode as S
using LinearAlgebra: I, det
using Test

@testset "group_stuff structure" begin
    # 2D: 2 axis permutations × 4 sign flips = 8 (octahedral / dihedral D_4 group)
    gs2 = S.group_stuff(2)
    @test length(gs2.elements) == 8
    @test length(gs2.mats) == 8
    @test all(m -> abs(det(m)) ≈ 1, gs2.mats)
    @test gs2.unitindex == findfirst(==(I), gs2.mats)

    # 3D: 6 × 8 = 48 (octahedral group O_h)
    gs3 = S.group_stuff(3)
    @test length(gs3.elements) == 48
    @test all(m -> abs(det(m)) ≈ 1, gs3.mats)

    # Cayley table closure: every product is an existing element
    @test all(0 .< gs3.cayley .≤ length(gs3.elements))

    # Inverses really invert
    for i in eachindex(gs3.elements)
        @test gs3.mats[gs3.inverse_indices[i]] * gs3.mats[i] ≈ I
    end
end
