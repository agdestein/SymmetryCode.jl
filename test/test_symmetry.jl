using SymmetryCode: SymmetryCode as S
using LinearAlgebra: I, det, eigen, norm
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

@testset "equivariant weight synthesis (D = $D)" for D in (2, 3)
    gs = S.group_stuff(D)
    nreg = length(gs.elements)
    nten = D^2

    (; r_lift, r_mid, r_sink) = S.get_weight_projectors(D)
    (; relidx, Qmat, s_lift, s_mid, s_sink) = S.get_weight_synthesis(D)

    # Synthesis basis columns are orthonormal (so they preserve the eigenbasis
    # init scaling) and live in the eigenvalue-one subspace of the projectors.
    for (r, s) in ((r_lift, s_lift), (r_mid, s_mid), (r_sink, s_sink))
        @test s' * s ≈ I                       # orthonormal columns
        @test r / nreg * s ≈ s                 # fixed by the Reynolds projector
    end

    # Same subspace as the eigendecomposition path: orthogonal projectors agree.
    e_lift = eigen(r_lift / nreg; sortby = -).vectors[:, 1:nten]
    e_mid = eigen(r_mid / nreg; sortby = -).vectors[:, 1:nreg]
    e_sink = eigen(r_sink / nreg; sortby = -).vectors[:, 1:nten]
    @test s_lift * s_lift' ≈ e_lift * e_lift'
    @test s_mid * s_mid' ≈ e_mid * e_mid'
    @test s_sink * s_sink' ≈ e_sink * e_sink'

    # Qmat is the tensor block of r_lift: Qmat[h][(x,y),(i,j)] = R[x,i] R[y,j],
    # with (x,y) flattened x-fastest (= the Conv weight's μ order).
    for (h, R) in enumerate(gs.mats)
        Q = zeros(Int, nten, nten)
        for j in 1:D, i in 1:D, y in 1:D, x in 1:D
            Q[x + (y - 1) * D, i + (j - 1) * D] = R[x, i] * R[y, j]
        end
        @test Qmat[h] == Q
    end

    # relidx is the group-circulant index n⁻¹·m.
    @test relidx == [gs.cayley[gs.inverse_indices[n], m] for m in 1:nreg, n in 1:nreg]

    # Bit-exact weight-level equivariance: synthesizing a mid block from random
    # learnables gives a weight that is *exactly* group-circulant — w[g·m, g·n]
    # == w[m,n] to the last bit (no tolerance), which the eigenbasis cannot do.
    cout, cin = 3, 2
    k = randn(Xoshiro(0), nreg, cout, cin)
    wmid = reshape(s_mid * reshape(k, nreg, :), nreg, nreg, cout, cin)
    for g in eachindex(gs.elements)
        @test wmid[gs.cayley[g, :], gs.cayley[g, :], :, :] == wmid
    end

    # Lift block is exactly the Q-orbit of its identity row: w[h] == Qmat[h]·c.
    c = randn(Xoshiro(1), nten, 4)
    wlift = reshape(s_lift * c, nreg, nten, 4)
    c0 = wlift[gs.unitindex, :, :]                # = Qmat[e]·c / norm = c·α
    for h in eachindex(gs.elements)
        @test wlift[h, :, :] == Qmat[h] * c0
    end
end
