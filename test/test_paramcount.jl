using SymmetryCode: SymmetryCode as S
using KernelAbstractions: CPU
using Test

# The closed-form `paramcount` (read straight off the layer widths) must equal
# the count of the actually-built `ps` (`learned_paramcounts`), so the formula
# can't silently drift from the architectures in `nets.jl`. Built on CPU; the
# `Chain` summaries that net construction prints are expected noise here.

@testset "closed-form paramcount == as-built ps (D = $D)" for D in (2, 3)
    # A couple of shapes per dimension, including a deeper net and the
    # capacity-matched conv (`same_as_equi`) branch the shipped setups skip.
    # Widths are kept small so the oracle build (which `paramcount` avoids)
    # stays cheap — especially the |G|-scaled `same_as_equi` conv.
    configs = (
        (; equi = [4, 4, 8], conv = [16, 32, 64], tbnn = [16, 32, 64], sae = false),
        (; equi = [8, 8, 8, 16], conv = [4, 8, 8], tbnn = [64, 64, 128], sae = true),
    )
    for cfg in configs
        setup = (;
            D,
            l = 2π,
            n_les = 64,
            backend = CPU(),
            train_setup = S.default_train_setup(),
            tbnn_setup = (; layers = cfg.tbnn),
            equi_setup = (; layers = cfg.equi),
            conv_setup = (; layers = cfg.conv, same_as_equi = cfg.sae),
        )
        oracle = S.learned_paramcounts(setup, [:tbnn, :equi, :conv])
        for key in (:tbnn, :equi, :conv)
            @test S.paramcount(setup, key) == oracle[key]
        end
    end
end

# A few fixed reference configs (deliberately decoupled from the shipped,
# frequently-tuned `*_setup` sizes) so a formula regression is obvious at a
# glance — closed-form only, no net is built.
@testset "paramcount reference values" begin
    setup = (;
        D = 3, l = 2π, n_les = 128, backend = CPU(),
        train_setup = S.default_train_setup(),
        tbnn_setup = (; layers = [64, 64, 128]),
        equi_setup = (; layers = [8, 8, 8, 16]),
        conv_setup = (; layers = [48, 64, 64, 64], same_as_equi = false),
    )
    @test S.paramcount(setup, :equi) == 12_544
    @test S.paramcount(setup, :tbnn) == 13_760
    @test S.paramcount(setup, :conv) == 12_320
end
