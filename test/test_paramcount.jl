using SymmetryCode: SymmetryCode as S
using KernelAbstractions: CPU
using Test

# The closed-form `paramcount(case, m)` (read straight off the tier's layer widths)
# must equal the number of parameters in the actually-built `ps`, so the formula
# can't silently drift from the architectures in `nets.jl` — including the +1
# Re_Δ input channel. Built on CPU; the `Chain` summaries that net construction
# prints are expected noise here.

# Recursively count array entries in a (nested) parameter NamedTuple/Tuple.
count_params(x::AbstractArray) = length(x)
count_params(x::NamedTuple) = sum(count_params, values(x); init = 0)
count_params(x::Tuple) = sum(count_params, x; init = 0)
count_params(::Any) = 0

# Minimal `case`-shaped view carrying only what `make_netsetup` / `paramcount` read.
make_case(D, tiers) = (;
    D, l = 2π, n_les = 64, backend = CPU(),
    schedule = (; seed = 0, precision = Float32),
    tiers,
)

@testset "closed-form paramcount == as-built ps (D = $D)" for D in (2, 3)
    # A couple of shapes per dimension, including a deeper net. Widths are kept
    # small so the oracle build (which `paramcount` avoids) stays cheap.
    tiers = (;
        a = (; tbnn = [16, 32, 64], equi = [4, 4, 8], conv = [16, 32, 64]),
        b = (; tbnn = [64, 64, 128], equi = [8, 8, 8, 16], conv = [4, 8, 8]),
    )
    case = make_case(D, tiers)
    for tier in keys(tiers), arch in (:tbnn, :equi, :conv), use_redelta in (false, true)
        m = (; arch, tier, netseed = 0, use_redelta)
        ns = S.build_net_stuff(
            S.make_netsetup(case, m.netseed), m.arch, case.tiers[tier][arch]; m.use_redelta,
        )
        @test S.paramcount(case, m) == count_params(ns.ps)
    end
end

# A few fixed reference values (deliberately decoupled from the shipped tiers) so
# a formula regression is obvious at a glance — closed-form only, no net is built.
@testset "paramcount reference values" begin
    case = make_case(
        3,
        (; ref = (; tbnn = [64, 64, 128], equi = [8, 8, 8, 16], conv = [48, 64, 64, 64])),
    )
    m(arch, ur) = (; arch, tier = :ref, netseed = 0, use_redelta = ur)
    # use_redelta = false: the architectures are unchanged from before the feature.
    @test S.paramcount(case, m(:equi, false)) == 12_544
    @test S.paramcount(case, m(:tbnn, false)) == 13_760
    @test S.paramcount(case, m(:conv, false)) == 12_320
    # use_redelta = true: one extra standardized log Re_Δ input channel.
    @test S.paramcount(case, m(:equi, true)) == 12_552   # + weight_re (c₁ scalars)
    @test S.paramcount(case, m(:tbnn, true)) == 13_824   # + nchan[1] (one input row)
    @test S.paramcount(case, m(:conv, true)) == 12_368   # + nchan[1] (one input row)
end
