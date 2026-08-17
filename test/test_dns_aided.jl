using SymmetryCode: SymmetryCode as S
using Random: Xoshiro, randn!
using Test

# Restored 2026-08-19 from `src/verify.jl`, function `dns_aid` (lines 344-389),
# deleted 2026-07-05 in "Clean up unused files" and recoverable at
# `git show 298c3e3^:src/verify.jl`. It lives under test/ now for one reason: a
# verification that passes looks exactly like unused code, and this one was
# deleted while the claim it establishes was still load-bearing.
#
# WHAT IT CHECKS. The SFS stress is defined as the *filtered DNS stress* minus
# the *discrete coarse-grid stress of the filtered field* -- the second term
# formed by the same routine the coarse solver calls, not by a continuous
# product. If that definition is right, it does not merely approximate the
# coarse dynamics, it closes them EXACTLY: step a DNS and a "DNS-aided" LES
# forward with the same integrator and the same dt, inject the stress at every
# step, and the LES state stays equal to the filtered DNS to round-off, for as
# many steps as you care to run. The identity is algebraic --
#
#     stress(v) + [F(sigma_dns) - stress(ubar)] = F(sigma_dns)   when v == ubar
#
# -- and holds for any linear Fourier-multiplier filter F, because F then
# commutes with the divergence, with the Leray projection and with a linear time
# step taken identically on both grids. The viscous halves cancel for the same
# reason. So the check is a genuine test of the code, not of the algebra: it
# fails if the filter chain, the coarse nonlinearity or the injection is wired
# up differently from the derivation.
#
# THE EXTENSION. The original ran a sharp `cutoff!` alone. The production
# pipeline (`sfs!`) applies a composite -- cutoff, then a Gaussian, then the 2/3
# dealiasing truncation -- and it is the *dealiasing* that the exactness
# argument actually leans on, since that is what makes the retained band of a
# pseudo-spectral product equal to the Galerkin product. `dealias = true` is
# therefore the case worth having: the coarse state and the injected stress are
# restricted to the 2/3 cube, and the coarse tendency is truncated the way
# `les.jl` truncates it, so the ghost modes above the cube never feed back.

"""
Step a DNS and a DNS-aided LES forward together and return the relative L2
error `‖v - ubar‖ / ‖ubar‖` between the LES state and the filtered DNS.

`Δ` adds a Gaussian of that width to the coarse filter (`nothing` = sharp
cutoff only); `dealias` restricts the coarse state, the injected stress and the
coarse tendency to the 2/3 cube. `(Δ = nothing, dealias = false)` is the
original check; `(Δ = 0.1, dealias = true)` is the production filter chain.

Returns the error itself -- the original returned its square.

Two departures from the deleted version, both forced by it returning `NaN` when
restored verbatim. It seeded the DNS with `randn!` straight into the spectral
array, which is white noise in k and reaches |u| ~ 200 in physical space; run
forward with explicit Euler that overflows, and the error becomes `Inf - Inf`.
So the initial condition is now `randomfield` with the k^(-5/3) profile that
`data.jl` uses in production, and the run is a fixed, short step count. Neither
weakens the check: the identity is per-step, so a mis-wired filter chain or
injection shows on the first step and cannot hide for twenty.
"""
function dns_aided_error(;
    D = 3,
    n_dns = 16,
    n_les = 8,
    visc = 4.0e-4,
    cfl = 0.5,
    nstep = 20,
    totalenergy = 0.2,
    Δ = nothing,
    dealias = false,
    break_filter = false,
    seed = 0,
)
    rng = Xoshiro(seed)
    g = S.Grid{D}(; l = 1.0, n = n_dns)
    gbar = S.Grid{D}(; l = 1.0, n = n_les)

    # Coarse-graining: the SAME operator is applied to the velocity and to the
    # DNS stress. That is the whole content of the identity above; applying two
    # different filters is the mistake this test would catch.
    function coarsen!(dst, src; gauss = true)
        S.apply!(S.cutoff!, gbar, (dst, src))
        (isnothing(Δ) || !gauss) || S.apply!(S.gaussianfilter!, gbar, (dst, Δ, gbar))
        dealias && S.apply!(S.twothirds!, gbar, (dst, gbar))
        return dst
    end

    # Production initial condition (`data.jl` uses the same call): a k^(-5/3)
    # spectrum, Hermitian-symmetrised and made divergence-free inside.
    u = S.randomfield(S.linear_profile_3D, g; rng, totalenergy)

    ubar = S.vectorfield(gbar)
    foreach(i -> coarsen!(ubar[i], u[i]), 1:D)
    v = map(copy, ubar)                                    # the LES starts on the filtered DNS

    fσ = S.tensorfield(gbar)      # F(sigma_dns)
    σf = S.tensorfield(gbar)      # stress(ubar), the DISCRETE coarse stress
    c = S.getcache(g)
    cbar = S.getcache(gbar)

    for _ in 1:nstep
        Δt = cfl * S.propose_timestep(u, g, visc, c)

        # DNS tendency.
        S.stress!(c.σ, c.vi_vj, c.v, u, c.plan, visc, g)
        S.apply!(S.tensordivergence!, g, (c.du, c.σ, g))

        # LES tendency: own coarse stress, plus the injected SFS stress.
        foreach(i -> coarsen!(ubar[i], u[i]), 1:D)
        S.stress!(σf, cbar.vi_vj, cbar.v, ubar, cbar.plan, visc, gbar)
        S.stress!(cbar.σ, cbar.vi_vj, cbar.v, v, cbar.plan, visc, gbar)
        foreach(i -> coarsen!(fσ[i], c.σ[i]; gauss = !break_filter), 1:S.tensordim(g))
        foreach(1:S.tensordim(g)) do i
            τ = fσ[i] .- σf[i]
            dealias && S.apply!(S.twothirds!, gbar, (τ, gbar))   # the stored tau is truncated
            cbar.σ[i] .+= τ
        end
        S.apply!(S.tensordivergence!, gbar, (cbar.du, cbar.σ, gbar))
        # les.jl truncates the closure term before it acts; ghost modes above the
        # cube are re-zeroed rather than integrated.
        dealias && foreach(i -> S.apply!(S.twothirds!, gbar, (cbar.du[i], gbar)), 1:D)

        # One explicit Euler step on each grid, same Δt, then project both.
        for i in 1:D
            u[i] .+= Δt .* c.du[i]
            v[i] .+= Δt .* cbar.du[i]
        end
        S.apply!(S.project!, g, (u, g))
        S.apply!(S.project!, gbar, (v, gbar))
    end

    foreach(i -> coarsen!(ubar[i], u[i]), 1:D)
    # A blown-up DNS would give Inf - Inf = NaN and read as a failure of the
    # identity rather than of the integrator, so say which one it is.
    all(x -> all(isfinite, x), u) || error("DNS diverged: the integrator, not the stress definition")
    num = sum(i -> sum(abs2, v[i] .- ubar[i]), 1:D)
    den = sum(i -> sum(abs2, ubar[i]), 1:D)
    return sqrt(num / den)
end

@testset "DNS-aided LES: the SFS stress closes the coarse equations exactly" begin
    # The original check: sharp cutoff, no dealiasing.
    e_cutoff = dns_aided_error(; Δ = nothing, dealias = false)
    @test e_cutoff < 1.0e-12

    # The production convention: 2/3 dealiasing on the coarse state, the
    # injected stress and the coarse tendency. This is the case the deleted
    # check never exercised, and the one the alias-free argument rests on.
    e_dealias = dns_aided_error(; Δ = nothing, dealias = true)
    @test e_dealias < 1.0e-12

    # The full production filter chain: cutoff, Gaussian, dealias.
    e_full = dns_aided_error(; Δ = 0.1, dealias = true)
    @test e_full < 1.0e-12

    # NEGATIVE CONTROL. Everything above passes at 1e-16, which is worth
    # nothing until the same harness is shown to fail when the definition is
    # wrong. Coarse-grain the two halves of tau with DIFFERENT operators --
    # Gaussian on the velocity, none on the DNS stress -- and the error must
    # jump by orders of magnitude. Without this, a harness that silently
    # compared a field with itself would look like a triumph.
    e_mismatch = dns_aided_error(; Δ = 0.1, dealias = true, break_filter = true)
    @test e_mismatch > 1.0e-3

    @info "DNS-aided LES relative errors" e_cutoff e_dealias e_full e_mismatch
end
