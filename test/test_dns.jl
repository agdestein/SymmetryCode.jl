using SymmetryCode: SymmetryCode as S
using FFTW: plan_rfft
using LinearAlgebra: norm
using Random: Xoshiro
using Test

# The truncation-aware DNS path (src/dns.jl) against the full-array solver.

relerr(a, b) = norm(a .- b) / max(norm(b), eps())
relerr(a::NamedTuple, b::NamedTuple) = maximum(map(relerr, a, b))

testgrids() = (S.Grid{2}(; l = 2π, n = 24), S.Grid{3}(; l = 2π, n = 12))
profile(::S.Grid{2}) = S.linear_profile_2D
profile(::S.Grid{3}) = S.linear_profile_3D

"Mean-free, 2/3-truncated, solenoidal full-array random field."
fullfield(g; seed = 1) = S.randomfield(profile(g), g; rng = Xoshiro(seed))

@testset "layout: pad! and truncate! round-trip" begin
    for g in testgrids()
        d = S.dnssetup(g)
        u = fullfield(g)
        uc = S.truncate!(S.compactvectorfield(g), u, d)
        @test size(uc.x) == S.compact_ndrange(g)
        v = S.pad!(S.vectorfield(g), uc, d)
        @test relerr(v, u) == 0
        @test S.compact_energy(uc) ≈ S.energy(u) rtol = 1.0e-13
    end
end

@testset "transforms match to_phys! and round-trip" begin
    for g in testgrids()
        d = S.dnssetup(g)
        plan = plan_rfft(S.spacescalarfield(g))
        u = fullfield(g)
        uc = S.truncate!(S.compactvectorfield(g), u, d)
        v = S.dns_to_phys!(S.spacevectorfield(g), uc, d)
        vref = map(c -> S.to_phys!(S.spacescalarfield(g), copy(c), plan, g), u)
        @test relerr(v, vref) < 1.0e-13
        wc = S.compactvectorfield(g)
        foreach((a, b) -> S.dns_to_spec!(a, b, d), wc, v)
        @test relerr(wc, uc) < 1.0e-13
        @test S.dns_vmax(uc, d) ≈ maximum(sum(c -> abs.(c), vref)) rtol = 1.0e-13
    end
end

@testset "convective RHS matches convectiondiffusion! on retained modes" begin
    for g in testgrids()
        d = S.dnssetup(g)
        u = fullfield(g)
        cache = S.getcache(g)
        du = S.vectorfield(g)
        S.convectiondiffusion!(du, u, g, cache; visc = 0.0)
        S.apply!(S.project!, g, (du, g))
        uc = S.truncate!(S.compactvectorfield(g), u, d)
        duc = S.dns_rhs!(S.compactvectorfield(g), uc, d)
        S.dns_project!(duc, d)
        ref = S.truncate!(S.compactvectorfield(g), du, d)
        @test relerr(duc, ref) < 1.0e-12
    end
end

@testset "inviscid IF-RK3 step equals wray3! to round-off" begin
    for g in testgrids()
        d = S.dnssetup(g)
        u = fullfield(g)
        uc = S.truncate!(S.compactvectorfield(g), u, d)
        cache, dc = S.getcache(g), S.dnscache(d)
        Δt = 0.5 * S.dns_timestep(S.dns_vmax(uc, d), d)
        for _ in 1:3
            S.wray3!(S.convectiondiffusion!, u, Δt, g, cache; visc = 0.0)
            foreach(c -> S.apply!(S.twothirds!, g, (c, g)), u)
            S.dns_step!(uc, Δt, d, dc)
        end
        ref = S.truncate!(S.compactvectorfield(g), u, d)
        @test relerr(uc, ref) < 1.0e-12
    end
end

@testset "integrating factor: a shear wave decays exactly" begin
    # u_x = cos(k₀ y) is a steady Euler solution, so only the linear part acts
    g = S.Grid{2}(; l = 2π, n = 24)
    k0 = 3
    for linop in (nothing, k2 -> 0.02 * k2 + 1.0e-6 * k2^4)
        d = S.dnssetup(g; visc = 0.05, linop)
        uc = S.compactvectorfield(g)
        uc.x[1, k0 + 1] = 0.5
        uc.x[1, d.m + 1 - k0] = 0.5 # compact index of ky = -k0
        u0 = copy(uc.x)
        rate = isnothing(linop) ? 0.05 * k0^2 : linop(k0^2)
        t, dc = 0.0, S.dnscache(d)
        for Δt in (0.1, 0.37, 0.05, 0.2) # arbitrary, far above any viscous limit
            S.dns_step!(uc, Δt, d, dc)
            t += Δt
        end
        @test relerr(uc.x, u0 .* exp(-rate * t)) < 1.0e-13
        @test maximum(abs, uc.y) < 1.0e-14
    end
end

@testset "viscous IF-RK3 converges to the explicit solver" begin
    g = S.Grid{2}(; l = 2π, n = 24)
    visc = 0.01
    u0 = fullfield(g; seed = 3)
    d = S.dnssetup(g; visc)
    function run_full(Δt, nstep)
        u, cache = map(copy, u0), S.getcache(g)
        for _ in 1:nstep
            S.wray3!(S.convectiondiffusion!, u, Δt, g, cache; visc)
            foreach(c -> S.apply!(S.twothirds!, g, (c, g)), u)
        end
        return S.truncate!(S.compactvectorfield(g), u, d)
    end
    function run_if(Δt, nstep)
        uc, dc = S.truncate!(S.compactvectorfield(g), u0, d), S.dnscache(d)
        foreach(_ -> S.dns_step!(uc, Δt, d, dc), 1:nstep)
        return uc
    end
    ref = run_full(1.0e-4, 2000)
    e1 = relerr(run_if(1.0e-2, 20), ref)
    e2 = relerr(run_if(5.0e-3, 40), ref)
    @test e2 < e1 / 6 # third order: ≈ 8
end

@testset "compact shell forcing matches the full-array version" begin
    for g in testgrids()
        d = S.dnssetup(g)
        u = fullfield(g)
        uc = S.truncate!(S.compactvectorfield(g), u, d)
        full = S.energy_shells(g, [1, 2], u)
        comp = S.compact_energy_shells(g, [1, 2], uc)
        @test all(((a, b),) -> a.eref ≈ b.eref, zip(full, comp))
        foreach(c -> (c .*= 1.3), uc)
        S.maintain_shell_energy!(uc, comp)
        e = map(s -> sum(c -> sum(abs2, view(c, s.energyinds)), uc) / 2, comp)
        @test all(((a, b),) -> a ≈ b.eref, zip(e, comp))
    end
end

@testset "Float32 stepping keeps its type" begin
    g = S.Grid{2}(; l = 2.0f0 * π, n = 24)
    d = S.dnssetup(g; visc = 0.01f0)
    uc = S.truncate!(S.compactvectorfield(g), fullfield(g), d)
    vmax = S.dns_step!(uc, 1.0f-3, d)
    @test eltype(uc.x) == ComplexF32
    @test vmax isa Float32
    @test S.dns_timestep(vmax, d) isa Float32
end
