if false
    include("src/SymmetryCode.jl")
    using .SymmetryCode
    using .SymmetryCode.Spectral
end

using LinearAlgebra
using Seneca
using SymmetryCode
using SymmetryCode.Spectral

g = Grid{2}(; l = 1.0, n = 16)
gbar = Grid{2}(; l = 1.0, n = 8)

u = Seneca.scalarfield(g)
ubar = Seneca.scalarfield(gbar)
a, b = zeros(9), reshape(1:16, 1, :)
# a, b = 1:9, zeros(1, 16)
u .= a .+ b
apply!(Spectral.cutoff!, gbar, (ubar, u, gbar, g))
ubar .|> Int
u .|> Int

Seneca.FFTW.fftfreq(8, 8) .|> Int

# DNS-aided LES
let
    visc = 4e-4
    t = 0.0
    cfl = 0.85
    tstop = 1e-1
    D = 2
    g = Grid{D}(; l = 1.0, n = 16)
    gbar = Grid{D}(; l = 1.0, n = 8)
    u = randomfield(g; kpeak = 5);
    ubar = vectorfield(gbar)
    foreach(i -> apply!(Spectral.cutoff!, gbar, (ubar[i], u[i], gbar, g)), 1:D)
    v = map(copy, ubar)
    fσ = tensorfield(gbar)
    σf = tensorfield(gbar)
    c = getcache(g);
    cbar = getcache(gbar);
    i = 0
    while t < tstop
        i += 1
        Δt = cfl * propose_timestep(u, c, visc, g)
        Δt = min(Δt, tstop - t)
        t += Δt
        @info "t = $t, Δt = $Δt"
        # DNS
        stress!(c.σ, c.vi_vj, c.v, u, c.plan, visc, g)
        apply!(tensordivergence!, g, (c.du, c.σ, g))
        # LES
        foreach(i -> apply!(Spectral.cutoff!, gbar, (ubar[i], u[i], gbar, g)), 1:D)
        stress!(σf, cbar.vi_vj, cbar.v, ubar, cbar.plan, visc, gbar)
        stress!(cbar.σ, cbar.vi_vj, cbar.v, v, cbar.plan, visc, gbar)
        foreach(i -> apply!(Spectral.cutoff!, gbar, (fσ[i], c.σ[i], gbar, g)), 1:tensordim(g))
        foreach(i -> (cbar.σ[i] .+= fσ[i] .- σf[i]), 1:tensordim(g))
        apply!(tensordivergence!, gbar, (cbar.du, cbar.σ, gbar))
        # Step
        for i = 1:dim(g)
            axpy!(Δt, c.du[i], u[i])
            axpy!(Δt, cbar.du[i], v[i])
        end
        apply!(project!, g, (u, g))
        apply!(project!, gbar, (v, gbar))
    end
    foreach(i -> apply!(Spectral.cutoff!, gbar, (ubar[i], u[i], gbar, g)), 1:D)
    sum(i -> sum(abs2, v[i] - ubar[i]) / sum(abs2, ubar[i]), 1:D)
end
