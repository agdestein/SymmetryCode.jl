# Decaying Taylor-Green DNS → Q-criterion snapshot series, for a 3D
# visualization video. Standalone: no LES, no closures, no coupling to the
# `case`/coordinate pipeline — one DNS march that extracts and stores Q
# (Float32, full resolution) at equispaced save times, plus a light metadata
# file. `scripts/tgv_q_animate.jl` turns the series into the film.
#
# Setup (see `qviz_params`): 512³, Re = 3000 (the resolution-matched analogue of
# the production Re-6000 / 810³ TGV: Re scales as n^(4/3), and 6000·(512/810)^(4/3)
# ≈ 3200), 200 snapshots over 20 convective times t_c = L/V0 (peak dissipation
# lands around t ≈ 9 t_c; 0.1 t_c cadence keeps the film smooth at 20 fps),
# plus small solenoidal seed noise so the TGV symmetries actually break (see
# `noise` in `qviz_params`).
# Storage: 200 · 512³ · 4 B ≈ 107 GB. Compute is Float64 throughout; only the
# stored Q is cast to Float32. The per-snapshot `kmax_eta` in the log verifies
# the Re choice a posteriori (want ≳ 1 at peak dissipation).
#
# Run on the cluster (fits comfortably in the default 2 h H100 job):
#
#     sbatch job.sh scripts/tgv_q_solve.jl
#
# Output (override the root with ENV["SYMMETRY_QVIZ_DIR"]):
#     <outdir>/q_0001.jld2 … q_<nsnap>.jld2   keys: q (Float32 n³), t, i
#     <outdir>/qmeta.jld2                     times, statistics, params, done
#
# The run is cache-guarded as a whole (a DNS march cannot skip ahead, so there
# is no per-snapshot resume): if `qmeta.jld2` says `done` and all snapshot files
# exist, the script is a no-op.

@info "Loading packages"
flush(stderr)

using CUDA
using JLD2
using Random: Xoshiro
import SymmetryCode as S

# Knobs as functions, not `const`s, so the script re-`include`s cleanly.
qviz_params() = (;
    n = 512,        # DNS grid (dealiased kmax = n/3)
    Re = 3000,      # nominal Re = V0·L/ν with L = 1 (l = 2π)
    visc = 5.0e-4,  # → V0 = Re·ν = 1.5, same amplitude as the production TGV
    cfl = 0.35,
    tconv = 20,     # convective times t_c = L/V0 = 1/V0 spanned
    nsnap = 200,    # 0.1 t_c cadence
    # Symmetry-breaking seed noise. The clean TGV lies in an invariant subspace
    # of its mirror/rotation symmetry group and the spectral discretization
    # preserves it exactly, so without noise the only breaking is Float64
    # round-off — invisible even after 20 t_c (the production Re-6000 slice
    # movie stays mirror-tiled throughout). A solenoidal perturbation of rms
    # `noise·V0` (energy ratio ~½noise² vs V0²/8, i.e. 4e-4 at 1e-2) is
    # amplified through the shear-layer instabilities and visibly breaks the
    # kaleidoscope around peak dissipation. Set `noise = 0.0` for the pure
    # symmetric (canonical) TGV.
    noise = 1.0e-2,
    noise_kpeak = 10,   # seed spectrum peak_profile(k; kpeak): small scales, big vortices untouched
    noise_seed = 0,
    outdir = get(ENV, "SYMMETRY_QVIZ_DIR", "/projects/prjs1757/SymmetryOutput/tgv-qviz"),
)

qfile(outdir, i) = joinpath(outdir, "q_$(lpad(i, 4, '0')).jld2")
qmetafile(outdir) = joinpath(outdir, "qmeta.jld2")

"""
Q-criterion `Q = ½(‖Ω‖² − ‖S‖²) = −½ ∂ᵢuⱼ ∂ⱼuᵢ` of the spectral velocity `u`,
accumulated into the physical scalar `Q` term by term: each velocity-gradient
component is formed in spectral space (`derivative!`), ghost modes zeroed
(`twothirds!`, as in `filter_reynolds`), and transformed to a physical scalar —
so peak scratch is one spectral + two physical scalars, never the 9-component
gradient tensor. Scratch aliases time-stepping cache buffers (`σ.xx` spectral,
`v.x`/`v.y` physical), which the next march overwrites anyway.
"""
function compute_q!(Q, u, g, c)
    spec = c.σ.xx
    a, b = c.v.x, c.v.y
    fill!(Q, 0)
    for i in 1:3    # diagonal terms (∂ᵢuᵢ)²
        S.apply!(S.derivative!, g, (spec, u[i], i, g))
        S.apply!(S.twothirds!, g, (spec, g))
        S.to_phys!(a, spec, c.plan, g)
        @. Q -= a^2 / 2
    end
    for (i, j) in ((1, 2), (2, 3), (3, 1))    # off-diagonal pairs ∂ⱼuᵢ·∂ᵢuⱼ, twice each
        S.apply!(S.derivative!, g, (spec, u[i], j, g))
        S.apply!(S.twothirds!, g, (spec, g))
        S.to_phys!(a, spec, c.plan, g)
        S.apply!(S.derivative!, g, (spec, u[j], i, g))
        S.apply!(S.twothirds!, g, (spec, g))
        S.to_phys!(b, spec, c.plan, g)
        @. Q -= a * b
    end
    return Q
end

function qviz_done(p)
    f = qmetafile(p.outdir)
    isfile(f) || return false
    jldopen(f, "r") do file
        haskey(file, "done") && file["done"]
    end || return false
    return all(i -> isfile(qfile(p.outdir, i)), 1:p.nsnap)
end

function solve_qviz(p = qviz_params())
    (; n, Re, visc, cfl, tconv, nsnap, noise, noise_kpeak, noise_seed, outdir) = p
    mkpath(outdir)
    l = 2π
    V0 = Re * visc
    tstop = tconv / V0    # t_c = L/V0 with L = 1

    @info "Q-criterion TGV DNS: n=$(n)³, Re=$(Re) (ν=$(visc), V0=$(V0)), " *
        "$(nsnap) snapshots over $(tconv) t_c = $(round(tstop; sigdigits = 4)) time units"
    flush(stderr)

    g = S.Grid{3}(; l, n, backend = S.default_backend())
    c = S.getcache(g)
    sc = S.statscache(g)
    u = S.taylorgreen(g, c.plan; V0)
    if noise > 0
        # Both fields are solenoidal spectral fields, so the sum is too.
        pert = S.randomfield(
            S.peak_profile, g;
            totalenergy = (noise * V0)^2 / 2, rng = Xoshiro(noise_seed),
            kpeak = noise_kpeak,
        )
        foreach((a, b) -> a .+= b, u, pert)
        pert = nothing
        @info "Added symmetry-breaking noise: rms $(noise)·V0 at kpeak = $(noise_kpeak)"
        flush(stderr)
    end
    Q = S.spacescalarfield(g)

    times = Float64[]
    stat0 = S.turbulence_statistics(u, visc, g, sc)
    statistics = typeof(stat0)[]

    savetimes = range(0.0, tstop, length = nsnap)
    walltime = time()
    t = 0.0
    nstep = 0
    for (i, tnext) in enumerate(savetimes)
        i == 1 || while t < tnext
            Δt = cfl * S.propose_timestep(u, g, visc, c)
            Δt = min(Δt, tnext - t)
            t += Δt
            nstep += 1
            S.wray3!(S.convectiondiffusion!, u, Δt, g, c; visc)    # decaying, no forcing
        end

        stat = S.turbulence_statistics(u, visc, g, sc)
        push!(times, t)
        push!(statistics, stat)

        compute_q!(Q, u, g, c)
        S.jldsave_atomic(qfile(outdir, i); q = Float32.(Array(Q)), t, i)

        # Metadata rewritten every snapshot, so a killed job leaves a readable
        # (partial, `done = false`) series behind.
        S.jldsave_atomic(
            qmetafile(outdir);
            times, statistics, l, n, Re, visc, V0, tconv, nsnap, nstep,
            noise, noise_kpeak, noise_seed,
            walltime = time() - walltime, done = i == nsnap,
        )

        @info "snapshot $(i)/$(nsnap)  t = $(round(t; sigdigits = 4)) " *
            "($(round(t * V0; digits = 2)) t_c)  step $(nstep)  " *
            "Re_λ = $(round(stat.Re_tay; digits = 1))  " *
            "kmax_η = $(round(stat.kmax_eta; digits = 2))"
        flush(stderr)
    end

    @info "Finished after $(round(time() - walltime; sigdigits = 4)) s, $(nstep) steps"
    flush(stderr)
    return nothing
end

if qviz_done(qviz_params())
    @info "Q snapshot series already complete — nothing to do"
else
    solve_qviz()
end
