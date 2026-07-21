# Render the Q-criterion snapshot series from `scripts/tgv_q_solve.jl` into a
# 3D video (GLMakie `volume`, isosurface or volumetric absorption). Standalone:
# needs only GLMakie + JLD2, not SymmetryCode.
#
# GLMakie is not a SymmetryCode dependency — add it once:
#
#     julia --project -e 'using Pkg; Pkg.add("GLMakie")'
#
# GLMakie needs OpenGL. Two routes that work:
#
# 1. Locally (macOS/Linux with a real GPU): downsample on the cluster first
#    (`scripts/tgv_q_downsample.jl`, 512³ → 256³, ~13 GB), rsync the `small/`
#    dir home, then
#        SYMMETRY_QVIZ_DIR=~/tgv-qviz-small julia --project scripts/tgv_q_animate.jl
#
# 2. Full resolution on Snellius: a `gpu_vis` remote-visualization desktop
#    (VNC / Open OnDemand; may need separate access), hardware GL via
#        vglrun julia --project scripts/tgv_q_animate.jl
#    https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660253
#
# Known-bad on Snellius: xvfb-run on the login/compute nodes. The Xvfb+Mesa GLX
# stack itself is fine (glxinfo renders via llvmpipe), but julia's GLFW
# segfaults in it (JLL X11 libs vs system GLX; verified 2026-07 with a minimal
# GLFW window test) — and llvmpipe would be far too slow for 512³ volumes
# anyway. GLMakie also cannot precompile without a display (GLFW init runs at
# module load), hence LocalPreferences.toml carries
# `[GLMakie] precompile_workload = false` where GLMakie is installed.
#
# Each frame is normalized by its own rms(Q) before thresholding: Q collapses by
# orders of magnitude over the decay, so a fixed threshold would blank out either
# the transition or the tail. The decay still reads off the film through the
# structure count and size. Output: <outdir>/tgv_q.mp4.

@info "Loading packages"
flush(stderr)

using GLMakie
using JLD2
using Statistics: std

anim_params() = (;
    outdir = get(ENV, "SYMMETRY_QVIZ_DIR", "/projects/prjs1757/SymmetryOutput/tgv-qviz"),
    mode = :iso,        # :iso (isosurface at qthresh) or :absorption (volumetric)
    qthresh = 3.0,      # threshold in units of the frame's rms(Q)
    stride = 1,         # spatial subsampling (2 → 256³ textures, much faster)
    fps = 20,
    resolution = (1080, 1080),
    azimuth_sweep = π / 2,    # slow camera rotation over the whole film
)

# Renders the mock series from `tgv_q_solve.jl mock` (its `mock/` subdir).
# Select with `julia --project scripts/tgv_q_animate.jl mock`.
anim_params_mock() = (;
    anim_params()...,
    outdir = joinpath(anim_params().outdir, "mock"),
    resolution = (540, 540),
)

qfile(outdir, i) = joinpath(outdir, "q_$(lpad(i, 4, '0')).jld2")
qmetafile(outdir) = joinpath(outdir, "qmeta.jld2")

"Load snapshot `i`, subsample, clip to Q > 0 (rotation-dominated), rms-normalize."
function loadframe(p, i)
    q = load(qfile(p.outdir, i), "q")
    s = p.stride
    s == 1 || (q = q[1:s:end, 1:s:end, 1:s:end])
    r = std(q)
    return max.(q, 0.0f0) ./ (r > 0 ? r : one(r))
end

function animate(p = anim_params())
    meta = load(qmetafile(p.outdir))
    meta["done"] || @warn "snapshot series incomplete (solve was interrupted?)"
    times, V0, Re = meta["times"], meta["V0"], meta["Re"]
    nsnap = length(times)

    qobs = Observable(loadframe(p, 1))
    title = Observable("t = 0.0 t_c")

    fig = Figure(; size = p.resolution, backgroundcolor = :black)
    ax = Axis3(
        fig[1, 1];
        aspect = (1, 1, 1), viewmode = :fit, protrusions = 0,
        elevation = π / 8, backgroundcolor = :black,
    )
    hidedecorations!(ax)
    hidespines!(ax)
    Label(
        fig[1, 1, Top()], title;
        color = :white, fontsize = 26, padding = (0, 0, 10, 0),
    )
    Label(
        fig[1, 1, Bottom()], "Decaying Taylor-Green vortex, Re = $(Re) — Q-criterion";
        color = (:white, 0.6), fontsize = 20, padding = (0, 0, 0, 10),
    )

    kwargs = if p.mode === :iso
        (;
            algorithm = :iso,
            isovalue = Float32(p.qthresh),
            isorange = Float32(p.qthresh / 4),
        )
    elseif p.mode === :absorption
        (; algorithm = :absorption, absorption = 6.0f0)
    else
        error("unknown mode $(p.mode)")
    end
    volume!(
        ax, (0, 1), (0, 1), (0, 1), qobs;
        colormap = :inferno, colorrange = (0.0f0, Float32(2 * p.qthresh)),
        kwargs...,
    )

    azimuth0 = 1.275π
    outfile = joinpath(p.outdir, "tgv_q.mp4")
    record(fig, outfile, 1:nsnap; framerate = p.fps) do i
        qobs[] = loadframe(p, i)
        title[] = "t = $(round(times[i] * V0; digits = 1)) t_c"
        ax.azimuth[] = azimuth0 + p.azimuth_sweep * (i - 1) / max(nsnap - 1, 1)
        i % 10 == 0 && (@info "rendered frame $(i)/$(nsnap)"; flush(stderr))
    end
    @info "Wrote $(outfile)"
    flush(stderr)
    return nothing
end

animate("mock" in ARGS ? anim_params_mock() : anim_params())
