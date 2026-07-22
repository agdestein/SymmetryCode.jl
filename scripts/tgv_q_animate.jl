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
# Each frame is normalized by a per-frame quantile of its positive Q values
# (`pnorm`), so the isosurface always encloses the same volume fraction. Q
# collapses by orders of magnitude over the decay AND its distribution changes
# shape — laminar frames are smooth (max(Q) < 3·rms, an rms-based threshold
# renders literally nothing between t ≈ 0.7 and 2.2 t_c), turbulent frames are
# heavy-tailed. The quantile threshold never blanks: the film opens on the
# smooth TGV vortex cores and morphs continuously into the turbulent worms.
# The decay still reads off the film through the structure count and size.
# Output: <outdir>/tgv_q_<mode>.mp4, so the :iso and :absorption versions can
# coexist. Styled for a light slide deck: white background, dark labels.

@info "Loading packages"
flush(stderr)

using GLMakie
using JLD2
using Statistics: quantile

anim_params() = (;
    outdir = get(ENV, "SYMMETRY_QVIZ_DIR", "/projects/prjs1757/SymmetryOutput/tgv-qviz"),
    # :iso (isosurface at the pnorm quantile) or :absorption (volumetric);
    # override with SYMMETRY_QVIZ_MODE=absorption to render the other version
    # alongside it.
    mode = Symbol(get(ENV, "SYMMETRY_QVIZ_MODE", "absorption")),
    pnorm = 0.98,       # per-frame threshold: quantile of the positive Q values
    stride = 1,         # spatial subsampling (2 → 256³ textures, much faster)
    fps = 25,   # 200 snapshots → 8 s film
    resolution = (1080, 1080),
    # Slow camera rotation over the whole film. The window [azimuth0, azimuth0 +
    # azimuth_sweep] must avoid multiples of π/2 (face-on views flatten the cube
    # to a depthless square — with the old 1.275π + π/2 sweep that happened
    # mid-film, right at peak dissipation).
    azimuth0 = 1.1π,
    azimuth_sweep = 0.3π,
    color = "#2e6f95",  # isosurface color (steel blue, reads well on white)
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

"""
Load snapshot `i`, subsample, clip to Q > 0 (rotation-dominated), and normalize
so the value 1 sits at the `pnorm` quantile of the positive values — the
isosurface then always encloses the same volume fraction (see header).
"""
function loadframe(p, i)
    q = load(qfile(p.outdir, i), "q")
    s = p.stride
    s == 1 || (q = q[1:s:end, 1:s:end, 1:s:end])
    qref = quantile(filter(>(0.0f0), vec(q)), p.pnorm)
    return max.(q, 0.0f0) ./ (qref > 0 ? Float32(qref) : 1.0f0)
end

function animate(p = anim_params())
    meta = load(qmetafile(p.outdir))
    meta["done"] || @warn "snapshot series incomplete (solve was interrupted?)"
    nsnap = length(meta["times"])

    qobs = Observable(loadframe(p, 1))

    # No text anywhere — the slide provides the caption.
    fig = Figure(; size = p.resolution, backgroundcolor = :white)
    ax = Axis3(
        fig[1, 1];
        aspect = (1, 1, 1), viewmode = :fit, protrusions = 0,
        elevation = π / 8, backgroundcolor = :white,
    )
    hidedecorations!(ax)
    hidespines!(ax)

    # Normalized frames put the threshold at 1 by construction (see loadframe).
    kwargs = if p.mode === :iso
        (;
            algorithm = :iso, isovalue = 1.0f0, isorange = 0.5f0,
            colormap = cgrad([p.color, p.color]),
        )
    elseif p.mode === :absorption
        # Alpha-ramped colormap: transparent below the threshold, so the vapor
        # reads on a white background instead of filling the cube.
        (;
            algorithm = :absorption, absorption = 6.0f0,
            colormap = cgrad(:dense, alpha = range(0, 1, length = 256)),
        )
    else
        error("unknown mode $(p.mode)")
    end
    volume!(ax, (0, 1), (0, 1), (0, 1), qobs; colorrange = (0.0f0, 2.0f0), kwargs...)

    # Faint domain box: grounds the slow camera rotation on the white background.
    boxcolor = (:gray60, 0.8)
    lines!(
        ax,
        Point3f[
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1),
        ];
        color = boxcolor, linewidth = 1,
    )
    for (a, b) in (((1, 0, 0), (1, 0, 1)), ((1, 1, 0), (1, 1, 1)), ((0, 1, 0), (0, 1, 1)))
        lines!(ax, [Point3f(a), Point3f(b)]; color = boxcolor, linewidth = 1)
    end

    outfile = joinpath(p.outdir, "tgv_q_$(p.mode).mp4")
    record(fig, outfile, 1:nsnap; framerate = p.fps) do i
        qobs[] = loadframe(p, i)
        ax.azimuth[] = p.azimuth0 + p.azimuth_sweep * (i - 1) / max(nsnap - 1, 1)
        i % 10 == 0 && (@info "rendered frame $(i)/$(nsnap)"; flush(stderr))
    end
    @info "Wrote $(outfile)"
    flush(stderr)
    return nothing
end

animate("mock" in ARGS ? anim_params_mock() : anim_params())
