# Downsample the Q snapshot series for rendering off-cluster: stride 2 turns
# 512³ frames (107 GB for 200 snapshots) into 256³ (13 GB) — small enough to
# rsync to a laptop and render there with hardware GL, where GLMakie just works.
# Pure JLD2/array ops, no GL, no GPU; run anywhere the series lives:
#
#     julia --project scripts/tgv_q_downsample.jl          # production series
#     julia --project scripts/tgv_q_downsample.jl mock     # mock series
#
# Output: <outdir>/small/ with the same q_%04d.jld2 / qmeta.jld2 schema, so the
# animate script runs on it unchanged — locally:
#
#     rsync -av snellius:/projects/prjs1757/SymmetryOutput/tgv-qviz/small/ ~/tgv-qviz-small/
#     SYMMETRY_QVIZ_DIR=~/tgv-qviz-small julia --project scripts/tgv_q_animate.jl

@info "Loading packages"
flush(stderr)

using JLD2

qviz_outdir() = get(ENV, "SYMMETRY_QVIZ_DIR", "/projects/prjs1757/SymmetryOutput/tgv-qviz")

down_params() = (;
    stride = 2,
    indir = qviz_outdir(),
)

qfile(dir, i) = joinpath(dir, "q_$(lpad(i, 4, '0')).jld2")
qmetafile(dir) = joinpath(dir, "qmeta.jld2")

function downsample_qviz(p = down_params())
    (; stride, indir) = p
    outdir = mkpath(joinpath(indir, "small"))
    meta = load(qmetafile(indir))
    nsnap = length(meta["times"])
    meta["done"] || @warn "snapshot series incomplete (found $(nsnap) snapshots)"
    for i in 1:nsnap
        isfile(qfile(outdir, i)) && continue
        q, t = load(qfile(indir, i), "q", "t")
        jldsave(qfile(outdir, i); q = q[1:stride:end, 1:stride:end, 1:stride:end], t, i)
        @info "downsampled $(i)/$(nsnap)"
        flush(stderr)
    end
    # Same metadata, with the grid size corrected for the animate title.
    meta["n"] = cld(meta["n"], stride)
    jldsave(qmetafile(outdir); (Symbol(k) => v for (k, v) in meta)...)
    @info "Wrote $(nsnap) downsampled snapshots to $(outdir)"
    flush(stderr)
    return nothing
end

downsample_qviz(
    "mock" in ARGS ?
        (; down_params()..., indir = joinpath(qviz_outdir(), "mock")) :
        down_params(),
)
