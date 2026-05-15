function setup_laptop()
    l = 1.0
    n_dns = 1024
    n_les = 64
    visc = 1.0e-5
    Δ = 3 * l / n_les
    outdir = joinpath(@__DIR__, "..", "output", "laptop") |> mkpath
    plotdir = joinpath(outdir, "plots") |> mkpath
    return (;
        name = "laptop",
        outdir,
        plotdir,
        D = 2,
        l,
        n_dns,
        n_les,
        Δ,
        visc,
        cfl = 0.35,
        warmup = (; totalenergy = 0.5, tstop = 5.0, seed = 0),
        datagen = (; nstep = 200, nsubstep = 100),
        backend = default_backend(),
    )
end

"3D forced HIT, small (n_dns=256). ~10 min on an RTX 4090; quick prototyping."
function setup_turbulator_small()
    l = 2π
    n_dns = 256
    n_les = 64
    visc = 6.0e-4
    Δ = 3 * l / n_les
    outdir =
        joinpath(@__DIR__, "..", "output", "turbulator_visc=$(visc)_n=$(n_dns)") |> mkpath
    plotdir = joinpath(outdir, "plots") |> mkpath
    return (;
        name = "turbulator",
        outdir,
        plotdir,
        D = 3,
        l,
        n_dns,
        n_les,
        Δ,
        visc,
        cfl = 0.35,
        warmup = (; totalenergy = 0.2, tstop = 10.0, seed = 0),
        datagen = (; nstep = 30, nsubstep = 60),
        backend = CUDABackend(),
    )
end

"3D forced HIT, medium (n_dns=384). Recommended default on a 24 GB GPU (RTX 4090)."
function setup_turbulator_medium()
    l = 2π
    n_dns = 384
    n_les = 64
    visc = 5.0e-4
    Δ = 3 * l / n_les
    outdir =
        joinpath(@__DIR__, "..", "output", "turbulator_visc=$(visc)_n=$(n_dns)") |> mkpath
    plotdir = joinpath(outdir, "plots") |> mkpath
    return (;
        name = "turbulator",
        outdir,
        plotdir,
        D = 3,
        l,
        n_dns,
        n_les,
        Δ,
        visc,
        cfl = 0.35,
        warmup = (; totalenergy = 0.2, tstop = 20.0, seed = 0),
        datagen = (; nstep = 50, nsubstep = 80),
        backend = CUDABackend(),
    )
end

"3D forced HIT, large (n_dns=512). Tight on a 24 GB GPU; closest to paper-quality."
function setup_turbulator_large()
    l = 2π
    n_dns = 512
    n_les = 64
    visc = 3.0e-4
    Δ = 3 * l / n_les
    outdir =
        joinpath(@__DIR__, "..", "output", "turbulator_visc=$(visc)_n=$(n_dns)") |> mkpath
    plotdir = joinpath(outdir, "plots") |> mkpath
    return (;
        name = "turbulator",
        outdir,
        plotdir,
        D = 3,
        l,
        n_dns,
        n_les,
        Δ,
        visc,
        cfl = 0.35,
        warmup = (; totalenergy = 0.2, tstop = 30.0, seed = 0),
        datagen = (; nstep = 50, nsubstep = 120),
        backend = CUDABackend(),
    )
end

function setup_snellius()
    l = 2π
    n_dns = 810
    n_les = 128
    visc = 2.0e-4
    Δ = 4 * l / n_les
    return (;
        name = "snellius",
        outdir = mkpath("/projects/prjs1757/SymmetryOutput/visc$(visc)"),
        plotdir = joinpath(@__DIR__, "..", "output", "snellius") |> mkpath,
        D = 3,
        l,
        n_dns,
        n_les,
        visc,
        Δ,
        cfl = 0.35,
        warmup = (; totalenergy = 0.2, tstop = 5.0, seed = 0),
        datagen = (; nstep = 100, nsubstep = 50),
        backend = CUDABackend(),
    )
end
