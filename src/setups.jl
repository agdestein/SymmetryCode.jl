export setup_laptop
function setup_laptop()
    l = 2π
    n_dns = 512
    n_les = 128
    visc = 1e-4
    Δ = 3 * l / n_les
    outdir = joinpath(@__DIR__, "..", "output", "laptop") |> mkpath
    plotdir = joinpath(outdir, "plots") |> mkpath
    (;
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
        warmup = (; totalenergy = 0.2, tstop = 5.0, seed = 0),
        datagen = (; nstep = 100, nsubstep = 25),
        backend = CPU(),
    )
end

export setup_turbulator
function setup_turbulator()
    # l = 1.0
    l = 2π
    # n_dns = 32
    # n_dns = 256
    n_dns = 512
    n_les = 64
    visc = 3e-4
    Δ = 3 * l / n_les
    outdir =
        joinpath(@__DIR__, "..", "output", "turbulator_visc=$(visc)_n=$(n_dns)") |> mkpath
    # plotdir = "~/Projects/SymmetryPaper/figures" |> expanduser |> mkpath
    plotdir = joinpath(outdir, "plots") |> mkpath
    (;
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
        warmup = (; totalenergy = 0.2, tstop = 2.0, seed = 0),
        datagen = (; nstep = 100, nsubstep = 25),
        backend = CUDABackend(),
    )
end

export setup_snellius
function setup_snellius()
    l = 2π
    n_dns = 810
    n_les = 128
    visc = 2e-4
    Δ = 4 * l / n_les
    (;
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
