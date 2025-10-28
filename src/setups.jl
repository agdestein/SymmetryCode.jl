export setup_laptop
function setup_laptop()
    l = 1.0
    n_les = 256
    Δ = 4 * l / n_les
    outdir = joinpath(@__DIR__, "..", "output", "laptop") |> mkpath
    plotdir = joinpath(outdir, "plots") |> mkpath
    (;
        name = "laptop",
        visc = 1e-5,
        outdir,
        plotdir,
        D = 2,
        l = 1.0,
        n_dns = 2048,
        n_les,
        kpeak = 5,
        totalenergy = 1.0,
        t_warmup = 0.5,
        Δ,
        ou_radius = 2.3,
        ou_time = 0.005,
        ou_energy = 0.01,
        # force = 8.1 => 0.1,
        force = 8.1 => nothing,
        backend = CUDABackend(),
    )
end

export setup_turbulator
function setup_turbulator()
    l = 1.0
    # n_dns = 32
    # n_dns = 256
    n_dns = 512
    n_les = 64
    Δ = 4 * l / n_les
    outdir = joinpath(@__DIR__, "..", "output", "turbulator$(n_dns)") |> mkpath
    # plotdir = "~/Projects/SymmetryPaper/figures" |> expanduser |> mkpath
    plotdir = joinpath(outdir, "plots") |> mkpath
    (;
        name = "turbulator",
        outdir,
        plotdir,
        visc = 1e-4,
        D = 3,
        l = 1.0,
        n_dns,
        n_les,
        kpeak = 2,
        totalenergy = 1.0,
        t_warmup = 0.1,
        Δ,
        # force = 8.1 => 0.3,
        # force = 8.1 => nothing,
        force = 16.1 => nothing,
        backend = CUDABackend(),
    )
end

export setup_snellius
function setup_snellius()
    l = 1.0
    n_les = 128
    Δ = 4 * l / n_les
    (;
        name = "snellius",
        # outdir = mkpath("/projects/prjs1757/SymmetryOutput"),
        outdir = mkpath("/projects/prjs1757/SymmetryOutput2"),
        plotdir = joinpath(@__DIR__, "..", "output", "snellius") |> mkpath,
        visc = 1e-4,
        D = 3,
        l = 1.0,
        n_dns = 810,
        n_les,
        kpeak = 5,
        Δ,
        ou_radius = 2.3,
        ou_time = 0.005,
        ou_energy = 0.01,
        backend = CUDABackend(),
    )
end
