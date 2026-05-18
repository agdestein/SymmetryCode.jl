"""
Problem setup.
The filter width `Δ` is `Δ_factor` multiplied by the grid spacing.
The fields `outdir` and `plotdir` are generated automatically.
"""
getsetup(;
    name,
    D,
    l,
    n_dns,
    n_les,
    Δ_factor,
    visc,
    cfl,
    warmup,
    datagen,
    tbnn_setup,
    equi_setup,
    conv_setup,
    backend = default_backend(),
    outdir = joinpath(@__DIR__, "..", "output", "$(name)_visc=$(visc)_n=$(n_dns)") |> mkpath,
    plotdir = joinpath(outdir, "plots") |> mkpath,
) = (;
    name,
    D,
    l,
    n_dns,
    n_les,
    Δ = Δ_factor * l / n_les,
    visc,
    cfl,
    warmup,
    datagen,
    tbnn_setup,
    equi_setup,
    conv_setup,
    backend,
    outdir,
    plotdir,
)

"2D forced HIT, small (n_dns=512). For quick prototyping on a laptop."
setup_laptop() = getsetup(;
    name = "laptop",
    D = 2,
    l = 2π,
    n_dns = 512,
    n_les = 64,
    Δ_factor = 3,
    visc = 1.0e-4,
    cfl = 0.35,
    warmup = (; totalenergy = 0.2, tstop = 10.0, seed = 0),
    datagen = (; nstep = 50, tstop = 10.0),
    tbnn_setup = (;), # Eventually put setup here
    equi_setup = (; layers = [4, 4, 4, 8]),
    conv_setup = (; layers = [16, 32, 64], same_as_equi = false),
)

"3D forced HIT, small (n_dns=256). ~10 min on an RTX 4090; quick prototyping."
setup_turbulator_small() = getsetup(;
    name = "turbulator_small",
    D = 3,
    l = 2π,
    n_dns = 256,
    n_les = 64,
    Δ_factor = 3,
    visc = 7.0e-4,
    cfl = 0.35,
    warmup = (; totalenergy = 0.2, tstop = 10.0, seed = 0),
    datagen = (; nstep = 30, tstop = 5.0),
    tbnn_setup = (;), # Eventually put setup here
    equi_setup = (; layers = [4, 4, 4, 8]), # 3_200 actual params
    conv_setup = (; layers = [16, 32, 64], same_as_equi = false), # 3_200 parameters
)

"3D forced HIT, medium (n_dns=384). Recommended default on a 24 GB GPU (RTX 4090)."
setup_turbulator_medium() = getsetup(;
    name = "turbulator_medium",
    D = 3,
    l = 2π,
    n_dns = 384,
    n_les = 64,
    Δ_factor = 3,
    visc = 5.0e-4,
    cfl = 0.35,
    warmup = (; totalenergy = 0.2, tstop = 20.0, seed = 0),
    datagen = (; nstep = 50, tstop = 10.0),
    tbnn_setup = (;), # Eventually put setup here
    equi_setup = (; layers = [4, 4, 4, 8]), # 3_200 actual params
    conv_setup = (; layers = [16, 32, 64], same_as_equi = false), # 3_200 parameters
)

"3D forced HIT, large (n_dns=512). Tight on a 24 GB GPU; closest to paper-quality."
setup_turbulator_large() =
    getsetup(;
    name = "turbulator_large",
    D = 3,
    l = 2π,
    n_dns = 512,
    n_les = 64,
    Δ_factor = 3,
    visc = 3.0e-4,
    cfl = 0.35,
    warmup = (; totalenergy = 0.2, tstop = 30.0, seed = 0),
    datagen = (; nstep = 50, tstop = 10.0),
    tbnn_setup = (;), # Eventually put setup here
    equi_setup = (; layers = [4, 4, 4, 8]), # 3_200 actual params
    conv_setup = (; layers = [16, 32, 64], same_as_equi = false), # 3_200 parameters
)

"3D forced HIT, large (n_dns=810). Fits in a 90 GB datacenter GPU (H100 on Snellius)."
function setup_snellius()
    s = (;
        name = "snellius",
        D = 3,
        l = 2π,
        n_dns = 810,
        n_les = 128,
        Δ_factor = 3,
        visc = 1.5e-4,
        cfl = 0.35,
        warmup = (; totalenergy = 0.2, tstop = 40.0, seed = 0),
        datagen = (; nstep = 50, tstop = 10.0),
        tbnn_setup = (;), # Eventually put setup here
        equi_setup = (; layers = [9, 8, 8, 16]), # 12_544 actual params
        conv_setup = (; layers = [48, 64, 64, 64], same_as_equi = false), # 12_320 parameters
    )
    outdir = "/projects/prjs1757/SymmetryOutput/visc=$(s.visc)_n=$(s.n_dns)" |> mkpath
    plotdir = joinpath(@__DIR__, "..", "output", "snellius") |> mkpath
    return getsetup(; s..., outdir, plotdir)
end
