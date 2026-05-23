"""
Shared training configuration (mirrors the per-model `*_setup.layers`).

- `precision` is the network's working float type (`Float32` recommended);
  the solver/data stay `Float64` and conversion happens at the net boundary.
- `val_fraction` is the *time-based* held-out tail of snapshots.
- `checkpoint_every` steps a resumable checkpoint is written (SLURM-safe).
"""
default_train_setup(;
    nepoch = 20,
    batchsize = 20,
    nsample = 50,
    learning_rate = 1.0e-3,
    seed = 0,
    val_fraction = 0.2,
    log_every = 10,
    precision = Float32,
    grad_clip = 1.0,
    patience = 5,
    warmup_frac = 0.05,
    checkpoint_every = 200,
) = (;
    nepoch,
    batchsize,
    nsample,
    learning_rate,
    seed,
    val_fraction,
    log_every,
    precision,
    grad_clip,
    patience,
    warmup_frac,
    checkpoint_every,
)

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
    train_setup = default_train_setup(),
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
    train_setup,
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
    tbnn_setup = (; layers = [16, 32, 64]), # 3_200 params (3D)
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
    tbnn_setup = (; layers = [16, 32, 64]), # 3_200 params (3D)
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
    tbnn_setup = (; layers = [16, 32, 64]), # 3_200 params (3D)
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
    tbnn_setup = (; layers = [16, 32, 64]), # 3_200 params (3D)
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
        tbnn_setup = (; layers = [64, 64, 128]), # 13_760 params
        equi_setup = (; layers = [9, 8, 8, 16]), # 12_544 actual params
        conv_setup = (; layers = [48, 64, 64, 64], same_as_equi = false), # 12_320 parameters
    )
    outdir = "/projects/prjs1757/SymmetryOutput/visc=$(s.visc)_n=$(s.n_dns)" |> mkpath
    plotdir = joinpath(@__DIR__, "..", "output", "snellius") |> mkpath
    return getsetup(; s..., outdir, plotdir)
end

"""
Build only the closure models named in `active`, in the requested order.

Trainable closures load their parameters via `create_*(setup, train_mode)`
(`:skip` reads `ps-<key>.jld2` without retraining); classical closures are
constructed against a fresh LES `Grid`. Keys not in `active` are never
instantiated.
"""
function build_models(setup, active; train_mode = :skip)
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    build = (;
        nomo = () -> (_, _) -> fill!(stack(spacetensorfield(g)), 0),
        dynsmag = () -> create_dynamic_smagorinsky(setup.Δ, g),
        clar = () -> create_clark(setup.Δ, g),
        smag = () -> create_smagorinsky(0.1, setup.Δ, g),
        vers = () -> create_verstappen(1.0, setup.Δ, g),
        bard = () -> create_bardina(2.0, setup.Δ, g),
        tbnn = () -> create_tbnn(setup, train_mode)[1],
        equi = () -> create_equi(setup, train_mode)[1],
        conv = () -> create_conv(setup, train_mode)[1],
    )
    return NamedTuple(k => build[k]() for k in active)
end
