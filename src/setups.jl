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
    forced = true,
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
    forced,
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
    datagen = (; nstep = 50, tstop = 10.0, n_train = 35),
    conv_setup = (; layers = [8, 16, 32], same_as_equi = false), # 824 params
    equi_setup = (; layers = [4, 8, 8]),                         # 836 params (pre-synthesis)
    tbnn_setup = (; layers = [10, 16, 32]),                      # 814 params
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
    datagen = (; nstep = 30, tstop = 5.0, n_train = 21),
    conv_setup = (; layers = [12, 24, 64], same_as_equi = false), # 2_416 params
    equi_setup = (; layers = [4, 4, 8]),                          # 2_428 params (pre-synthesis)
    tbnn_setup = (; layers = [12, 24, 64]),                       # 2_432 params
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
    warmup = (; totalenergy = 0.2, tstop = 10.0, seed = 0),
    datagen = (; nstep = 100, tstop = 15.0, n_train = 50),
    conv_setup = (; layers = [12, 24, 64], same_as_equi = false), # 2_416 params
    equi_setup = (; layers = [4, 4, 8]),                          # 2_428 params (pre-synthesis)
    tbnn_setup = (; layers = [12, 24, 64]),                       # 2_432 params
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
    warmup = (; totalenergy = 0.2, tstop = 10.0, seed = 0),
    datagen = (; nstep = 50, tstop = 10.0, n_train = 35),
    conv_setup = (; layers = [12, 24, 64], same_as_equi = false), # 2_416 params
    equi_setup = (; layers = [4, 4, 8]),                          # 2_428 params (pre-synthesis)
    tbnn_setup = (; layers = [12, 24, 64]),                       # 2_432 params
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
        warmup = (; totalenergy = 0.2, tstop = 10.0, seed = 0),

        # nstep=100 over tstop=20 covers ~2.86 turnover times (t_int ≈ 7).
        # First n_train=70 snapshots (t ∈ [0, 14] ≈ 2 t_int) are the training
        # pool; the held-out tail (t ∈ (14, 20] ≈ 0.86 t_int) drives all
        # post-hoc analysis (LES rollout, a-priori metrics, densities, Q-R).
        datagen = (; nstep = 100, tstop = 20.0, n_train = 70), # 25685 seconds
        conv_setup = (; layers = [44, 64, 64], same_as_equi = false), # 7_864 params
        equi_setup = (; layers = [4, 8, 16]),                         # 7_888 params (pre-synthesis)
        tbnn_setup = (; layers = [46, 64, 64]),                       # 7_892 params
    )
    outdir = "/projects/prjs1757/SymmetryOutput/visc=$(s.visc)_n=$(s.n_dns)" |> mkpath
    plotdir = joinpath(@__DIR__, "..", "output", "snellius") |> mkpath
    return getsetup(; s..., outdir, plotdir)
end

"""
Derive a decaying Taylor-Green vortex *test* setup from a 3D training setup
`train`. The closure-relevant parameters (`visc`, `n_les`, the filter width `Δ`,
`l`, and the per-model `*_setup`) are copied verbatim so the trained models
apply unchanged; only the flow changes — initially laminar, transitioning to
turbulence, then decaying with no forcing (`forced = false`).

The initial amplitude is `V0 = Re_target * visc` so the case sits at the
canonical `Re_target` benchmark (default 1600, with `L = 1`, so
`Re = V0 L / visc = Re_target`). The DNS runs for `tconv` convective times
`t_c = L / V0 = 1 / V0` (the vortex peaks near `t* = 9` and has largely decayed
by `t* = 20`). The saved snapshots cover essentially the whole transition.

Models are built from `train` (which owns the `ps-*.jld2`); this setup only owns
the Taylor-Green `data.jld2` and the post-hoc artifacts under its own `outdir`.
The extra fields `V0`/`Re_target` are consumed by [`create_data_tgv`](@ref) and
the dissipation-benchmark plot.
"""
function setup_taylorgreen(
        train;
        Re_target = 1600,
        nstep = 100,
        tconv = 20,
        outdir = joinpath(
            @__DIR__, "..", "output", "tgv_$(train.name)_Re=$(Re_target)_n=$(train.n_dns)",
        ) |> mkpath,
        plotdir = joinpath(outdir, "plots") |> mkpath,
    )
    (;
        D, l, n_dns, n_les, visc, cfl, backend, warmup,
        tbnn_setup, equi_setup, conv_setup, train_setup,
    ) = train
    @assert D == 3 "setup_taylorgreen expects a 3D training setup"
    V0 = Re_target * visc
    tstop = tconv / V0
    Δ_factor = train.Δ / (l / n_les) # Recover the filter-width factor used in training
    base = getsetup(;
        name = "tgv_$(train.name)",
        D,
        l,
        n_dns,
        n_les,
        Δ_factor,
        visc,
        cfl,
        warmup,
        datagen = (; nstep, tstop, n_train = 1),
        tbnn_setup,
        equi_setup,
        conv_setup,
        train_setup,
        backend,
        forced = false,
        outdir,
        plotdir,
    )
    return (; base..., V0, Re_target)
end

"""
Build the inference closures listed in `models` against eval `setup`, returned as
a NamedTuple keyed by `modelname(m)`. Learned entries are model coordinates
`(; arch, tier, netseed, use_redelta)`, loaded via [`build_model`](@ref) from
`psfile`; classical entries are symbols built against a fresh LES grid. (`:convsym`
and the seed sweep are handled by the eval driver, not here.)
"""
function build_models(case, setup, models)
    g = Grid{case.D}(; case.l, n = case.n_les, case.backend)
    classical = (;
        nomo = () -> (_, _) -> fill!(stack(spacetensorfield(g)), 0),
        dynsmag = () -> create_dynamic_smagorinsky(setup.Δ, g),
        clar = () -> create_clark(setup.Δ, g),
        smag = () -> create_smagorinsky(0.1, setup.Δ, g),
        vers = () -> create_verstappen(1.0, setup.Δ, g),
        bard = () -> create_bardina(2.0, setup.Δ, g),
    )
    return NamedTuple(
        (m isa NamedTuple ? modelname(m) => build_model(case, m, setup) : m => classical[m]())
            for m in models
    )
end
