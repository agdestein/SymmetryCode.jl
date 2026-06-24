# experiment.jl — coordinate-driven configuration for the Re_Δ experiment.
#
# Design (Notes/ReExperiment.md, "clean-slate refactor"): the swept axes —
# viscosity, DNS seed, filter ratio, network architecture/size/init-seed — are
# *loose coordinates*, never fused into a monolithic config. A `case` holds only
# what is shared across the whole sweep. `make_setup(case, dns, Δ_factor)` derives
# the per-(ν, Δ) `setup` view (the NamedTuple the rest of the pipeline
# destructures) **once**, from coordinates — never rebuilt from another setup.
# Pure path functions locate every artifact from its coordinates, so no block has
# to destructure-and-rebuild a setup to find its inputs.
#
# This is the live configuration layer: `make_setup` and the path functions drive
# data generation, training, evaluation, and the plots for both the forced-HIT
# runs (`dns_runs`) and the decaying Taylor-Green test (`tgv_runs`).

"""
Fixed-budget training schedule shared by every learned closure. The nets are tiny
and train fast, so there is **no early stopping, patience, or file
checkpointing**; `val_fraction` is a *monitoring-only* holdout for the convergence
curve, never for model selection. The network-init seed is a coordinate
(`model.netseed`), not part of the schedule.
"""
default_schedule(;
    nepoch = 20,
    batchsize = 20,
    nsample = 50,
    learning_rate = 1.0e-3,
    val_fraction = 0.2,
    warmup_frac = 0.05,
    grad_clip = 1.0,
    log_every = 10,
    precision = Float32,
) = (;
    nepoch, batchsize, nsample, learning_rate,
    val_fraction, warmup_frac, grad_clip, log_every, precision,
)

"""
Per-size-tier network widths, **parameter-matched across the three models** so the
capacity comparison isolates inductive bias rather than raw capacity. Widths are
placeholders pending the Phase-0b capacity sweep; use `paramcount(case, m)` to
check the match across the three architectures.
"""
default_tiers() = (;
    small = (; tbnn = [10, 16, 32], equi = [4, 4, 8], conv = [8, 16, 32]),
    medium = (; tbnn = [12, 24, 64], equi = [4, 8, 8], conv = [12, 24, 64]),
    saturated = (; tbnn = [46, 64, 64], equi = [4, 8, 16], conv = [44, 64, 64]),
)

"""
Everything shared across the whole ν/Δ/seed sweep — never the swept axes
themselves. `rootdir` is the (cluster) artifact root; `plotdir` collects every
figure. Sampling is given in integral turnovers; the actual save times are
resolved against the *measured* turnover at data generation. Train uses a few
sparse snapshots over ~2 turnovers (a-priori diversity); test a denser series
over ~1 turnover (a-posteriori reference; Clark blows up past ~1 turnover).
"""
function case_snellius(;
        backend = default_backend(),
        rootdir = "/projects/prjs1757/SymmetryOutput/redelta",
        plotdir = joinpath(@__DIR__, "..", "output", "redelta") |> mkpath,
    )
    return (;
        name = "snellius",
        D = 3,
        l = 2π,
        n_dns = 810,
        n_les = 128,
        cfl = 0.35,
        forced = true,
        totalenergy = 0.2,
        warmup_tstop = 0.05,
        train_sampling = (; nsnap = 8, nturnover = 0.05),
        test_sampling = (; nsnap = 40, nturnover = 1),
        tgv_sampling = (; nsnap = 100, tconv = 20),   # decaying TGV: span tconv convective times
        filters_train = [2.0, 3.0, 4.0],          # Δ/h; window [2, 5] (ReExperiment.md §B)
        filters_test = [2.5, 3.5, 5.0],           # interp {2.5, 3.5} + extrap {5}
        tiers = default_tiers(),
        schedule = default_schedule(),
        backend, rootdir, plotdir,
    )
end

# --- sweep coordinates: explicit lists / zips, never a product of setups ---

"""
The DNS runs of the experiment as explicit `(; visc, seed, role)` coordinates.
ν is held at well-resolved values for training (kmax_η ≥ 1.5 at n_dns=810) and
pushed toward the resolution floor only for the high-Re OOD *test* (kmax_η≈1.1,
where the ~40η-wide filter keeps the saved ūbar/τ targets reliable). The training
ν's break the Re_Δ ∝ |Ā| collinearity that single-Re data cannot (ReExperiment.md
§2, [[redelta-gate0-confound]]).
"""
function dns_runs()
    train = [(; visc, seed = 1, role = :train) for visc in (1.5e-4, 2.5e-4, 4.0e-4)]
    test = [
        (; visc = 2.5e-4, seed = 2, role = :test_indist),   # new seed, trained ν
        (; visc = 1.0e-4, seed = 3, role = :test_ood),      # higher Re, held-out ν
    ]
    return (; train, test, all = [train; test])
end

"""
The decaying Taylor-Green vortex test run(s): the trained forced-HIT closures are
applied to a transitioning-then-decaying TGV at `Re_target` (initial amplitude
`V0 = Re_target·ν`, with `L = 1` so `Re = Re_target`). The IC is deterministic, so
`seed` only keeps the artifact path distinct from the forced runs. Reuses the case
grid and test filters, so the whole `predict_sfs` / `solve_les` / plot pipeline
applies unchanged on the `(tgv, Δf)` eval points.
"""
tgv_runs() = [(; visc = 2.5e-4, seed = 0, role = :tgv, Re_target = 1600)]

"""
Per-(ν, Δ) `setup` view consumed by the rest of the pipeline. Built **once** from
coordinates: `dns = (; visc, seed, role)` and `Δ_factor` (the filter-to-grid
ratio; the filter width is `Δ = Δ_factor · l / n_les`). Never derived from another
setup — that is the whole point of the coordinate design.
"""
function make_setup(case, dns, Δ_factor)
    (; D, l, n_dns, n_les, cfl, backend, forced) = case
    Δ = Δ_factor * l / n_les
    return (;
        case.name, D, l, n_dns, n_les, Δ, Δ_factor,
        dns.visc, dns.seed, dns.role,
        cfl, forced,
        case.totalenergy,
        case.warmup_tstop,
        sampling = dns.role === :train ? case.train_sampling :
            dns.role === :tgv ? case.tgv_sampling : case.test_sampling,
        case.schedule,
        case.tiers,
        backend,
        outdir = datadir(case, dns, Δ_factor) |> mkpath,
        plotdir = case.plotdir,
    )
end

# --- pure path functions: every artifact self-locates from its coordinates ---

"Per-DNS directory; encodes (ν, seed) so each realization is isolated."
dnsdir(case, dns) =
    joinpath(case.rootdir, "$(case.name)_visc=$(dns.visc)_seed=$(dns.seed)_n=$(case.n_dns)")

"Per-(ν, seed, Δ) data directory."
datadir(case, dns, Δ_factor) = joinpath(dnsdir(case, dns), "delta=$(Δ_factor)")

"DNS warmup + raw-trajectory state (once per (ν, seed))."
dnsfile(case, dns) = joinpath(dnsdir(case, dns) |> mkpath, "dns.jld2")

"Δ-independent DNS-side metadata (times, DNS spectra/statistics) — once per DNS."
dnsmetafile(case, dns) = joinpath(dnsdir(case, dns) |> mkpath, "dns_meta.jld2")

"Heavy (ūbar, τ) field series + per-snapshot Re_Δ — per (ν, seed, Δ)."
fieldsfile(case, dns, Δ_factor) = joinpath(datadir(case, dns, Δ_factor) |> mkpath, "fields.jld2")

"Light LES-side metadata (LES spectra only; filtered-field turbulence stats are dropped)."
lesmetafile(case, dns, Δ_factor) = joinpath(datadir(case, dns, Δ_factor) |> mkpath, "les_meta.jld2")

"""
Canonical artifact key for a learned-model coordinate
`(; arch, tier, netseed, use_redelta)`. Every coordinate is encoded uniformly —
there is no "canonical seed keeps the plain key" special case (that existed only
for back-compat, which the clean slate removes).
"""
modelkey(m) = Symbol(m.arch, :_, m.tier, m.use_redelta ? :_re : Symbol(), :_seed, m.netseed)

"Trained-parameter file for a learned-model coordinate (models span the trainpool, so they live at the root)."
psfile(case, m) = joinpath(case.rootdir |> mkpath, "ps-$(modelkey(m)).jld2")

"Artifact name for a closure: the model-coordinate key, or the classical symbol itself."
modelname(m::NamedTuple) = modelkey(m)
modelname(m::Symbol) = m

"A-priori SFS prediction series for closure `m` on test dataset (dns, Δ)."
sfsfile(case, dns, Δf, m) = joinpath(datadir(case, dns, Δf) |> mkpath, "sfs-$(modelname(m)).jld2")

"Aggregated a-priori SFS statistics for closure `m` on (dns, Δ)."
sfsstatsfile(case, dns, Δf, m) =
    joinpath(datadir(case, dns, Δf) |> mkpath, "sfsstats-$(modelname(m)).jld2")

"""
Reduced a-posteriori metrics for closure `m` on (dns, Δ): the light artifact the
rollout writes after discarding the heavy LES fields (see [`solve_les`](@ref)).
"""
apostfile(case, dns, Δf, m) =
    joinpath(datadir(case, dns, Δf) |> mkpath, "apost-$(modelname(m)).jld2")

"Full LES rollout field series for the single showcase case (`savefields=true`)."
apostfieldsfile(case, dns, Δf, m) =
    joinpath(datadir(case, dns, Δf) |> mkpath, "apostfields-$(modelname(m)).jld2")

"A-priori equivariance-error series for closure `m` on (dns, Δ)."
equipriorfile(case, dns, Δf, m) =
    joinpath(datadir(case, dns, Δf) |> mkpath, "equiprior-$(modelname(m)).jld2")

"Phase-0 pointwise-Re_Δ binning diagnostic on (dns, Δ) (needs no trained model)."
redeltabinningfile(case, dns, Δf) =
    joinpath(datadir(case, dns, Δf) |> mkpath, "redelta-binning.jld2")

"""
Family key for a learned-model coordinate with the `netseed` dropped — the unit
the seed sweep aggregates over (every `modelkey` in the family shares it but for
the `_seed<i>` suffix).
"""
familyname(m) = Symbol(m.arch, :_, m.tier, m.use_redelta ? :_re : Symbol())

"Netseed-aggregated scalar metrics for all model families at evaluation point (dns, Δ)."
seedstatsfile(case, dns, Δf) = joinpath(datadir(case, dns, Δf) |> mkpath, "seedstats.jld2")

"""
Per-(dns, Δ) figure directory under `case.plotdir`, encoding the evaluation point
so the same figure name from different (ν, seed, Δ) points never clobbers.
"""
figdir(case, dns, Δf) =
    joinpath(case.plotdir, "$(dns.role)_visc=$(dns.visc)_seed=$(dns.seed)_delta=$(Δf)") |> mkpath

"Per-DNS (Δ-independent) figure directory under `case.plotdir`."
dnsfigdir(case, dns) =
    joinpath(case.plotdir, "$(dns.role)_visc=$(dns.visc)_seed=$(dns.seed)") |> mkpath
