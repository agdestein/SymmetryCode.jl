# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for the paper "Approaching the optimal closure: equivariance, inductive bias, and Reynolds-number generalization in data-driven LES" (Agdestein & Sanderse, 2026). It is a pseudo-spectral incompressible Navier–Stokes solver in 2D/3D with classical and learned LES closure models, plus the analysis and plotting pipeline used to generate the paper's figures.

The current line of work is the **Re_Δ experiment**: train the closures across a grid of viscosities ν and filter ratios Δ, then test out-of-distribution (a held-out higher-Re ν, interpolated/extrapolated filter ratios, and a decaying Taylor-Green vortex), optionally feeding the closures the filter-scale Reynolds number `Re_Δ = Δ²·√⟨|∇ū|²⟩/ν` as an extra input. The pipeline is **coordinate-driven** (see below) — the whole sweep is built from plain Cartesian products of loose coordinates, not a monolithic config.

CI only runs CompatHelper. **Companion paper & cross-repo sync.** The LaTeX source is at `../SymmetryPaper` (sibling under the `Symmetry/` umbrella).

## Commands

```bash
julia --project=. -e 'using SymmetryCode'   # precompile; cold-load smoke check
julia --project=test test/runtests.jl       # test suite (may lag the refactor — see note)
julia --project run-dns.jl              # stage 1: DNS warm-up + (ūbar,τ) data, all runs
julia --project run-les.jl                  # stage 2: train + evaluate the (ν,Δ) grid + figures
julia --project run-tgv.jl                  # stage 3: apply the trained closures to the TGV
sbatch job.sh                               # SLURM submission on a single H100
runic -i .                                  # format all code in place (run from repo root)
```

**REPL workflow — prefer `julia-mcp` over `julia -e ...`.** A persistent Julia session is available via the `julia-mcp` MCP server (`mcp__julia__julia_eval`, `julia_list_sessions`, `julia_restart`). Use it for smoke checks, iterative debugging, and running pipeline sections. **Revise** is loaded in every session, so source edits are picked up without restart (after *deleting* a function, `isdefined` may stay `true` while calls throw `MethodError` — expected). **TestEnv** is in the global env (`using TestEnv; TestEnv.activate()` makes the `test/` deps loadable in-process). Only `julia_restart` as a last resort (segfault, GPU stuck, native-code change Revise can't patch).

The three drivers are run in order (`run-dns` → `run-les` → `run-tgv`) and are structured as scripts meant to be evaluated section by section. Each stage is **cache-guarded**: a rerun short-circuits when the artifact for that coordinate already exists; `config.experiments` toggles which stages run and `config.force` invalidates a stage's cache. Format with [Runic.jl](https://github.com/fredrikekre/Runic.jl) (no per-repo config) before committing.

> **Note on the test suite.** `test/runtests.jl` predates the coordinate refactor and may reference removed functions; treat a red test as "needs migrating", not as a regression in the new code. The package itself cold-loads and runs.

## The coordinate design

This is the spine of the codebase (`src/experiment.jl`). The swept axes — viscosity, DNS seed, filter ratio, network architecture/size/init-seed, Re_Δ on/off — are **loose coordinates**, never fused into one config. Sweeps are plain Cartesian products or zips. No code block ever "destructures a list of setups that carry redundant info and rebuilds them with one parameter changed".

- **`case`** (`case_snellius()`) — everything shared across the whole sweep: grid (`l`, `n_dns`, `n_les`), `cfl`, sampling schedules, train/test filter ratios, the size tiers, the training `schedule`, the artifact `rootdir` (cluster) and figure `plotdir`. Never holds a swept axis.
- **Run coordinates** — `dns_runs()` returns `(; visc, seed, role)` for the forced HIT runs (`role ∈ {:train, :test_indist, :test_ood}`); `tgv_runs()` returns `(; visc, seed, role=:tgv, Re_target)` for the decaying Taylor-Green test. A run is identified by `(visc, seed)`.
- **Filter ratio** — `Δ_factor` (filter width `Δ = Δ_factor · l / n_les`). Training uses `case.filters_train`, tests use `case.filters_test`.
- **Model coordinate** — `(; arch, tier, netseed, use_redelta)` with `arch ∈ {:conv, :equi, :tbnn}`, `tier ∈ keys(case.tiers)`. The tiers form a **capacity grid** `pN` (target ≈ N parameters, fixed 3 hidden layers, width-only scaling; roughly parameter-matched across archs within a tier — check with `paramcount`). The runs sweep `p120, p400, p1200, p3000` for all three archs (`run-les` `sizes`); `p3000` is `config.top`, the matched tier reused outside the saturation sweep (the forced / equal-capacity / Re_Δ comparisons). Higher tiers (`p8000`, `p16000`) are still *defined* in `default_tiers` but **not run** — the equivariant rollout's `|G|·c·n³` activation OOMs past ~8k and the models saturate well before, so `run-les` leaves those `sizes` commented out. A **family** is the tuple minus `netseed` (`familyname`); the seed sweep aggregates over `netseed`. Classical closures are bare symbols (`:nomo`, `:clar`, `:smag`, `:dynsmag`, `:vers`, `:bard`).

`make_setup(case, dns, Δ_factor)` derives the per-`(ν, Δ)` `setup` view (the NamedTuple the rest of the pipeline destructures) **once**, from coordinates — never rebuilt from another setup. **Pure path functions** in `experiment.jl` locate every artifact from its coordinates, so no block destructures-and-rebuilds a setup to find its inputs:

| path fn | artifact | keyed by |
|---|---|---|
| `dnsfile` | DNS warm-up state | `(case, dns)` |
| `dnsmetafile` | Δ-independent DNS metadata (times, spectra, stats, t_int) | `(case, dns)` |
| `fieldsfile` | heavy `(ūbar, τ)` field series + per-snapshot `redelta` | `(case, dns, Δf)` |
| `lesmetafile` | light LES-side spectra + mean Re_Δ (`redelta_mean`; TGV also: peak-instant Re_Δ `redelta_peak`) | `(case, dns, Δf)` |
| `psfile` | trained params + states + loss curves + `redelta_norm` | `(case, m)` |
| `sfsstatsfile` | aggregated a-priori SFS statistics (reduced on the fly) | `(case, dns, Δf, m)` |
| `apostfile` | reduced a-posteriori metrics (one rollout) | `(case, dns, Δf, m)` |
| `apostfieldsfile` | full LES field series (showcase only) | `(case, dns, Δf, m)` |
| `equipriorfile` | a-priori equivariance error series | `(case, dns, Δf, m)` |
| `redeltabinningfile` / `seedstatsfile` | Phase-0 binning / netseed aggregate | `(case, dns, Δf)` |
| `figdir` / `dnsfigdir` | per-(ν,seed,Δ) / per-(ν,seed) figure dir | under `case.plotdir` |

Heavy fields and light metadata live in **separate files** so plot iteration never reloads the heavy `fieldsfile`. `psfile`/`dns*`/`fields*`/`les*` are multi-key archives — read them with `load(file, "key")`; `sfs*`/`apost*`/`equiprior*`/`redeltabinning*`/`seedstats*` are single-object — read with `load_object(file)`.

## Pipeline

```
case_snellius()  +  dns_runs() / tgv_runs()        (loose coordinates)
  create_dns(case, dns)            → dnsfile               (DNS warm-up; not for TGV)
  create_data(case, dns)           → dnsmetafile + per-Δ fieldsfile / lesmetafile
  create_data_tgv(case, tgv)       → same schema from a decaying analytic TGV
  train_model(case, m, trainpool)  per m in the purposeful model lists  → psfile
  per (dns, Δf) eval point:
    compute_sfs_stats (reduce-on-the-fly)  → sfsstatsfile               (a-priori)
    solve_les (reduce-on-the-fly)          → apostfile (+ apostfieldsfile showcase)
    apriori_equivariance_error             → equipriorfile
    compute_redelta_binning                → redeltabinningfile
    get_seed_statistics                    → seedstatsfile             (netseed spread)
  plot_saturation / plot_trend_vs_redelta / plot_* → figdir / dnsfigdir / case.plotdir
```

`run-les.jl` assembles the model coordinates into a few **purposeful lists** (never one Cartesian product), each broad along a single axis, and trains every distinct model once over `build_trainpool(case)`:

- **A — Saturation** (`plot_saturation`, the headline): +Re, every architecture across its full size grid, evaluated at one in-distribution `(ν, Δ)`. Error vs **parameter count** → all archs saturate to the same floor, the inductive-bias models reach it at far fewer params (Langford–Moser optimal closure).
- **B — Equal capacity / equivariance**: the matched top tier (`config.top`, `:p3000`) ±Re at the same in-distribution point — the bars, equivariance, errors + timing tables (reads A's artifacts).
- **C — Re_Δ trend** (`plot_trend_vs_redelta`): top tier ±Re across the full `(ν, Δ)` test grid → dissipation ratio / a-priori error / a-posteriori error vs global Re_Δ.

A is broad in *size* / narrow in *eval*; C is broad in *eval* / narrow in *size*; the +Re top models are shared (trained once, `all_models` dedupes). `run-tgv.jl` (**D**) reuses C's top-tier ±Re `psfile`s on the TGV `(tgv, Δf)` points. The suite is cache-guarded per coordinate, so adding a size or a seed later recomputes only the gaps.

## Architecture

`SymmetryCode.jl` includes a flat set of per-topic source files. The hierarchy is conceptual, not file-organizational:

- **`solver.jl`** — the core. `Grid{D}`, field allocators (`scalarfield`/`vectorfield`/`tensorfield` + `space*` physical-space counterparts), kernels, time integrators (`forwardeuler!`, `abcn!`, `wray3!`), `turbulence_statistics`, `spectrum`/`spectral_stuff`, `filter_reynolds` (the global Re_Δ), and `tabulate`/`reset_tables`. **Cross-cutting utilities go here.**
- **`experiment.jl`** — the coordinate layer: `case_snellius`, `dns_runs`/`tgv_runs`, `make_setup`, the model keys (`modelkey`/`modelname`/`familyname`), and every path function. **The live configuration spine.**
- **`setups.jl`** — just `build_models(case, setup, models)`: build the inference closures for a model list (classical via a small table, learned via `build_model`).
- **`filtering.jl`** — `cutoff!` (spectral truncation) and `gaussianfilter!` (with a "don't filter forced shells" carve-out).
- **`symmetry.jl`** — octahedral group machinery: `group_stuff`, `get_weight_synthesis`/`get_weight_projectors`, `transform_vector`/`transform_tensor`.
- **`closures.jl`** — classical models (Clark / Smagorinsky / dynamic Smagorinsky / Verstappen / Bardina). Each kernel uses `tensorat` / `store_symtensor!`; each wrapper goes through `run_closure_kernel`.
- **`nets.jl`** — Lux architectures, all pointwise MLPs (no spatial mixing) via `PointwiseConv` (1×1 channel matmul; avoids the cuDNN scratch blow-up — see the file header). `equivariant_net` is a group convolution with closed-form weight tying (`project_*`); `mlp` is the non-equivariant baseline; `tbnn_net` predicts trace-free tensor-basis coefficients from the gradient invariants. `fullchain` (equi/conv) and `tbnn` wrap a trained net into a `(u, ∇ū) -> τ` closure for `les!`. `paramcount(case, m)` is the closed-form param count. `symmetrize_pointwise` (the `:convsym` baseline) is available but not in the default sweep.
- **`training.jl`** — the fixed-budget `train` loop (Lux + Zygote + AdamW with `ClipNorm`, linear-warmup→cosine LR; **no early stopping, no checkpointing** — the nets are tiny; validation is monitoring-only), the losses, the netsetup/net builders (`make_netsetup`, `build_net_stuff`), `build_model` (inference), `compute_redelta_norm`, `build_trainpool`, and the orchestrators `train_model` / `train_models`.
- **`data.jl`** — DNS warm-up (`create_dns`), `(ūbar, τ)` generation (`create_data`, `create_data_tgv`; both call `sfs!`), the CPU dataloaders (`create_dataloader`, `create_dataloader_tbnn`), and `append_redelta` (the Re_Δ input channel).
- **`les.jl`** — LES RHS with closure (`les!`) and `solve_les` / `solve_les!`: the a-posteriori rollout **reduces metrics on the fly and discards the heavy LES fields** (one rollout → one light `apostfile`; `savefields` keeps the field series for the single showcase case).
- **`analysis.jl`** — `compute_sfs_stats` (a-priori stats, **reduced on the fly** — the closure is evaluated per snapshot and reduced immediately, never writing a predicted-SFS field series), `apriori_equivariance_error`, `compute_redelta_binning` (the Phase-0 diagnostic), and `get_seed_statistics` (netseed aggregate → `seedstatsfile`).
- **`plots.jl`** — all `plot_*`, the coordinate-aware `plotlabel`/`plotstyle` (a model resolves to its arch's label/color; the `+Re` variant is dashed; `tierlabel`/`famlabel` add the capacity tag where several sizes share an axis), `getlabels`/`getstyles` (the canonical style table — **every plot uses it**), `plot_saturation` (error vs parameter count — the headline), `plot_trend_vs_redelta` (the Re_Δ trend), `plot_tgv_vs_redelta` (places the TGV on that same Re_Δ axis, at the peak-dissipation-instant Re_Δ), and `write_errors_table` / `write_timing_table` (the paper-ready LaTeX, from the seed aggregate + classical values).

## Conventions to know before editing

**FFT scaling.** Spectral fields use `û_code = F[u_phys] / n^D`. With this, `Σ_k |û_code|² = ⟨|u|²⟩_phys`, so `energy(u)` returns the physical mean KE `⟨½ uᵢuᵢ⟩` and `get_dissipation!` returns physical ε directly. `to_spec!` divides by `n^D`, `to_phys!` multiplies; prefer them over inline `mul!`/`ldiv!` + scaling.

**RFFT energy accounting.** Real FFTs store only `kx ∈ 0:kmax`; energy/dot-product sums must double-count `kx ∈ 1:kmax-1` — see `getenergy`, `spectralsum`, `spectraldot` in `solver.jl`. The `shells` from `getshells` carry both the storage and conjugate-pair indices for this.

**Field representation.** Vector/tensor fields are **NamedTuples of arrays** (`u.x`, `τ.xx`, …; non-symmetric tensors add `yx`, `zx`, `zy`, `xz`). Allocators dispatch on `Grid{D}`. NN-facing code uses **stacked** packed arrays `(n,…,n, tensordim)`; `unstack_symtensor` converts back to the NamedTuple-of-views form.

**Dimension dispatch.** 2D/3D variants are separate methods on `Grid{2}`/`Grid{3}`, not runtime branches; solver kernels are deliberately duplicated 2D/3D — the user has opted **not** to unify them.

**GPU dispatch.** Backends from `KernelAbstractions` (`CPU()`, `CUDABackend()`). Use `apply!(kernel!, grid, args)` (wraps workgroup/ndrange/synchronize). Adapt with `adapt(backend)` / `cpu_device()`. Bound the per-model GPU footprint with a lazy build + `clean()` between models (the drivers do this).

**Cache structs.** Time integrators take `cache = getcache(grid)` (or extended forms). The cache holds **all scratch buffers** for an RHS eval; extend it rather than allocating in the hot path.

**Sub-grid stress convention.** The SFS stress is **deviatoric**: the DNS target is made trace-free in `sfs!`, and all three learned closures produce a deviatoric stress by construction (TBNN via `deviator`, equi/conv via a trace-removing `symm` tail). The networks regress the **normalized** target `τ / (Δ²·‖∇ū‖²)` from the normalized gradient `∇ū/‖∇ū‖`; the inference wrappers (`fullchain`, `tbnn`) multiply back by `Δ²·‖∇ū‖²`. Keep the two consistent.

**Re_Δ input feature.** `use_redelta=true` appends one standardized `log Re_Δ` channel to each net's input — an extra VGT channel for `mlp`, an extra invariant for `tbnn_net`, and a **trivial-rep channel at the equi lift** (a per-output-channel `weight_re` broadcast equally to all |G| copies, so octahedral equivariance is bit-exact). The global `Re_Δ = Δ²·√⟨|∇ū|²⟩/ν` is the same at training (stored `redelta` via `filter_reynolds`) and inference (reusing the pointwise `|∇ū|²` already formed in `fullchain`/`tbnn` — no extra VGT pass). The standardization `(μ, σ)` is computed over the trainpool (`compute_redelta_norm`) and stored in `psfile`. `use_redelta=false` rebuilds the previous nets bit-for-bit.

**Train/val split.** The dataloaders return `(trainloader, valloader)` via `split_loaders`: a **time-based** holdout (last `schedule.val_fraction` of the time-ordered snapshots). Training is **fixed-budget** — the validation loss is tracked only for the convergence curve, never for model selection (no early stopping, no checkpointing). `split_loaders` folds the last spatial axis into the batch dimension (the 1×1 models use no neighbor info) and casts to `schedule.precision`.

**Net vs. solver precision.** Networks train in `schedule.precision` (default `Float32`); the solver, data generation, and spectral fields stay `Float64`. `psfile` keeps the trained precision, but `build_model` upcasts `ps` to `Float64` at load, so the inference forward pass through `fullchain`/`tbnn` is uniformly `Float64` (otherwise the equivariance diagnostic would be pinned to Float32 eps).

**Seed sweep.** `netseed` is a model coordinate (network init + batch shuffling only; the data is unaffected). The driver uses one seed set (`netseeds`) shared by the saturation curve (one eval point) and the top-tier grid (full ν × Δ). `get_seed_statistics` aggregates the per-`familyname` spread at each eval point into `seedstatsfile`, which the trend figure, the seed-aggregated bar plots (a-priori / dissipation / backscatter / equivariance, mean ± std whiskers), and `write_errors_table` all consume; `plot_saturation` reads the per-model artifacts directly (no aggregate file) so adding sizes needs no rebuild. The dense per-curve figures (densities, budget, spectra, error-vs-time) take a curated one-seed, top-tier subset (`series_models` in the drivers).

**Shell-clamp forcing.** Forced HIT maintains low-wavenumber shell energy at its initial value rather than an explicit body force (`shells = energy_shells(grid, [1,2], u)`; per-step `maintain_shell_energy!`). The decaying TGV is unforced — `solve_les` derives `forced = case.forced && dns.role !== :tgv`.
