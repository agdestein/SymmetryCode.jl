# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for the paper "Comparison of Data-Driven Symmetry-Preserving Closure Models for Large-Eddy Simulation" (Agdestein & Sanderse, 2026). It is a pseudo-spectral incompressible Navier–Stokes solver in 2D/3D with classical and learned LES closure models, plus the analysis and plotting pipeline used to generate the paper's figures.

CI only runs CompatHelper. The repository is driven from the REPL via `main.jl`.

## Commands

```bash
julia --project=. -e 'using SymmetryCode'   # precompile; the primary smoke check
julia --project=test test/runtests.jl       # run the test suite
julia --project main.jl                     # run the whole pipeline (long; usually run interactively)
sbatch job.sh                               # SLURM submission; runs main.jl on a single H100
```

The test environment is a separate project under `test/` that uses `[sources] SymmetryCode = {path = ".."}` to depend on the dev checkout — `Pkg.instantiate` from `test/` is enough to set it up.

`main.jl` is the canonical pipeline driver. It is structured as a script meant to be evaluated section by section in a REPL — sections create artifacts (`output/<name>/dns.jld2`, `data.jld2`, `ps-*.jld2`, `u-post-*.jld2`, `kde_*.jld2`, …) that downstream sections consume. Reruns short-circuit if the artifact already exists for that setup.

Code is formatted with [Runic.jl](https://github.com/fredrikekre/Runic.jl) (no per-repo config).

## Pipeline

```
setup_*()  →  create_dns       →  dns.jld2
              create_data      →  data.jld2          (ubar,τ) pairs
              create_{tbnn,equi,conv} (training)  →  ps-*.jld2
              solve_les               →  u-post-*.jld2
              predict_sfs / compute_densities / apriori_error / compute_qr / …
              plot_*                  →  output/<setup>/plots/*.pdf|png
```

Three setups in `src/setups.jl`:
- `setup_laptop` — 2D, small (`n_dns=1024`), uses `default_backend()` (CUDA if available, else CPU).
- `setup_turbulator` — 3D, `n_dns=512`, hard-codes `CUDABackend()`.
- `setup_snellius` — 3D, `n_dns=810`, writes to `/projects/prjs1757/...` (cluster path).

A "setup" is a NamedTuple — fields are destructured throughout the codebase (`(; D, l, n_les, backend, Δ, visc, cfl) = setup`).

## Architecture

`SymmetryCode.jl` includes a flat set of per-topic source files. The hierarchy is conceptual, not file-organizational:

- **`solver.jl`** — the core. Defines `Grid{D}`, field allocators (`scalarfield` / `vectorfield` / `tensorfield` and their `space*` physical-space counterparts), kernels, time integrators (`forwardeuler!`, `abcn!`, `wray3!`), and the shared helpers everything else builds on. **When adding cross-cutting utilities, put them here.**
- **`filtering.jl`** — `cutoff!` (spectral truncation) and `gaussianfilter!` (with a "don't filter forced shells" carve-out).
- **`symmetry.jl`** — octahedral group machinery: `group_stuff`, `get_weight_projectors`, `transform_vector` / `transform_tensor` (used by the equivariant network and by `test_equivariance_post`).
- **`closures.jl`** — classical models (Clark / Smagorinsky / dynamic Smagorinsky / Verstappen Q-R). Each kernel uses `tensorat` / `store_symtensor!`; each wrapper goes through `run_closure_kernel`.
- **`nets.jl`** — Lux network architectures for the learned closures: `equivariant_net` (G-Conv with octahedral weight-tying via the projectors from `symmetry.jl`), `cnn` (plain Conv baseline), `tbnn` (Tensor Basis Neural Network using `ninvariant` / `nbasis` / `build_tensorbasis`), plus `fullchain` which wraps a trained net + projection into a `(u, G) -> τ` closure usable by `les!`.
- **`training.jl`** — generic `train` loop (Lux + Zygote + AdamW, keeps best-validation params), losses (`create_loss`, `create_loss_tbnn`), and the per-model orchestrators `create_tbnn` / `create_equi` / `create_conv` that build the net, optionally train, persist `ps-*.jld2`, and return a ready-to-use closure.
- **`data.jl`** — DNS warmup (`create_dns`), `(ubar,τ)` pair generation (`create_data`, calls `sfs!`), and CPU dataloaders.
- **`les.jl`** — LES RHS with closure (`les!`) and the post-training rollout (`solve_les` / `solve_les!`).
- **`analysis.jl`** — post-hoc evaluation: `predict_sfs`, `compute_densities`, `apriori_error`, `apriori_equivariance_error`, `get_dissipation_errors`, `compute_qr`.
- **`plots.jl`** — all `plot_*` functions, plus `getlabels()` (the canonical map from model key to display label).
- **`verify.jl`** — REPL-only sanity checks (`test_equivariant_*`, `dns_aid`, `test_equivariance_post`). Not part of the pipeline.

## Conventions to know before editing

**FFT scaling.** Spectral fields use a non-standard `(n/l)^D` convention: forward divides by it (`to_spec!`), inverse multiplies by it (`to_phys!`). Both helpers live in `solver.jl`. Prefer them over inline `mul!(...); ./= fac` / `ldiv!(...); .*= fac` — only one place in `les.jl` keeps the inline form (because it allocates fresh component arrays).

**RFFT energy accounting.** Real FFTs store only `kx ∈ 0:kmax`. Energy/dot-product sums must add the contribution of `kx ∈ 1:kmax-1` twice — see `getenergy`, `spectralsum`, `spectraldot` in `solver.jl`. The `shells` returned by `getshells` carry both the storage indices and the conjugate-pair "energy" indices for this reason.

**Field representation.** Vector and tensor fields are **NamedTuples of arrays**, not multi-dim arrays. Components are named (`u.x`, `u.y`, `u.z`; `τ.xx`, `τ.yy`, `τ.zz`, `τ.xy`, `τ.yz`, `τ.zx` for symmetric tensors; with `yx`, `zx`, `zy`, `xz` added for non-symmetric ones in `tensorfield_nonsym`). Allocators dispatch on `Grid{D}` for 2D vs 3D shape. NN-facing functions deal with **stacked** packed arrays (`(n,...,n, tensordim)`); `unstack_symtensor` converts back to the NamedTuple-of-views form expected by `strain_from_gradient`, `contract_dissipation`, etc.

**Dimension dispatch.** 2D/3D variants are written as separate methods specialized on `Grid{2}` / `Grid{3}` rather than runtime branches. Kernels for solver primitives (`project!`, `twothirds!`, `wavenumber_*`, `viscosity!`, `tensordivergence!`, …) are deliberately duplicated 2D/3D for clarity; the user has opted **not** to unify these.

**GPU dispatch.** Backends come from `KernelAbstractions` (`CPU()`, `CUDABackend()`). Use `apply!(kernel!, grid, args)` rather than calling kernels directly — it wraps the workgroup-size, ndrange, and `synchronize` boilerplate. Adapt CPU arrays to the backend with `adapt(backend)` / `cpu_device()`.

**Cache structs.** Time integrators take a `cache = getcache(grid)` (or its extended forms with `dissfield`, `G`, etc.). The cache holds **all scratch buffers** used inside an RHS evaluation. When extending an RHS, add scratch to the cache rather than allocating inside the hot path.

**Sub-grid stress convention.** The model output `τ` is the **deviatoric** SFS stress. `make_tracefree!` enforces this and is applied symmetrically to model predictions and to filtered-DNS ground truth.

**Shell-clamp forcing.** The pipeline maintains low-wavenumber shell energy at its initial value rather than using an explicit body force. Setup: `shells = energy_shells(grid, [1, 2], u)`. Per-step: `maintain_shell_energy!(u, shells)`. Forcing-via-`forced_rhs!` exists but is currently commented out at all call sites.
