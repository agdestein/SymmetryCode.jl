# SymmetryCode

Source code for the paper

> Approaching the optimal closure: equivariance, inductive bias, and Reynolds-number generalization in data-driven LES
> Syver Døving Agdestein and Benjamin Sanderse, arXiv preprint, 2026
> <https://arxiv.org/abs/2603.05325>

## What this is

A Julia pseudo-spectral incompressible Navier–Stokes solver in 2D/3D (CPU or
CUDA), with classical and learned large-eddy-simulation closure models, and the
analysis pipeline that produces the paper's figures and tables.

The experiment trains three learned closures — a plain convolutional network, a
group-equivariant network, and a tensor-basis network — across a grid of
viscosities `ν` and filter ratios `Δ`, and compares them against the classical
dynamic Smagorinsky and Clark models and the no-closure baseline. The trained
models are then tested out of distribution: at a held-out higher Reynolds
number, at interpolated and extrapolated filter ratios, and on a decaying
Taylor–Green vortex. Each closure can optionally take the filter-scale Reynolds
number `Re_Δ = Δ²·√⟨|∇ū|²⟩/ν` as an extra input.

## Installation

Julia 1.12 or later. From this directory:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using SymmetryCode'   # precompile
```

A CUDA GPU is used when available, with a CPU fallback otherwise. The committed
`LocalPreferences.toml` pins the CUDA runtime to 13.2.

The published setup is sized for a single **NVIDIA H100 with 94 GB of memory**
(the Snellius `gpu_h100` partition) and will run out of memory on smaller cards.
To fit other hardware, reduce the grid sizes in `case_snellius` in
`src/experiment.jl`: `n_dns = 810` (the DNS resolution, the dominant cost) and
`n_les = 128` (the LES resolution). Note that the training viscosities in
`dns_runs` are chosen so the DNS stays well resolved at `n_dns = 810`
(`kmax·η ≥ 1.5`); a coarser DNS needs correspondingly larger `ν`. Dropping
`train_sampling.nturnover` and `test_sampling.nturnover` to ~0.05 shortens the
rollouts to near-free, which is enough for a structural smoke test.

## Running

Artifacts are written under a root directory that defaults to a cluster path.
Set `SYMMETRY_ROOTDIR` to run — or to re-plot an `scp`-ed copy of the results —
anywhere else. Figures always go to `output/`.

```bash
export SYMMETRY_ROOTDIR=/path/to/artifacts
julia --project run-dns.jl   # stage 1: DNS warm-up and filtered (ū, τ) training data
julia --project run-les.jl   # stage 2: train and evaluate the (ν, Δ) grid, write figures
julia --project run-tgv.jl   # stage 3: apply the trained closures to the Taylor–Green vortex
```

The three drivers run in that order, and are written so they can equally be
evaluated section by section in a REPL. Every stage is cached: a rerun skips any
unit whose artifact already exists.

On a SLURM cluster, `./submit.sh` submits all three stages as one dependency
chain, fanning each out over a job array containing only the missing units:

```bash
./submit.sh          # all stages
./submit.sh les tgv  # or a subset, once the earlier artifacts exist
```

Tests run with `julia --project=test test/runtests.jl`.

## Layout

| path | contents |
|---|---|
| `src/` | solver, filtering, closures, networks, training, analysis, plotting |
| `run-*.jl` | the three pipeline drivers |
| `submit.sh`, `job.sh` | SLURM submission |
| `reference/` | published reference data (see `reference/README.md`) |
| `scripts/` | result syncing and one-off migration helpers |
