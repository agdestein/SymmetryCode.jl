# Reference data

## `tgv_re1600.csv`

Published reference for the canonical **Re = 1600** Taylor–Green vortex, used by
`plot_dissipation_tgv` (`src/plots.jl`) and the `run-tgv.jl` driver as the
external overlay on the dissipation-rate benchmark.

**Expected format** — whitespace- or comma-separated, `#` comments allowed,
three columns, already in standard nondimensional units (`V0 = L = 1`, so
`t* = t`):

```
# t*    E_k        epsilon
0.0     0.125      0.0
...
```

- column 1: `t*`  — nondimensional time `t·V0/L`
- column 2: `E_k` — volume-averaged kinetic energy `⟨½ uᵢuᵢ⟩` (≈ `0.125` at `t*=0`)
- column 3: `ε`   — dissipation rate (either `-dE_k/dt` or the enstrophy-based
  `2ν⟨SᵢⱼSᵢⱼ⟩`; they coincide for the resolved reference). Peaks near `t* ≈ 9`.

**Canonical source.** 512³ pseudo-spectral DNS reference from the 1st
International Workshop on High-Order CFD Methods (case C3.5), originating from
van Rees et al., *J. Comput. Phys.* 230 (2011) and Brachet et al., *J. Fluid
Mech.* 130 (1983). Drop the tabulated file in as `tgv_re1600.csv`.

If this file is absent, `plot_dissipation_tgv` warns and plots only the
self-generated DNS reference plus the LES closures.
