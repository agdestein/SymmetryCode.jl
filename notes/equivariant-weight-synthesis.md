# Closed-form equivariant weight synthesis (replacing the Reynolds + eigendecomposition path)

**Status:** spec / not yet implemented. This is the bridge artifact between the
code (`SymmetryCode`) and the paper (`SymmetryPaper`). Implement the code first,
verify, commit; then update the paper's §3.3 to describe what the code does.

## Goal

`equivariant_net` (`src/nets.jl`) currently builds the equivariant weights by:
1. constructing the Reynolds/projection operators `r_lift`, `r_mid`, `r_sink`
   (`get_weight_projectors` in `src/symmetry.jl`) — dense matrices that average
   a weight block over the octahedral group;
2. `eigen`-decomposing each and keeping the eigenvalue-1 eigenvectors
   (`e_lift`, `e_mid`, `e_sink` in `nets.jl:41-43`);
3. storing those as frozen `Dense` weights (`proj_lift/mid/sink`) that map the
   compact learnables to a full weight block each forward pass.

For the octahedral group with **regular-representation** hidden channels and a
**tensor representation** at the lift/sink boundaries, the equivariant subspace
has a closed form, so steps 1–2 are unnecessary. Replacing them with a direct
synthesis is simpler, removes the `nreg²×nreg²` operators, and — crucially —
makes equivariance **bit-exact** (see below), which the eigenbasis is not.

## Background: the subspace

For the **regular representation** the group acts by a permutation `P_g`. The
paper indexes the `|G|=48` channels by group elements (`u_j(g)`), with
`(P_g u)(h) = u(g⁻¹ h)`. A weight block `w ∈ R^{48×48}` is equivariant iff it
commutes with every `P_g`, which forces it to depend only on the relative group
element: `w(g,h) = k(h⁻¹ g)` for a single `k : G → R` (the paper's form,
`groupconv.tex` §3.2). Equivalently `w(g,h) = κ(g⁻¹ h)` with `κ(x)=k(x⁻¹)` — same
subspace, inverse coordinatization. Either way: **48 free parameters per channel
pair**, a *group-circulant* matrix.

For the **lift** (tensor `R^9` → regular `R^48`) and **sink** (regular → tensor)
boundaries, the weight is an intertwiner between the tensor rep `Q_g = R_g⊗R_g`
and `P_g`. Such an intertwiner into/out of the regular representation is fixed by
its value at the identity element (Frobenius reciprocity): the rest is the
**orbit** of that value under the group. **9 free parameters per channel.**

These dimensions (48 / 9 / 9) match exactly the column counts kept today in
`e_mid` (`nreg`) and `e_lift`/`e_sink` (`nten`).

## The synthesis (exact recipes, in terms of `group_stuff` data)

`group_stuff(D)` already provides everything: `mats` (the 3×3 `R_g`, indexed
1..48), `cayley[a,b]` (= flat index of `mats[a]·mats[b]`, i.e. `a·b`),
`inverse_indices[a]` (= index of `a⁻¹`), `unitindex` (= identity, == 1).

### Mid layers (regular → regular): group-circulant gather

Match the paper's `w(g,h)=k(h⁻¹ g)`. With row = output channel-group index `m`
(= paper's `g`) and col = input `n` (= paper's `h`):

```
relidx[m,n] = cayley[inverse_indices[n], m]      # flat index of  n⁻¹·m  ∈ 1..48
w_mid[m,n,cout,cin] = k[relidx[m,n], cout, cin] / sqrt(nreg)
```

`k` is the learnable tensor of shape `(nreg, cout, cin)`. The synthesis is a pure
gather (index permutation of `k`) — no matmul against an eigenbasis. (Sanity:
`w_mid[m,n] = w_mid[gm,gn]` holds because `(gn)⁻¹(gm) = n⁻¹m`.)

### Lift (tensor 9 → regular 48): orbit of the tensor rep

```
Qmat[h] = 9×9 tensor-rep matrix for element h         # see ordering note below
w_lift[h, :, cout] = (Qmat[h] * c[:, cout]) / sqrt(nreg)
```

`c` learnable, shape `(nten, cout)`. Derivation: equivariance is `P_g w = w Q_g`;
evaluating at the identity row gives `w[h,:] = Qmat[h] · c` with `c := w[e,:]`.

### Sink (regular 48 → tensor 9): orbit, transposed

```
w_sink[:, h, cin] = (Qmat[h] * d[:, cin]) / sqrt(nreg)
```

`d` learnable, shape `(nten, cin)`. Same `Qmat`; the projector `r_sink` is the
transpose-structured intertwiner `Q_g w = w P_g`.

### `Qmat` ordering note (must verify against `r_lift`)

`Qmat[h]` is the *same* 9×9 tensor block already implicit in `r_lift`, i.e.
`Qmat[h][(x,y),(i,j)] = s[x] s[y] (p[x]==i) (p[y]==j)` for element `h=(p,s)`,
flattened with the **same** `(x,y) ↔ μ` order the Conv weight uses. In Julia's
column-major `vec`, `vec(R X Rᵀ) = (R⊗R) vec(X)`, so `Qmat[h] = kron(mats[h],
mats[h])` *if* the lift input packs the flattened gradient column-major. Build
`Qmat` directly from the `r_lift` index expression (don't assume) and assert
`Qmat[h]` reproduces `r_lift`'s tensor block, or test the whole synthesis against
the eigen path (see verification).

## Why this is correct and strictly better here

- **Same subspace.** The circulant/orbit bases span exactly the eigenvalue-1
  subspaces of `r_mid`/`r_lift`/`r_sink`. The eigenbasis is just an arbitrary
  orthonormal basis of the same space.
- **Well-conditioned.** The natural basis is orthogonal up to one global factor:
  `⟨R_a,R_b⟩_F = |G| δ_ab` and `⟨w(c),w(c')⟩_F = |G| c·c'`. The `1/sqrt(nreg)`
  in the recipes normalizes it to match the orthonormal eigenbasis, so existing
  `glorot_uniform` init scaling is preserved.
- **Bit-exact equivariance.** Every `P_g` is a 0/1 permutation and every
  `Q_g = R_g⊗R_g` has entries in `{0,±1}`, so the gather/orbit synthesis is exact
  in Float32/Float64. The `eigen` vectors are generically irrational, injecting a
  floating-point equivariance error — exactly the effect the `create_model`
  Float64-upcast and the `nets.jl`/`training.jl` comments work around. The closed
  form makes the a-priori equivariance error **structurally zero**, not just small.

The eigendecomposition is the representation-agnostic fallback; keep it for cases
without a closed form (sub-π/2 rotations, steerable/irrep features). It is not
needed for the current octahedral + regular-rep setup.

## Code change map (`SymmetryCode`, do first)

- `src/symmetry.jl`: add `relidx` (Cayley-derived gather index) and a `Qmat`
  builder alongside `get_weight_projectors`. **Keep `get_weight_projectors`** —
  `verify.jl` and the cross-check below still use it.
- `src/nets.jl` `equivariant_net`: replace `eigen(...)` + `proj_*` `Dense`
  layers with the gather (`project_mid`) and `Qmat`-orbit (`project_lift`,
  `project_sink`) syntheses. Learnable `ps` shapes are unchanged
  (`(nten,nchan[1])`, `(nreg,nchan[i+1],nchan[i])`, `(nten,nchan[end])`); bias
  handling unchanged. Optionally gate the old eigen path behind a `kwarg` flag.
- `src/verify.jl`: `test_equivariant_conv_sparse` already eigendecomposes — add a
  check that the new synthesis equals the eigen path's column space and that
  `apriori_equivariance_error` is exactly 0 (not just < tol).

### Convention alignment (keep code ↔ paper in lockstep)

The paper (committed) uses `w(g,h)=k(h⁻¹g)`; the `relidx` above
(`cayley[inverse_indices[n], m]` = `n⁻¹m` = `h⁻¹g`) matches it. **Do not** flip to
`m⁻¹n` in code without also changing the paper. Record any change here.

## Verification

1. `apriori_equivariance_error` for the equivariant net → **exactly 0.0** (the
   headline improvement). Compare against the eigen path's small-but-nonzero value.
2. New synthesis weights span the same subspace as `e_lift/e_mid/e_sink`
   (project a random `w̃` through both; compare images).
3. Smoke-train (few steps) on `setup_laptop` (2D, fast) — loss decreases; then a
   short `setup_turbulator_small` run to confirm 3D parity with the eigen path
   within training noise.
4. `runic -i .`, run `test/runtests.jl`.

## Paper change map (`SymmetryPaper`, do second, after code is verified)

- `Draft/sections/groupconv.tex` §3.3 ("Weight sharing…", `sec:weight-sharing`):
  the section currently derives the eigendecomposition `P = E Λ E⁻¹` and the
  synthesis `S = E^trunc`. Rewrite to present the closed-form synthesis: the
  group-circulant `k(h⁻¹g)` (already foreshadowed at the end of §3.2) and the
  `Q_h`-orbit for lift/sink; mention the eigendecomposition only as the general
  fallback. Keep the `k(h⁻¹g)` convention consistent with `relidx`.
- Reconcile the `figures/group-conv-build/` figure (it currently illustrates the
  eigendecomposition / retained eigenvectors) — regenerate on the SymmetryCode
  side per the paper's "don't hand-edit figure PDFs" rule.
- Re-`latexmk`; check `\cref{sec:weight-sharing}` and the §3.2 bridge sentence
  still read correctly.
