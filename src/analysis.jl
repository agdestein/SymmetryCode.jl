# Post-hoc evaluation of LES closures: a-priori SFS statistics (tensor error,
# dissipation KDE / backscatter), equivariance commutation errors, the Phase-0
# Re_Δ binning diagnostic, and the netseed aggregate.

"""
Scalar moments of a sample vector: mean, median, std, and the third
standardized moment (skewness). No external dependency.
"""
function moments(x)
    m = mean(x)
    s = std(x)
    return (;
        mean = m,
        median = median(x),
        std = s,
        skewness = iszero(s) ? zero(m) : mean(((xi - m) / s)^3 for xi in x),
    )
end

"""
A-priori statistics for one closure `m` on test dataset (dns, Δf), reduced **on
the fly**: the closure is evaluated on every filtered-DNS snapshot and reduced to
scalar metrics immediately, so the heavy predicted-SFS field series is never
written to disk (mirrors the reduce-on-the-fly a-posteriori path in
[`solve_les`](@ref); see Notes/ReExperiment.md). Persists the light
`sfsstatsfile(case, dns, Δf, m)`:

- `apriori` — `(; relerr, crosscor)` against the filtered-DNS reference over the
  full trace-free tensor. `(0.0, 1.0)` for `:ref`, `(1.0, 0.0)` for `:nomo`.
- `diss` — `(; mean, median, std, skewness, backscatter)` of the pointwise SFS
  dissipation `ε_sfs = -τᵢⱼSᵢⱼ`. `backscatter` is the fraction with `ε_sfs < 0`;
  exactly 0 for `τ = -2νₜS` with `νₜ ≥ 0` (e.g. Smagorinsky), ≈ 0.3–0.4 for
  filtered HIT. Convention: `ε_sfs > 0` drain, `< 0` backscatter.
- `kde.diss` — KDE of `ε_sfs` (the backscatter evidence; only the dissipation PDF
  is kept, per ReExperiment.md).

`m === :ref` uses the filtered-DNS `outputs` directly; `m === :nomo` is zero
stress; every other closure calls `getmodel()` — a zero-arg thunk built only on a
cache miss and never for `:ref`/`:nomo` — and re-evaluates it per snapshot.
"""
function compute_sfs_stats(case, m, dns, Δf, getmodel = () -> nothing; force = false)
    (; D, l, n_les, backend) = case
    file = sfsstatsfile(case, dns, Δf, m)
    skip_if_cached(file; force, label = "SFS stats for $(modelname(m))") && return

    inputs, outputs = load(fieldsfile(case, dns, Δf), "inputs", "outputs")

    g = Grid{D}(; l, n = n_les, backend)
    u = vectorfield(g)
    τ_model = spacetensorfield(g)
    τ_ref = spacetensorfield(g)
    dissfield = spacescalarfield(g)
    τhat = scalarfield(g)
    plan = plan_rfft(τ_model.xx)
    A = tensorfield_nonsym(g)
    AA = spacetensorfield_nonsym(g)

    model = m in (:ref, :nomo) ? nothing : getmodel()

    @info "Computing SFS stats for $(modelname(m))"
    flush(stderr)

    diss_samples = Float64[]
    relerrs = Float64[]
    crosscors = Float64[]

    for i in eachindex(inputs)
        # Physical velocity gradient from ūbar — feeds both the strain rate and
        # (for a learned/classical closure) the model input, in a single pass.
        foreach(copyto!, u, inputs[i])
        apply!(vectorgradient!, g, (A, u, g))
        for (AA, A) in zip(AA, A)
            apply!(twothirds!, g, (A, g))
            to_phys!(AA, A, plan, g)
        end
        S = strain_from_gradient(AA, g)

        # Reference τ (spectral → physical, trace-free).
        for (τ, τcpu) in zip(τ_ref, outputs[i])
            copyto!(τhat, τcpu)
            apply!(twothirds!, g, (τhat, g))
            to_phys!(τ, τhat, plan, g)
        end
        make_tracefree!(τ_ref, g)

        # Model τ, evaluated on the fly. A learned/classical closure is dealiased
        # (it can excite harmonics above the 2/3 cutoff) and trace-freed.
        if m === :nomo
            for t in τ_model
                fill!(t, 0)
            end
        elseif m === :ref
            for (t, r) in zip(τ_model, τ_ref)
                copyto!(t, r)
            end
        else
            foreach(copyto!, τ_model, unstack_symtensor(model(u, AA), g))
            for t in τ_model
                dealias_phys!(t, τhat, plan, g)
            end
            make_tracefree!(τ_model, g)
        end

        # ε_sfs = -τᵢⱼSᵢⱼ (positive = drain).
        dissfield .= contract_dissipation(τ_model, S, g)
        append!(diss_samples, .-vec(Array(dissfield)))

        # A-priori metrics over the full trace-free tensor vs reference.
        if m === :ref
            push!(relerrs, 0.0)
            push!(crosscors, 1.0)
        elseif m === :nomo
            push!(relerrs, 1.0)
            push!(crosscors, 0.0)
        else
            a = Array(stack(τ_ref))
            b = Array(stack(τ_model))
            bb, aa = b .- mean(b), a .- mean(a)
            push!(relerrs, norm(b - a) / norm(a))
            push!(crosscors, dot(bb, aa) / sqrt(dot(bb, bb) * dot(aa, aa)))
        end
    end

    kdiss = if iszero(std(diss_samples))
        (; x = Float64[], density = Float64[])
    else
        e = kde(diss_samples)
        (; x = collect(e.x), density = collect(e.density))
    end
    result = (;
        kde = (; diss = kdiss),
        diss = (; moments(diss_samples)..., backscatter = mean(<(0), diss_samples)),
        apriori = (; relerr = mean(relerrs), crosscor = mean(crosscors)),
    )
    save_object_atomic(file, result)
    return
end

"""
Phase-0 diagnostic for the Re_Δ-as-input experiment (`Notes/ReExperiment.md`).

Bins the a-priori filtered-DNS pairs by **pointwise** filter-scale Reynolds
number `Re_Δ = Δ²|Ā|/ν` and reports, per bin, the median of the two quantities
the closures hold *scale-invariant*:

- `C_ε = -τᵢⱼSᵢⱼ / (Δ²|Ā|³)` — the dimensionless SFS dissipation (transfer)
  coefficient;
- `‖τ‖_F / (Δ²|Ā|²)` — the normalized SFS stress magnitude (the exact target the
  dataloaders regress in `create_dataloader`).

A **flat** curve means the normalized target is independent of Re_Δ within this
single flow (pure self-similarity → no usable Re_Δ signal); a **sloped** curve is
a within-flow Re_Δ dependence.

Why this gates the experiment (see `Notes/ReDependence.md`): with ν and Δ fixed,
`Re_Δ ∝ |Ā|`, so this is *amplitude/intermittency* spread at a single Reynolds
number, not Reynolds-number variation. The Re_Δ feature only pays off if this
**within-flow** slope has the *same sign* as the **across-flow** trend in
`fig:dissipation-vs-re` (lower global Re_Δ → less sub-filter energy → smaller
normalized stress, i.e. a *positive* across-flow slope). Opposite or flat ⇒
Simpson confound ⇒ pivot to a *global* Re_Δ feature + a viscosity sweep before
spending any DNS hours.

Operates on the per-(ν, Δ) `fieldsfile` (filtered-DNS reference stress); needs no
trained model. `subsample` strides the per-snapshot points to cap memory. Persists
`redeltabinningfile(case, dns, Δf)`; returns nothing.
"""
function compute_redelta_binning(
        case, dns, Δf;
        force = false, nbin = 30, quantiles = (0.005, 0.995), subsample = 1,
    )
    (; D, l, n_les, backend) = case
    visc = dns.visc
    Δ = Δf * l / n_les
    file = redeltabinningfile(case, dns, Δf)
    skip_if_cached(file; force, label = "Re_Δ binning") && return

    inputs_eval, outputs_eval = load(fieldsfile(case, dns, Δf), "inputs", "outputs")

    g = Grid{D}(; l, n = n_les, backend)
    u = vectorfield(g)
    τ_ref = spacetensorfield(g)
    τhat = scalarfield(g)
    plan = plan_rfft(τ_ref.xx)
    A = tensorfield_nonsym(g)
    AA = spacetensorfield_nonsym(g)
    ϵ = eps(eltype(AA.xx))   # scalar 0-guard; must stay out of the `@.` blocks

    re = Float64[]        # pointwise Re_Δ = Δ²|Ā|/ν
    cdiss = Float64[]     # C_ε   = -τ:S / (Δ²|Ā|³)
    cstress = Float64[]   # ‖τ‖_F / (Δ²|Ā|²)

    @info "Re_Δ binning over $(length(inputs_eval)) eval snapshots"
    flush(stderr)
    for i in eachindex(inputs_eval)
        # Physical velocity gradient from the filtered DNS — same recipe as
        # `compute_sfs_stats`.
        foreach(copyto!, u, inputs_eval[i])
        apply!(vectorgradient!, g, (A, u, g))
        for (AA, A) in zip(AA, A)
            apply!(twothirds!, g, (A, g))
            to_phys!(AA, A, plan, g)
        end
        S = strain_from_gradient(AA, g)

        # Reference deviatoric SFS stress (spectral → physical, trace-free).
        for (τ, τcpu) in zip(τ_ref, outputs_eval[i])
            copyto!(τhat, τcpu)
            apply!(twothirds!, g, (τhat, g))
            to_phys!(τ, τhat, plan, g)
        end
        make_tracefree!(τ_ref, g)

        # |Ā|² = Σ_{ij} (∂ūᵢ/∂xⱼ)² pointwise (all D² gradient components).
        A2 = zero(AA.xx)
        for a in AA
            @. A2 += a^2
        end
        Anorm = sqrt.(A2)

        # Pointwise Re_Δ and the two scale-invariant targets. `contract_dissipation`
        # with both arguments equal to τ yields ‖τ‖_F² (it already carries the
        # factor-2 on the off-diagonals), so no separate norm helper is needed.
        reδ = @. Δ^2 * Anorm / visc
        eps_sfs = .-contract_dissipation(τ_ref, S, g)            # -τ:S
        cε = @. eps_sfs / (Δ^2 * Anorm^3 + ϵ)
        τF2 = contract_dissipation(τ_ref, τ_ref, g)             # ‖τ‖_F²
        cσ = @. sqrt(τF2) / (Δ^2 * A2 + ϵ)

        idx = 1:subsample:length(reδ)
        append!(re, view(vec(Array(reδ)), idx))
        append!(cdiss, view(vec(Array(cε)), idx))
        append!(cstress, view(vec(Array(cσ)), idx))
    end

    # Log-spaced bins between robust quantiles of Re_Δ (heavy-tailed |Ā|).
    lo, hi = quantile(re, quantiles[1]), quantile(re, quantiles[2])
    edges = 10.0 .^ range(log10(lo), log10(hi); length = nbin + 1)
    centers = sqrt.(edges[1:(end - 1)] .* edges[2:end])
    bin = map(re) do x
        lo ≤ x ≤ hi ? clamp(searchsortedlast(edges, x), 1, nbin) : 0
    end

    binstats(samples) = let per = [Float64[] for _ in 1:nbin]
        for (b, v) in zip(bin, samples)
            b == 0 || push!(per[b], v)
        end
        (;
            median = map(b -> isempty(b) ? NaN : median(b), per),
            q25 = map(b -> isempty(b) ? NaN : quantile(b, 0.25), per),
            q75 = map(b -> isempty(b) ? NaN : quantile(b, 0.75), per),
            count = map(length, per),
        )
    end
    diss = binstats(cdiss)
    stress = binstats(cstress)

    # Within-flow slope: count-weighted least squares of the bin median against
    # log10(Re_Δ) over non-empty bins. The *sign* is the deliverable.
    function slope(st)
        ok = findall(b -> st.count[b] > 0 && isfinite(st.median[b]), 1:nbin)
        x, y, w = log10.(centers[ok]), st.median[ok], Float64.(st.count[ok])
        wmean(a) = sum(w .* a) / sum(w)
        return sum(w .* (x .- wmean(x)) .* (y .- wmean(y))) / sum(w .* (x .- wmean(x)) .^ 2)
    end
    slopes = (; diss = slope(diss), stress = slope(stress))

    save_object_atomic(
        file,
        (;
            centers, edges, diss, stress, slope = slopes,
            npoint = length(re), re_range = (; lo, hi, quantiles),
        ),
    )
    @info "Within-flow slope per decade of Re_Δ — " *
        "C_ε: $(round(slopes.diss; sigdigits = 3)), " *
        "‖τ‖_norm: $(round(slopes.stress; sigdigits = 3)). " *
        "Compare the sign to the across-flow trend in fig:dissipation-vs-re " *
        "(same sign ⇒ the pointwise Re_Δ feature is worth pursuing)."
    flush(stderr)
    return
end

# --- Direct conditional-mean estimate (the one-point optimal closure) ---
#
# Notes/ExperimentFollowups.md item 1: nonparametrically estimate
# `E[τ̃ | normalized VGT]` on the a-priori data and evaluate its relative tensor
# error with the same metric as `compute_sfs_stats`. If it lands on the shared
# a-priori floor of the learned closures, the floor IS the Langford–Moser
# one-point optimal closure by construction.

"""
TBNN-style feature/target construction for one filtered-DNS snapshot, shared by
the conditional-mean fit and evaluation: the gradient invariants λ, the tensor
basis `b`, `|Ā|²`, and the physical trace-free reference stress (stacked,
`(nx…, tensordim)`), all on the backend. Mirrors [`create_dataloader_tbnn`](@ref)
(dealiased gradient, normalized VGT) and the `:ref` recipe of
[`compute_sfs_stats`](@ref).
"""
function condmean_snapshot(case, g, input, output)
    u = vectorfield(g)
    τhat = scalarfield(g)
    ττ = spacetensorfield(g)
    plan = plan_rfft(ττ.xx)
    foreach(copyto!, u, input)
    G = getgradient(u, g)
    i, b, a2 = build_tensorbasis(G, g)
    for (τ, τcpu) in zip(ττ, output)
        copyto!(τhat, τcpu)
        apply!(twothirds!, g, (τhat, g))
        to_phys!(τ, τhat, plan, g)
    end
    make_tracefree!(ττ, g)
    return i, b, a2, stack(ττ)
end

"""
Per-dimension quantile bin edges (`nq + 1` values each) of the *independent*
gradient invariants over a strided subsample of the `pool` datasets. The first
invariant is dropped: after the `|Ā| = 1` normalization the invariants satisfy
`λ₁ − λ₂ = 1` (`tr(S²) − tr(R²) = |Ā|²`), so the feature space is the
`(ninvariant − 1)`-dimensional manifold spanned by `λ₂…` (4-D in 3D). Returned
as a tuple so the hot accumulation/eval loops specialize on the feature count.
"""
function condmean_edges(case, pool; nq, stride)
    (; D, l, n_les, backend) = case
    g = Grid{D}(; l, n = n_les, backend)
    ni = ninvariant(g)
    samples = [Float64[] for _ in 1:(ni - 1)]
    for (dns, Δf) in pool
        inputs = load(fieldsfile(case, dns, Δf), "inputs")
        u = vectorfield(g)
        for j in eachindex(inputs)
            foreach(copyto!, u, inputs[j])
            G = getgradient(u, g)
            i, _, _ = build_tensorbasis(G, g)
            icpu = reshape(Array(i), :, ni)
            for d in eachindex(samples)
                append!(samples[d], @view icpu[1:stride:end, d + 1])
            end
        end
    end
    return Tuple(map(s -> quantile(s, range(0, 1; length = nq + 1)), samples))
end

"Cell of the invariants at point `p` (row of `icpu`) on the quantile grid `edges`."
@inline cell_of(icpu, p, edges::NTuple{NF, Vector{Float64}}, nq) where {NF} =
    CartesianIndex(
    ntuple(d -> clamp(searchsortedlast(edges[d], icpu[p, d + 1]), 1, nq), Val(NF)),
)

"""
Accumulate the per-cell least-squares normal equations of the basis-coefficient
fit `ỹ ≈ Σₖ cₖ bₖ` over one snapshot's points: `A[l,k,cell] += Σ_c bₖ[c] bₗ[c]`
(lower triangle) and `r[k,cell] += Σ_c bₖ[c] ỹ[c]`, with the packed tensor
components weighted equally — the same (unweighted stacked-component) norm as the
training loss and the [`compute_sfs_stats`](@ref) error metric.
"""
function accumulate_condmean!(A, r, cnt, icpu, bcpu, ycpu, edges, stride)
    nq = size(cnt, 1)
    npt, nt, nb = size(bcpu)
    @inbounds for p in 1:stride:npt
        I = cell_of(icpu, p, edges, nq)
        cnt[I] += 1
        for k in 1:nb
            rk = 0.0
            for c in 1:nt
                rk += bcpu[p, c, k] * ycpu[p, c]
            end
            r[k, I] += rk
            for l in k:nb
                akl = 0.0
                for c in 1:nt
                    akl += bcpu[p, c, k] * bcpu[p, c, l]
                end
                A[l, k, I] += akl
            end
        end
    end
    return
end

"Sum the per-cell normal equations into the next-coarser (halved) dyadic grid."
function coarsen_condmean(A, r, cnt)
    nb = size(A, 1)
    n2 = size(cnt, 1) ÷ 2
    dims = ntuple(Returns(n2), ndims(cnt))
    A2 = zeros(nb, nb, dims...)
    r2 = zeros(nb, dims...)
    c2 = zeros(Int, dims...)
    for I in CartesianIndices(cnt)
        J = CartesianIndex(cld.(Tuple(I), 2))
        @views A2[:, :, J] .+= A[:, :, I]
        @views r2[:, J] .+= r[:, I]
        c2[J] += cnt[I]
    end
    return A2, r2, c2
end

"""
Solve the per-cell least squares on the dyadic hierarchy `nq = 1, 2, …, nq_max`
(coarser levels are block sums of the finest accumulator, and the coarse edges
are every-2ᵐ-th quantile edge, so the levels nest exactly). Cells with fewer
than `minbincount` points inherit the enclosing parent cell's coefficients —
the estimate degrades gracefully toward the coarser conditional mean in sparse
tails instead of going singular. Returns the levels coarsest → finest as
`(; nq, coefs, solved, count)`.
"""
function solve_condmean_levels(A, r, cnt, minbincount)
    nb = size(A, 1)
    hierarchy = [(A, r, cnt)]
    while size(last(hierarchy)[3], 1) > 1
        push!(hierarchy, coarsen_condmean(last(hierarchy)...))
    end
    levels = []
    parent = nothing
    for (Aq, rq, cq) in reverse(hierarchy)
        coefs = zeros(nb, size(cq)...)
        solved = falses(size(cq))
        for I in CartesianIndices(cq)
            if cq[I] >= minbincount
                M = Matrix(@view Aq[:, :, I])
                for k in 1:nb, l in (k + 1):nb
                    M[k, l] = M[l, k]   # only the lower triangle was accumulated
                end
                # Tiny Tikhonov ridge: a cell dominated by (near-)degenerate
                # basis configurations would otherwise be singular.
                λ = 1.0e-8 * tr(M) / nb + eps()
                for k in 1:nb
                    M[k, k] += λ
                end
                coefs[:, I] = M \ rq[:, I]
                solved[I] = true
            elseif !isnothing(parent)
                coefs[:, I] = parent.coefs[:, CartesianIndex(cld.(Tuple(I), 2))]
            end
        end
        level = (; nq = size(cq, 1), coefs, solved, count = cq)
        push!(levels, level)
        parent = level
    end
    return levels
end

"""
Fit the binned conditional-mean estimator on the datasets in `pool` (each a
`(dns, Δ_factor)` coordinate): stream every snapshot once, accumulating the
per-cell normal equations on the `edges` grid, then solve the full dyadic level
hierarchy. Returns `(; edges, levels, npoints)`.
"""
function condmean_fit(case, pool, edges; minbincount, stride)
    (; D, l, n_les, backend) = case
    g = Grid{D}(; l, n = n_les, backend)
    nb, nt, ni = nbasis(g), tensordim(g), ninvariant(g)
    nq = length(first(edges)) - 1
    dims = ntuple(Returns(nq), length(edges))
    A = zeros(nb, nb, dims...)
    r = zeros(nb, dims...)
    cnt = zeros(Int, dims...)
    ϵ = eps(typeof(l))
    for (dns, Δf) in pool
        inputs, outputs, Δ =
            load(fieldsfile(case, dns, Δf), "inputs", "outputs", "Δ")
        @info "Conditional mean: accumulating $(length(inputs)) snapshots " *
            "(visc=$(dns.visc), seed=$(dns.seed), Δf=$(Δf))"
        flush(stderr)
        for j in eachindex(inputs)
            i, b, a2, y = condmean_snapshot(case, g, inputs[j], outputs[j])
            ỹ = y ./ (Δ^2 .* reshape(a2, size(a2)..., 1) .+ ϵ)   # dataloader normalization
            icpu = reshape(Array(i), :, ni)
            bcpu = reshape(Array(b), :, nt, nb)
            ycpu = reshape(Array(ỹ), :, nt)
            accumulate_condmean!(A, r, cnt, icpu, bcpu, ycpu, edges, stride)
        end
    end
    return (; edges, levels = solve_condmean_levels(A, r, cnt, minbincount), npoints = sum(cnt))
end

"""
Predict the normalized stress at every point from level `lev`'s per-cell
coefficients (`f = nq_max ÷ lev.nq` maps the finest-level cells onto this
level's) and restore the physical `Δ²|Ā|²` scale. Returns the number of points
that landed in a cell with inherited (not directly solved) coefficients.
"""
function predict_condmean!(pred, bcpu, scale, fine, lev, f)
    (; coefs, solved) = lev
    npt, nt = size(pred)
    nb = size(bcpu, 3)
    nfallback = 0
    @inbounds for p in 1:npt
        J = CartesianIndex(cld.(Tuple(fine[p]), f))
        solved[J] || (nfallback += 1)
        for c in 1:nt
            acc = 0.0
            for k in 1:nb
                acc += coefs[k, J] * bcpu[p, c, k]
            end
            pred[p, c] = acc * scale[p]
        end
    end
    return nfallback
end

"""
Evaluate the fitted estimator `est` a priori on test dataset (dns, Δf), at every
level of the hierarchy, with the *same* metric pipeline as
[`compute_sfs_stats`](@ref): the predicted stress is dealiased, made trace-free,
and compared against the filtered-DNS reference per snapshot (relative tensor
error + cross-correlation, averaged over snapshots). Returns one summary
NamedTuple per level (coarsest → finest).
"""
function condmean_eval(case, dns, Δf, est)
    (; D, l, n_les, backend) = case
    g = Grid{D}(; l, n = n_les, backend)
    Δ = Δf * l / n_les
    nb, nt, ni = nbasis(g), tensordim(g), ninvariant(g)
    nqmax = est.levels[end].nq
    ϵ = eps(typeof(l))

    inputs, outputs = load(fieldsfile(case, dns, Δf), "inputs", "outputs")
    τ_model = spacetensorfield(g)
    τhat = scalarfield(g)
    plan = plan_rfft(τ_model.xx)

    nlev = length(est.levels)
    relerrs = [Float64[] for _ in 1:nlev]
    crosscors = [Float64[] for _ in 1:nlev]
    nfallback = zeros(Int, nlev)
    npoint = 0
    for j in eachindex(inputs)
        i, b, a2, y = condmean_snapshot(case, g, inputs[j], outputs[j])
        icpu = reshape(Array(i), :, ni)
        bcpu = reshape(Array(b), :, nt, nb)
        scale = vec(Array(a2))
        @. scale = Δ^2 * scale + ϵ
        yref = Array(y)
        npt = length(scale)
        npoint += npt
        fine = [cell_of(icpu, p, est.edges, nqmax) for p in 1:npt]
        pred = Matrix{Float64}(undef, npt, nt)
        for (ilev, lev) in enumerate(est.levels)
            nfallback[ilev] +=
                predict_condmean!(pred, bcpu, scale, fine, lev, nqmax ÷ lev.nq)
            predgpu = adapt(backend, reshape(pred, size(a2)..., nt))
            foreach(copyto!, τ_model, unstack_symtensor(predgpu, g))
            for τ in τ_model
                dealias_phys!(τ, τhat, plan, g)
            end
            make_tracefree!(τ_model, g)
            bb = Array(stack(τ_model))
            push!(relerrs[ilev], norm(bb - yref) / norm(yref))
            bbc, aac = bb .- mean(bb), yref .- mean(yref)
            push!(crosscors[ilev], dot(bbc, aac) / sqrt(dot(bbc, bbc) * dot(aac, aac)))
        end
    end
    return map(enumerate(est.levels)) do (ilev, lev)
        (;
            lev.nq,
            ncell = length(lev.count),
            nsolved = count(lev.solved),
            relerr = mean(relerrs[ilev]),
            crosscor = mean(crosscors[ilev]),
            fallback = nfallback[ilev] / npoint,
            relerrs = relerrs[ilev],
        )
    end
end

"""
Direct nonparametric estimate of the one-point optimal (conditional-mean) closure
`E[τ̃ | Ā/|Ā|]` and its a-priori error on test dataset (dns, Δf) —
Notes/ExperimentFollowups.md item 1. Post-processing only: no DNS, no training.

**Estimator.** The TBNN feature space: bin the independent invariants λ of the
normalized velocity gradient ([`condmean_edges`](@ref)) and least-squares the 7
tensor-basis coefficients per bin on the *normalized* target `τ/(Δ²|Ā|²)` — the
exact regression target of the dataloaders. Per bin, that LS projection is the
sample conditional mean restricted to the equivariant (tensor-basis) function
class, i.e. the binned Langford–Moser optimal one-point closure with the
`Δ²|Ā|²` prefactor the learned closures share. The bin-resolution hierarchy
`nq = 1, 2, …, nq_max` per invariant is the built-in binning-sensitivity check:
if the a-priori error plateaus in `nq` at the learned closures' shared floor,
the floor *is* the one-point optimal closure.

**Fits.** Two estimators are evaluated: `train` fits on `fitpool` (default: the
full training pool — statistically the same forced HIT and exactly the data the
networks were fit on) and `self` fits in-sample on the test dataset itself (the
direct conditional mean of that flow; with millions of points per bin the
in-sample bias is negligible). Agreement between the two is a further
robustness check.

Persists to [`condmeanfile`](@ref) `(; train, self, …)`, each a per-level vector
of `(; nq, ncell, nsolved, relerr, crosscor, fallback, relerrs)`;
[`plot_condmean`](@ref) draws the error-vs-resolution curves against the
saturated learned floor.
"""
function compute_condmean(
        case, dns, Δf;
        fitpool = build_trainpool(case),
        nq_max = 16, minbincount = 32,
        stride_edges = 17, stride_fit = 1,
        selffit = true, force = false,
    )
    @assert ispow2(nq_max) "the level hierarchy is dyadic; nq_max must be a power of 2"
    file = condmeanfile(case, dns, Δf)
    skip_if_cached(file; force, label = "conditional-mean estimate") && return

    function estimate(pool, label)
        @info "Conditional mean ($label fit): computing invariant bin edges over $(length(pool)) dataset(s)"
        flush(stderr)
        edges = condmean_edges(case, pool; nq = nq_max, stride = stride_edges)
        est = condmean_fit(case, pool, edges; minbincount, stride = stride_fit)
        @info "Conditional mean ($label fit): evaluating on (visc=$(dns.visc), seed=$(dns.seed), Δf=$(Δf))"
        flush(stderr)
        return (; levels = condmean_eval(case, dns, Δf, est), est.npoints)
    end

    train = estimate(fitpool, "train")
    self = selffit ? estimate([(dns, Δf)], "self") : nothing

    save_object_atomic(
        file,
        (;
            train, self,
            fitpool = [(; d.visc, d.seed, Δf = f) for (d, f) in fitpool],
            nq_max, minbincount, stride_edges, stride_fit,
        ),
    )
    finest = last(train.levels)
    @info "Conditional-mean a-priori relative error at (visc=$(dns.visc), seed=$(dns.seed), Δf=$(Δf)): " *
        "$(round(finest.relerr; sigdigits = 4)) (train fit, nq=$(finest.nq), " *
        "fallback $(round(100 * finest.fallback; sigdigits = 2))%)" *
        (isnothing(self) ? "" : ", $(round(last(self.levels).relerr; sigdigits = 4)) (self fit)") *
        ". Compare against the learned saturation floor (fig. condmean)."
    flush(stderr)
    return
end

"""
A-priori equivariance error for closure `m` on test dataset (dns, Δf): compares
`R(model(G))` with `model(R(G))` for every octahedral group element, averaged over
the full filtered-DNS snapshot series. Persists the per-element error series (mean
over snapshots) to `equipriorfile(case, dns, Δf, m)`; [`get_seed_statistics`](@ref)
reduces it to the per-family mean that the errors table
([`write_errors_table`](@ref)) consumes.
"""
function apriori_equivariance_error(case, m, dns, Δf, getmodel; force = false)
    file = equipriorfile(case, dns, Δf, m)
    return cached(file; force, label = "a-priori equi errors ($(modelname(m)))") do
        model = getmodel()
        (; D, l, n_les, backend) = case
        (; elements, permutations, signs) = group_stuff(D)
        g = Grid{D}(; l, n = n_les, backend)
        inputs = load(fieldsfile(case, dns, Δf), "inputs")
        u = vectorfield(g)
        @info "Computing a-priori equi errors for $(modelname(m)) over $(length(inputs)) snapshots"
        flush(stderr)
        # Per-element error accumulated over the snapshot series, then averaged.
        err = zeros(length(elements))
        for i in eachindex(inputs)
            foreach(copyto!, u, inputs[i])
            G = getgradient(u, g)
            # Copy: dynamic Smagorinsky reuses an internal τ buffer across calls.
            mG = copy(model(u, G))
            mG_split = unstack_symtensor(mG, g)
            for (j, e) in enumerate(elements)
                ip, is = e
                p, s = permutations[ip], signs[is]
                # Rotate u in physical space, return to spectral, then recompute G
                space_u = inverse_vector_fourier(u, g)
                space_ru = transform_vector(space_u, g, (p, s))
                ru = forward_vector_fourier(space_ru, g)
                foreach(u -> apply!(twothirds!, g, (u, g)), ru)
                rG = getgradient(ru, g)
                mrG = model(ru, rG)
                rmG_split = transform_tensor(mG_split, g, (p, s))
                rmG = stack(rmG_split)
                err[j] += norm(rmG - mrG) / norm(mrG)
            end
        end
        err ./= length(inputs)
        save_object_atomic(file, err)
        err
    end
end

"""
Snapshot count of a *complete* a-posteriori rollout at `(dns, Δf)`. The `:ref`
reduction never integrates (it just reduces the filtered-DNS series), so it always
reaches every scheduled snapshot — the reliable bar for divergence detection.
"""
apost_nfull(case, dns, Δf) = length(load_object(apostfile(case, dns, Δf, :ref)).e_post)

"""
`true` if model `m`'s a-posteriori rollout at `(dns, Δf)` bailed out early. The
`solve_les` instability guard (`les.jl`) returns the moment the rollout blows up,
so `e_post` holds only the pre-blow-up prefix and is shorter than a complete
rollout ([`apost_nfull`](@ref)). A truncated rollout's `mean(e_post)` averages just
that prefix — artificially low when the blow-up is sudden — so callers treat a
diverged point as `missing` rather than plotting/tabulating the mean.
"""
function apost_diverged(case, dns, Δf, m)
    f = apostfile(case, dns, Δf, m)
    isfile(f) || return false
    return length(load_object(f).e_post) < apost_nfull(case, dns, Δf)
end

"""
Divergence-aware time-mean a-posteriori solution error: `mean(e_post)` for a
completed rollout, `missing` if the rollout diverged ([`apost_diverged`](@ref)) or
no artifact exists. Use this instead of a bare `mean(load_object(apostfile(...)).e_post)`.
"""
function apost_emean(case, dns, Δf, m)
    f = apostfile(case, dns, Δf, m)
    isfile(f) || return missing
    ep = load_object(f).e_post
    return length(ep) < apost_nfull(case, dns, Δf) ? missing : mean(ep)
end

"""
Aggregate the netseed spread of the scalar closure metrics at evaluation point
(dns, Δf), for each learned-model family in `families` (each `(; arch, tier,
use_redelta)`). For every family and every seed in `netseeds`, reads the per-seed
a-priori stats ([`sfsstatsfile`](@ref)), a-priori equivariance error
([`equipriorfile`](@ref)) and a-posteriori rollout ([`apostfile`](@ref)), and
collects, per family (keyed by [`familyname`](@ref)), vectors over seeds of:

- `relerr`, `crosscor` — a-priori SFS tensor error / cross-correlation;
- `diss_median` — median pointwise SFS dissipation normalized by `:ref`;
- `backscatter` — local backscatter fraction;
- `equi` — mean a-priori equivariance error (`missing` if not computed);
- `e_post` — time-mean a-posteriori solution error (`missing` if no rollout);
- `e_post_series` — the full error-vs-time series per seed (for spread bands).

Persists the aggregate to [`seedstatsfile`](@ref)`(case, dns, Δf)`. Missing
artifacts are skipped with a warning, so a partially completed sweep still
aggregates; rerun with `force = true` once more seeds have landed.
"""
function get_seed_statistics(case, families, dns, Δf, netseeds; force = false)
    file = seedstatsfile(case, dns, Δf)
    return cached(file; force, label = "seed statistics (Δ=$(Δf))") do
        refmed = load_object(sfsstatsfile(case, dns, Δf, :ref)).diss.median

        stat = map(collect(families)) do fam
            rows = map(collect(netseeds)) do s
                m = (; fam..., netseed = s)
                fst = sfsstatsfile(case, dns, Δf, m)
                if !isfile(fst)
                    @warn "Missing $(fst); skipping seed $(s) for $(familyname(fam))"
                    return nothing
                end
                st = load_object(fst)
                fe = equipriorfile(case, dns, Δf, m)
                equi = isfile(fe) ? mean(load_object(fe)) : missing
                fa = apostfile(case, dns, Δf, m)
                eser = isfile(fa) ? load_object(fa).e_post : missing
                (;
                    seed = s,
                    relerr = st.apriori.relerr,
                    crosscor = st.apriori.crosscor,
                    diss_median = st.diss.median / refmed,
                    backscatter = st.diss.backscatter,
                    equi,
                    # Divergence-aware: a truncated (blown-up) rollout's mean is
                    # meaningless, so the scalar is `missing`; the raw series is kept
                    # for the error-vs-time plots (which should show the blow-up).
                    e_post = apost_emean(case, dns, Δf, m),
                    e_post_series = eser,
                )
            end
            rows = filter(!isnothing, rows)
            familyname(fam) => (;
                seeds = [r.seed for r in rows],
                relerr = [r.relerr for r in rows],
                crosscor = [r.crosscor for r in rows],
                diss_median = [r.diss_median for r in rows],
                backscatter = [r.backscatter for r in rows],
                equi = [r.equi for r in rows],
                e_post = [r.e_post for r in rows],
                e_post_series = [r.e_post_series for r in rows],
            )
        end |> NamedTuple
        save_object_atomic(file, stat)
        stat
    end
end

"""
Format a collection of per-seed values as `mean ± std` for the text tables
(the `±` is omitted when fewer than two values survive `skipmissing`).
"""
function pm_string(vals; sigdigits = 4)
    v = collect(skipmissing(vals))
    isempty(v) && return "missing"
    m = round(mean(v); sigdigits)
    length(v) == 1 && return string(m)
    return string(m, " ± ", round(std(v); sigdigits = 2))
end
