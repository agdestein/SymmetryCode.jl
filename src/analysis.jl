# Post-hoc evaluation of LES closures: SFS prediction series, KDE densities,
# a-priori errors, equivariance commutation errors, dissipation comparisons,
# and Q-R invariant joint distributions.

"""
Predict the SFS stress series for a single closure keyed under `key` by applying
it to every eval-window filtered-DNS snapshot. `getmodel` is a zero-arg thunk
that builds the closure; it is only invoked on a cache miss. Persists the
resulting `Vector{NamedTuple}` to `sfs_<key>.jld2`; returns nothing.
"""
function predict_sfs(setup, key, getmodel; force = false)
    (; outdir, D, l, n_les, backend) = setup
    file = "$(outdir)/sfs_$(key).jld2"
    skip_if_cached(file; force, label = "SFS for $(key)") && return
    model = getmodel()

    data = joinpath(setup.outdir, "data.jld2") |> load_object
    inputs_eval = data.inputs[data_ranges(setup).eval]

    g = Grid{D}(; l, n = n_les, backend)
    u = vectorfield(g)
    plan = plan_rfft(spacescalarfield(g))
    A = tensorfield_nonsym(g)
    AA = spacetensorfield_nonsym(g)

    @info "Computing SFS for $(key)"
    flush(stderr)
    τ_series = map(inputs_eval) do ucpu
        GC.gc()
        CUDA.reclaim()

        # Velocity gradient in physical space
        foreach(copyto!, u, ucpu)
        apply!(vectorgradient!, g, (A, u, g))
        for (AA, A) in zip(AA, A)
            apply!(twothirds!, g, (A, g))
            to_phys!(AA, A, plan, g)
        end

        # Prediction by LES model
        unstack_symtensor(model(u, AA), g) |> cpu_device()
    end
    save_object_atomic(file, τ_series)
    return
end

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
Aggregate per-snapshot SFS samples into a single statistics artifact per key.

For each key in `keys`, persists `sfs_stats_<key>.jld2` containing:

- `kde.xx`, `kde.xy`, `kde.diss` — kernel density estimates of the τ₁₁,
  τ₁₂, and pointwise SFS dissipation rate `ε_sfs = -τᵢⱼSᵢⱼ` distributions
  over the eval window. **Convention** (shared by every dissipation-flavored
  statistic in this pipeline, including [`compute_budget`](@ref) and
  [`compute_spectral_transfer`](@ref)): `ε_sfs > 0` is drain on resolved
  KE; `ε_sfs < 0` is local backscatter.
- `diss` — `(; mean, median, std, skewness, backscatter)` of the pointwise
  `ε_sfs` samples. `backscatter` is the fraction of points with `ε_sfs < 0`
  (equivalently `τᵢⱼSᵢⱼ > 0`); it is exactly 0 for any closure of the form
  `τ = -2νₜS` with `νₜ ≥ 0` (e.g. Smagorinsky), and is ≈ 0.3–0.4 for
  filtered HIT.
- `apriori` — `(; relerr, crosscor)` against the filtered-DNS reference over
  the full trace-free tensor. `(0.0, 1.0)` for `:ref`, `(1.0, 0.0)` for `:nomo`.

`:ref` reads its tensor from `data.outputs[eval]` directly; other keys load
`sfs_<key>.jld2` (produced by [`predict_sfs`](@ref)). `:nomo` is treated as
zero SFS stress (no `sfs_nomo.jld2` is required).

Replaces the legacy `compute_densities` / `apriori_error` /
`get_dissipation_errors` trio, which all iterated the same per-snapshot
sample collection separately.
"""
function compute_sfs_stats(setup, keys; force = false)
    (; outdir, D, l, n_les, backend) = setup

    todo = filter(k -> force || !isfile("$(outdir)/sfs_stats_$(k).jld2"), keys)
    if isempty(todo)
        @info "All SFS stats cached"
        flush(stderr)
        return
    end

    data = joinpath(setup.outdir, "data.jld2") |> load_object
    eval_range = data_ranges(setup).eval
    inputs_eval = data.inputs[eval_range]
    outputs_eval = data.outputs[eval_range]

    g = Grid{D}(; l, n = n_les, backend)
    u = vectorfield(g)
    τ_model = spacetensorfield(g)
    τ_ref = spacetensorfield(g)
    dissfield = spacescalarfield(g)
    τhat = scalarfield(g)
    plan = plan_rfft(τ_model.xx)
    A = tensorfield_nonsym(g)
    AA = spacetensorfield_nonsym(g)

    for key in todo
        file = "$(outdir)/sfs_stats_$(key).jld2"
        @info "Computing SFS stats for $(key)"
        flush(stderr)

        # Per-snapshot model source. :nomo has no file; :ref uses the
        # spectral-space reference (re-IFFTed in the loop). Others load
        # the cached physical-space prediction.
        τ_series = if key in (:nomo, :ref)
            nothing
        else
            load_object("$(outdir)/sfs_$(key).jld2")
        end

        xx_samples = Float64[]
        xy_samples = Float64[]
        diss_samples = Float64[]
        relerrs = Float64[]
        crosscors = Float64[]

        for i in eachindex(inputs_eval)
            # Strain rate from ubar (model-independent; recomputed per key
            # for memory simplicity — cheap compared to the model work).
            foreach(copyto!, u, inputs_eval[i])
            apply!(vectorgradient!, g, (A, u, g))
            for (AA, A) in zip(AA, A)
                apply!(twothirds!, g, (A, g))
                to_phys!(AA, A, plan, g)
            end
            S = strain_from_gradient(AA, g)

            # Reference τ (spectral → physical, trace-free)
            for (τ, τcpu) in zip(τ_ref, outputs_eval[i])
                copyto!(τhat, τcpu)
                apply!(twothirds!, g, (τhat, g))
                to_phys!(τ, τhat, plan, g)
            end
            make_tracefree!(τ_ref, g)

            # Model τ
            if key == :nomo
                for t in τ_model
                    fill!(t, 0)
                end
            elseif key == :ref
                for (t, r) in zip(τ_model, τ_ref)
                    copyto!(t, r)
                end
            else
                foreach(copyto!, τ_model, τ_series[i])
                for t in τ_model
                    dealias_phys!(t, τhat, plan, g)
                end
                make_tracefree!(τ_model, g)
            end

            # Sample the three fields of interest.
            # ε_sfs = -τᵢⱼSᵢⱼ is the SFS dissipation rate (positive = drain).
            append!(xx_samples, vec(Array(τ_model.xx)))
            append!(xy_samples, vec(Array(τ_model.xy)))
            dissfield .= contract_dissipation(τ_model, S, g)
            append!(diss_samples, .-vec(Array(dissfield)))

            # A-priori metrics over the full trace-free tensor vs reference
            if key == :ref
                push!(relerrs, 0.0)
                push!(crosscors, 1.0)
            elseif key == :nomo
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

        # Aggregate. KDE the three sample arrays (zero-variance :nomo is
        # handled by saving an empty estimate; plot_densities filters :nomo out).
        kde_or_empty(samples) = if iszero(std(samples))
            (; x = Float64[], density = Float64[])
        else
            e = kde(samples)
            (; x = collect(e.x), density = collect(e.density))
        end
        result = (;
            kde = (;
                xx = kde_or_empty(xx_samples),
                xy = kde_or_empty(xy_samples),
                diss = kde_or_empty(diss_samples),
            ),
            diss = (;
                moments(diss_samples)...,
                # Fraction of points with ε_sfs < 0 (local backscatter).
                # Exactly 0 for any closure of the form τ = -2νₜS with
                # νₜ ≥ 0 (e.g. Smag), since then ε_sfs = 2νₜ|S|² ≥ 0.
                backscatter = mean(<(0), diss_samples),
            ),
            apriori = (; relerr = mean(relerrs), crosscor = mean(crosscors)),
        )
        save_object_atomic(file, result)
    end
    return
end

"""
A-priori equivariance error for a single closure: compares `R(model(G))`
with `model(R(G))` for every octahedral group element. Persists to
`equi-errors-prior-<key>.jld2`; load all keys back with
[`load_equivariance_errors`](@ref) for plotting.
"""
function apriori_equivariance_error(setup, key, getmodel; force = false)
    file = "$(setup.outdir)/equi-errors-prior-$(key).jld2"
    return cached(file; force, label = "a-priori equi errors ($(key))") do
        model = getmodel()
        (; D, l, n_les, backend) = setup
        (; elements, permutations, signs) = group_stuff(D)
        g = Grid{D}(; l, n = n_les, backend)
        data = joinpath(setup.outdir, "data.jld2") |> load_object
        u = map(copy, data.inputs[data_ranges(setup).eval[1]]) |> adapt(setup.backend)
        G = getgradient(u, g)
        @info "Computing a-priori equi errors for $(key)"
        flush(stderr)
        # Copy: dynamic Smagorinsky reuses an internal τ buffer across calls.
        mG = copy(model(u, G))
        mG_split = unstack_symtensor(mG, g)
        err = map(elements) do e
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
            norm(rmG - mrG) / norm(mrG)
        end
        save_object_atomic(file, err)
        err
    end
end

@kernel function qr_kernel!(q, r, GG, g::Grid{3})
    T = eltype(q)
    I = @index(Global, Cartesian)
    G = SMatrix{3, 3, T, 9}(
        GG.xx[I],
        GG.yx[I],
        GG.zx[I],
        GG.xy[I],
        GG.yy[I],
        GG.zy[I],
        GG.xz[I],
        GG.yz[I],
        GG.zz[I],
    )
    q[I] = -tr(G * G) / 2
    r[I] = -tr(G * G * G) / 3
end

"""
Compute Q-R invariant KDEs from post-processed velocity-gradient fields.

The saved samples are nondimensionalized by the mean Kolmogorov time of the
LES-filtered reference data before density estimation.
"""
function compute_qr(setup, modelkeys; force = false)
    (; D, l, n_les, backend) = setup

    data = joinpath(setup.outdir, "data.jld2") |> load_object
    eval_range = data_ranges(setup).eval

    g = Grid{D}(; l, n = n_les, backend)
    Ghat = scalarfield(g)
    G = spacetensorfield_nonsym(g)
    q = spacescalarfield(g)
    r = spacescalarfield(g)
    u = vectorfield(g)
    plan = plan_rfft(G.xx)

    # Kolmogorov time from the eval window — the q,r nondimensionalization
    # then matches the window being plotted.
    t_kol = mean(x -> x.t_kol, data.statistics_les[eval_range])

    for k in modelkeys
        file = "$(setup.outdir)/qr_$(k).jld2"
        skip_if_cached(file; force, label = "Q-R for $(k)") && continue
        @info "Computing Q-R for $(k)"
        u_series = if k == :ref
            data.inputs[eval_range]
        else
            load_object(upostfile(setup, k)).u
        end

        qr = map(u_series) do ucpu
            foreach(copyto!, u, ucpu)
            for j in 1:D, i in 1:D
                s = [:x, :y, :z]
                ij = Symbol(s[i], s[j])
                apply!(derivative!, g, (Ghat, u[i], j, g))
                apply!(twothirds!, g, (Ghat, g))
                to_phys!(G[ij], Ghat, plan, g)
            end
            apply!(qr_kernel!, g, (q, r, G, g); ndrange = space_ndrange(g))
            qvec = q |> cpu_device() |> vec
            rvec = r |> cpu_device() |> vec
            qvec .*= t_kol^2
            rvec .*= t_kol^3
            qvec, rvec
        end
        qstack = stack(qr -> qr[1], qr) |> vec
        rstack = stack(qr -> qr[2], qr) |> vec
        dens = kde(
            (rstack, qstack);
            npoints = (1000, 1000),
        )
        @info "Saving Q-R density to $(file)"
        save_object_atomic(file, (; dens.x, dens.y, dens.density))
    end
    return nothing
end

"""
A-posteriori equivariance error for a single closure: integrates the LES
forward in time from a filtered-DNS IC under each octahedral group element
and compares against the inverse-rotated reference trajectory. Persists to
`equi-errors-post-<key>.jld2`; load all keys back with
[`load_equivariance_errors`](@ref) for plotting.
"""
function apost_equivariance_error(setup, key, getmodel; force = false, tstop = 1.0e-1)
    file = "$(setup.outdir)/equi-errors-post-$(key).jld2"
    return cached(file; force, label = "a-posteriori equi errors ($(key))") do
        model = getmodel()
        data = joinpath(setup.outdir, "data.jld2") |> load_object
        ustart = data.inputs[end] |> adapt(setup.backend)
        (; indices) = group_stuff(setup.D)
        @info "Computing a-posteriori equi errors for $(key)"
        flush(stderr)
        err = map(indices) do i
            @info "Element $(i) of $(length(indices))"
            flush(stderr)
            test_equivariance_post(
                setup, ustart, model;
                groupindex = i, tstop, dolog = false,
            )
        end
        save_object_atomic(file, err)
        err
    end
end

"""
Load per-key equivariance error series persisted by
[`apriori_equivariance_error`](@ref) / [`apost_equivariance_error`](@ref)
into a NamedTuple suitable for [`plot_equivariance_errors`](@ref).

`tag` is `:prior` or `:post`.
"""
load_equivariance_errors(setup, keys, tag::Symbol) = NamedTuple(
    k => load_object("$(setup.outdir)/equi-errors-$(tag)-$(k).jld2") for k in keys
)

"""
A-posteriori resolved KE / dissipation budget over the eval-window rollout.

For each snapshot, records:
- `ke = ⟨½ uᵢ uᵢ⟩` — resolved kinetic energy.
- `eps_visc = ν ⟨|∇u|²⟩` — viscous dissipation on resolved scales.
- `eps_sfs = -⟨τᵢⱼ Sᵢⱼ⟩` — SFS dissipation rate. Positive = forward
  transfer to unresolved scales (the model removes energy from resolved
  scales); negative = net backscatter at the integral level. Same drain
  convention as [`compute_sfs_stats`](@ref) and
  [`compute_spectral_transfer`](@ref).

For `key == :ref`, the budget is computed from filtered-DNS state
(`data.inputs[eval]`) and reference SFS stress (`data.outputs[eval]`);
no closure is needed (the default `getmodel = () -> nothing` is left as is).
For other keys, the state comes from `u-post-<key>.jld2` and τ is re-evaluated
by the closure that `getmodel` builds on that rolled-out state — so the result
reflects the *closure during integration*, not the a-priori prediction on
filtered DNS. `getmodel` is a zero-arg thunk invoked only on a cache miss (and
not at all for `:ref`/`:nomo`).

Persists `budget_<key>.jld2 :: (; t, ke, eps_visc, eps_sfs)`.
"""
function compute_budget(setup, key, getmodel = () -> nothing; force = false)
    file = "$(setup.outdir)/budget_$(key).jld2"
    skip_if_cached(file; force, label = "budget for $(key)") && return

    (; D, l, n_les, backend, visc) = setup
    g = Grid{D}(; l, n = n_les, backend)
    data = joinpath(setup.outdir, "data.jld2") |> load_object
    eval_range = data_ranges(setup).eval
    times = data.times[eval_range]

    if key == :ref
        u_series = data.inputs[eval_range]
        τ_ref_series = data.outputs[eval_range]
    else
        rollout = load_object(upostfile(setup, key))
        u_series = rollout.u
        τ_ref_series = nothing
    end

    # The closure is only needed for the learned/classical branch; :ref reads the
    # reference τ and :nomo contributes zero SFS, so neither builds a model.
    model = key in (:ref, :nomo) ? nothing : getmodel()

    u = vectorfield(g)
    A = tensorfield_nonsym(g)
    AA = spacetensorfield_nonsym(g)
    τphys = spacetensorfield(g)
    τspec = tensorfield(g)
    diss_spec = KernelAbstractions.zeros(backend, typeof(l), ndrange(g))
    plan = plan_rfft(spacescalarfield(g))

    @info "Computing budget for $(key)"
    flush(stderr)

    nsnap = length(u_series)
    ke = zeros(nsnap)
    eps_visc = zeros(nsnap)
    eps_sfs = zeros(nsnap)

    for i in eachindex(u_series)
        foreach(copyto!, u, u_series[i])
        foreach(u -> apply!(twothirds!, g, (u, g)), u)
        ke[i] = sum(getenergy, u) / 2
        eps_visc[i] = get_dissipation!(diss_spec, u, visc, g)

        if key == :nomo
            eps_sfs[i] = 0.0
            continue
        end

        # Strain rate in physical space, shared by all non-nomo branches
        apply!(vectorgradient!, g, (A, u, g))
        for (AA, A) in zip(AA, A)
            apply!(twothirds!, g, (A, g))
            to_phys!(AA, A, plan, g)
        end
        S = strain_from_gradient(AA, g)

        if key == :ref
            for (t, tcpu) in zip(τspec, τ_ref_series[i])
                copyto!(t, tcpu)
                apply!(twothirds!, g, (t, g))
            end
            for (tp, ts) in zip(τphys, τspec)
                to_phys!(tp, ts, plan, g)
            end
            make_tracefree!(τphys, g)
            τ = τphys
        else
            τstack = model(u, AA)
            τ = unstack_symtensor(τstack, g)
        end

        eps_sfs[i] = -mean(contract_dissipation(τ, S, g))
    end

    save_object_atomic(file, (; t = times, ke, eps_visc, eps_sfs))
    return
end

"""
Eval-window-averaged spectral SFS dissipation rate `ε_sfs(k)` for one model.

Per shell, `ε_sfs(k) = -Σ_{|k'|=k} Re(û_i*(k') · (-i k'ⱼ τ̂_ij(k')))` — the
closure-induced drain on resolved KE at wavenumber `k`. Positive = drain
at that shell (forward transfer to unresolved scales); negative = local
backscatter at that shell. Same drain convention as [`compute_budget`](@ref)
and [`compute_sfs_stats`](@ref). Averaged over the eval window.

`key == :ref` reads spectral τ from `data.outputs[eval]` directly (no
model call). `key == :nomo` writes a zero curve. For other keys, τ is
obtained by re-evaluating the closure on the LES rollout state — same
a-posteriori convention as [`compute_budget`](@ref). `getmodel` is a zero-arg
thunk that builds the closure, invoked only on a cache miss (and not for
`:ref`/`:nomo`).

Persists `transfer_<key>.jld2 :: (; k, eps_sfs)`.
"""
function compute_spectral_transfer(setup, key, getmodel = () -> nothing; force = false)
    file = "$(setup.outdir)/transfer_$(key).jld2"
    skip_if_cached(file; force, label = "spectral transfer for $(key)") && return

    (; D, l, n_les, backend) = setup
    g = Grid{D}(; l, n = n_les, backend)
    data = joinpath(setup.outdir, "data.jld2") |> load_object
    eval_range = data_ranges(setup).eval

    if key == :ref
        u_series = data.inputs[eval_range]
        τ_ref_series = data.outputs[eval_range]
    else
        rollout = load_object(upostfile(setup, key))
        u_series = rollout.u
        τ_ref_series = nothing
    end

    u = vectorfield(g)
    A = tensorfield_nonsym(g)
    AA = spacetensorfield_nonsym(g)
    τphys = spacetensorfield(g)
    τspec = tensorfield(g)
    clo = vectorfield(g)
    plan = plan_rfft(spacescalarfield(g))
    stuff = spectral_stuff(g)
    Tlocal = KernelAbstractions.zeros(backend, typeof(l), ndrange(g))

    @info "Computing spectral transfer for $(key)"
    flush(stderr)

    nshells = length(stuff.k)
    T_accum = zeros(nshells)
    nsnap = length(u_series)

    if key == :nomo
        save_object_atomic(file, (; k = collect(stuff.k), eps_sfs = T_accum))
        return
    end

    # Only the learned/classical branch needs the closure; :ref reads its τ from
    # the reference series, so it never builds a model.
    model = key === :ref ? nothing : getmodel()

    for i in eachindex(u_series)
        foreach(copyto!, u, u_series[i])
        foreach(u -> apply!(twothirds!, g, (u, g)), u)

        if key == :ref
            for (t, tcpu) in zip(τspec, τ_ref_series[i])
                copyto!(t, tcpu)
                apply!(twothirds!, g, (t, g))
            end
        else
            apply!(vectorgradient!, g, (A, u, g))
            for (AA, A) in zip(AA, A)
                apply!(twothirds!, g, (A, g))
                to_phys!(AA, A, plan, g)
            end
            τstack = model(u, AA)
            foreach(copyto!, τphys, unstack_symtensor(τstack, g))
            make_tracefree!(τphys, g)
            for (ts, tp) in zip(τspec, τphys)
                to_spec!(ts, tp, plan, g)
                apply!(twothirds!, g, (ts, g))
            end
        end

        # clo_i = -i kⱼ τ̂_ij is the closure contribution to ∂ₜ û_i;
        # `tensordivergence!` writes exactly that into a vector field.
        apply!(tensordivergence!, g, (clo, τspec, g))

        if D == 3
            @. Tlocal = real(conj(u.x) * clo.x + conj(u.y) * clo.y + conj(u.z) * clo.z)
        else
            @. Tlocal = real(conj(u.x) * clo.x + conj(u.y) * clo.y)
        end

        for (j, shell) in enumerate(stuff.shells)
            T_accum[j] += sum(view(Tlocal, shell))
        end
    end

    # T_accum holds the per-mode contribution to ∂ₜ(½|û|²) summed over
    # shells (positive = backscatter). Negate to switch to the drain
    # convention `ε_sfs(k)` shared by the rest of the pipeline.
    eps_sfs = .-T_accum ./ nsnap
    save_object_atomic(file, (; k = collect(stuff.k), eps_sfs))
    return
end

"""
Persist+cache wrapper around `get_les_statistics`.

The aggregate is written to `les_stat.jld2` under `setup.outdir`. With
`force=false` (default) and the file present, the cached NamedTuple is
returned without recomputing — useful when iterating on plots that
consume it. Set `force=true` (or delete the file) after changing the
active model set.
"""
function get_les_statistics_cached(setup, modelkeys; force = false)
    file = "$(setup.outdir)/les_stat.jld2"
    return cached(file; force, label = "LES statistics") do
        les_stat = get_les_statistics(setup, modelkeys)
        save_object_atomic(file, les_stat)
        les_stat
    end
end

"""
Aggregate the seed-sweep scalar metrics for the learned closures.

For each `key` in `mkeys` and each seed in `seeds`, reads the per-seed
artifacts produced by the `:seeds` stage (`sfs_stats_<skey>.jld2`,
`equi-errors-prior-<skey>.jld2`, `u-post-<skey>.jld2`, where `skey` is
[`seed_key`](@ref); the canonical seed reuses the plain-key artifacts) and
collects, per model, vectors over seeds of:

- `relerr`, `crosscor` — a-priori SFS tensor error / cross-correlation;
- `diss_median` — median pointwise SFS dissipation normalized by `:ref`;
- `backscatter` — local backscatter fraction;
- `equi` — mean a-priori equivariance error (`missing` if not computed);
- `e_post` — time-mean a-posteriori solution error (`missing` if no rollout);
- `e_post_series` — the full error-vs-time series per seed (for spread bands).

Persists the aggregate to `seed_stats.jld2`. Missing artifacts are skipped
with a warning, so a partially completed sweep still aggregates; rerun with
`force = true` once more seeds have landed.
"""
function get_seed_statistics_cached(setup, mkeys, seeds; force = false)
    file = "$(setup.outdir)/seed_stats.jld2"
    return cached(file; force, label = "seed statistics") do
        refmed = load_object("$(setup.outdir)/sfs_stats_ref.jld2").diss.median

        # One get_les_statistics pass for every seed key with a rollout, so
        # the (large) data.jld2 is loaded once rather than per key.
        rolled = [
            sk for key in mkeys
                for sk in (seed_key(setup, key, s) for s in seeds)
                if isfile(upostfile(setup, sk))
        ]
        les = isempty(rolled) ? (;) : get_les_statistics(setup, Tuple(rolled))

        stat = map(collect(mkeys)) do key
            rows = map(collect(seeds)) do s
                skey = seed_key(setup, key, s)
                f = "$(setup.outdir)/sfs_stats_$(skey).jld2"
                if !isfile(f)
                    @warn "Missing $(f); skipping seed $(s) for $(key)"
                    return nothing
                end
                st = load_object(f)
                fe = "$(setup.outdir)/equi-errors-prior-$(skey).jld2"
                equi = isfile(fe) ? mean(load_object(fe)) : missing
                eser = haskey(les, skey) ? les[skey].e_post : missing
                (;
                    seed = s,
                    relerr = st.apriori.relerr,
                    crosscor = st.apriori.crosscor,
                    diss_median = st.diss.median / refmed,
                    backscatter = st.diss.backscatter,
                    equi,
                    e_post = ismissing(eser) ? missing : mean(eser),
                    e_post_series = eser,
                )
            end
            rows = filter(!isnothing, rows)
            key => (;
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
