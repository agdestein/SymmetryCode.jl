# LES time integration: right-hand side with closure, solver, and statistics.

"Compute LES right-hand side with closure model (put force in `du`)."
function les!(du, u, grid, cache; model, visc)
    D = dim(grid)
    (; plan, σ, vi_vj, v, G, GG) = cache

    # Coarse DNS stress
    stress!(σ, vi_vj, v, u, plan, visc, grid)

    # Get VGT in physical space
    apply!(vectorgradient!, grid, (G, u, grid))
    for (GG, G) in zip(GG, G)
        apply!(twothirds!, grid, (G, grid))
        to_phys!(GG, G, plan, grid)
    end

    # Closure model stress (in physical space)
    y = model(u, GG)

    # Add closure stress to existing stress (in spectral space).
    # du.x and vi_vj are scratch.
    for (i, σ) in enumerate(σ)
        copyto!(vi_vj, selectdim(y, D + 1, i))
        to_spec!(du.x, vi_vj, plan, grid)
        σ .+= du.x
    end

    # Final force
    apply!(tensordivergence!, grid, (du, σ, grid))

    return
end

"""
A-posteriori rollout for one closure `m` (a learned coordinate or a classical
symbol) on test dataset (dns, Δf), reducing the metrics **on the fly** and
discarding the heavy LES fields (Notes/ReExperiment.md — storing full LES
trajectories blows up the disk). The LES is integrated from the first
filtered-DNS snapshot across the reference save times; at each save time the
reduced metrics are accumulated. Persists the light `apostfile(case, dns, Δf, m)`:

- `t` — save times reached (truncated if the rollout blew up early).
- `e_post` — relative solution error ‖u_les − ūbar‖/‖ūbar‖ vs the filtered DNS.
- `spectra_les` — resolved energy spectrum per snapshot.
- `ke`, `eps_visc`, `eps_sfs` — resolved KE, viscous dissipation ν⟨|∇u|²⟩, and SFS
  dissipation −⟨τᵢⱼSᵢⱼ⟩ (drain positive) per snapshot.
- `transfer` — `(; k, eps_sfs)`, the eval-mean shell-resolved SFS dissipation ε_sfs(k).

`m === :ref` reduces the filtered-DNS reference directly (no integration; the SFS
stress is the reference `outputs`); `m === :nomo` rolls out with no closure
(eps_sfs ≡ 0). `getmodel` is a zero-arg thunk built only on a cache miss and never
for `:ref`/`:nomo`. With `savefields`, the dealiased LES field series is also
written to `apostfieldsfile` (reserved for the single showcase case).
"""
function solve_les(case, m, dns, Δf, getmodel = () -> nothing; force = false, savefields = false)
    (; D, l, n_les, backend, cfl, forced) = case
    visc = dns.visc
    file = apostfile(case, dns, Δf, m)
    skip_if_cached(file; force, label = "a-posteriori for $(modelname(m))") && return

    inputs, outputs = load(fieldsfile(case, dns, Δf), "inputs", "outputs")
    times = load(dnsmetafile(case, dns), "times")
    @assert length(inputs) == length(times) "fields/metadata snapshot count mismatch"

    g = Grid{D}(; l, n = n_les, backend)
    T = typeof(l)
    stuff = spectral_stuff(g)
    plan = plan_rfft(spacescalarfield(g))

    # Reduction scratch (reused every snapshot).
    u_ref = vectorfield(g)
    diss = KernelAbstractions.zeros(backend, T, ndrange(g))
    A = tensorfield_nonsym(g)
    AA = spacetensorfield_nonsym(g)
    τphys = spacetensorfield(g)
    τspec = tensorfield(g)
    clo = vectorfield(g)
    Tlocal = KernelAbstractions.zeros(backend, T, ndrange(g))

    # On-the-fly accumulators (heavy fields never stored).
    e_post = Float64[]
    ke = Float64[]
    eps_visc = Float64[]
    eps_sfs = Float64[]
    spectra_les = Vector{Float64}[]
    T_accum = zeros(length(stuff.k))
    fields = savefields ? typeof(map(Array, u_ref))[] : nothing

    model = m in (:ref, :nomo) ? nothing : getmodel()

    # Reduce one (dealiased, spectral) LES state `u` against reference snapshot `i`.
    reduce! = function (i, u)
        foreach(copyto!, u_ref, inputs[i])
        foreach(x -> apply!(twothirds!, g, (x, g)), u_ref)
        push!(e_post, norm(stack(u) - stack(u_ref)) / norm(stack(u_ref)))
        push!(ke, sum(getenergy, u) / 2)
        push!(eps_visc, get_dissipation!(diss, u, visc, g))
        push!(spectra_les, spectrum(u, g, stuff).s)

        if m === :nomo
            push!(eps_sfs, 0.0)   # no closure ⇒ no SFS drain or transfer
        else
            # Physical strain rate from the resolved field.
            apply!(vectorgradient!, g, (A, u, g))
            for (AA, A) in zip(AA, A)
                apply!(twothirds!, g, (A, g))
                to_phys!(AA, A, plan, g)
            end
            S = strain_from_gradient(AA, g)

            # Deviatoric SFS stress: reference `outputs` for :ref, else the closure
            # re-evaluated on the rolled-out state (a-posteriori convention).
            if m === :ref
                for (ts, tcpu) in zip(τspec, outputs[i])
                    copyto!(ts, tcpu)
                    apply!(twothirds!, g, (ts, g))
                end
                for (tp, ts) in zip(τphys, τspec)
                    to_phys!(tp, ts, plan, g)
                end
            else
                foreach(copyto!, τphys, unstack_symtensor(model(u, AA), g))
            end
            make_tracefree!(τphys, g)

            push!(eps_sfs, -mean(contract_dissipation(τphys, S, g)))

            # Shell-resolved drain: clo_i = -i kⱼ τ̂_ij is the closure term in ∂ₜûᵢ.
            for (ts, tp) in zip(τspec, τphys)
                to_spec!(ts, tp, plan, g)
                apply!(twothirds!, g, (ts, g))
            end
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

        isnothing(fields) || push!(fields, map(Array, u))
        return
    end

    timing = 0.0
    if m === :ref
        @info "Reducing reference a-posteriori metrics (visc=$(visc), Δ=$(Δf))"
        flush(stderr)
        u = vectorfield(g)
        for i in eachindex(inputs)
            foreach(copyto!, u, inputs[i])
            foreach(x -> apply!(twothirds!, g, (x, g)), u)
            reduce!(i, u)
        end
    else
        @info "Solving LES with $(modelname(m)) (visc=$(visc), Δ=$(Δf))"
        flush(stderr)
        u_model = vectorfield(g)

        # Warm up (compile) on a throwaway run so timing excludes compilation.
        foreach(copyto!, u_model, inputs[1])
        solve_les!(u_model; times = [0.0, 1.0e-6], grid = g, visc, model, cfl, forced)

        foreach(copyto!, u_model, inputs[1])
        timing = time()
        solve_les!(u_model; times, grid = g, visc, model, cfl, forced, onsnapshot = reduce!)
        timing = time() - timing
    end

    nred = length(e_post)
    nred < length(times) &&
        @warn "rollout stored $(nred)/$(length(times)) snapshots (blew up early)"
    transfer = (; k = collect(stuff.k), eps_sfs = .-T_accum ./ max(nred, 1))
    save_object_atomic(
        file,
        (; t = times[1:nred], e_post, ke, eps_visc, eps_sfs, spectra_les, transfer, timing),
    )
    savefields &&
        save_object_atomic(apostfieldsfile(case, dns, Δf, m), (; t = times[1:nred], u = fields))
    return
end

function solve_les!(u; times, grid, visc, model, cfl, forced = true, onsnapshot = (i, u) -> nothing)
    cache = getcache(grid)
    if !isnothing(model)
        # Allocate velocity gradient for closure
        cache = (;
            cache...,
            G = tensorfield_nonsym(grid),
            GG = spacetensorfield_nonsym(grid),
        )
    end

    # Match DNS/data generation: low-shell energy is clamped instead of adding
    # an explicit forcing term to the RHS. Disabled for decaying flows
    # (`forced = false`, e.g. the Taylor-Green vortex).
    shells = forced ? energy_shells(grid, [1, 2], u) : nothing

    t = times[1]
    j = 0
    for (i, tstop) in enumerate(times)
        # Skip first step to get initial condition
        i == 1 || while t < tstop
            Δt = cfl * propose_timestep(u, grid, visc, cache)
            Δt = min(Δt, tstop - t)
            t += Δt

            # Unforced step
            if isnothing(model)
                # Without closure
                wray3!(convectiondiffusion!, u, Δt, grid, cache; visc)
            else
                # With closure
                wray3!(les!, u, Δt, grid, cache; model, visc)
            end

            forced && maintain_shell_energy!(u, shells)

            if j % 1 == 0
                e = energy(u)
                flush(stderr)
                # Bail out on clearly unstable rollouts; the caller keeps whatever
                # `onsnapshot` reduced so far so downstream comparisons can run.
                forever = Δt < 1.0e-8
                boom = e > 1.0e5
                if forever || boom
                    forever && @warn "This will never finish"
                    boom && @warn "Boom!"
                    flush(stderr)
                    return
                end
            end
            j += 1
        end

        # Reduce the current (dealiased) state on the fly, then continue.
        foreach(u -> apply!(twothirds!, grid, (u, grid)), u)
        onsnapshot(i, u)
    end

    return
end

function getdissipation(g, u, m)
    G = getgradient(u, g)
    τ = unstack_symtensor(m(G), g)
    S = strain_from_gradient(G, g)
    return contract_dissipation(τ, S, g)
end
