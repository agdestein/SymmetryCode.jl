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

get_upostfiles(setup) = map(
    name -> "$(setup.outdir)/u-post-$(name).jld2",
    (;
        nomo = "nomo",
        smag = "smag",
        dynsmag = "dynsmag",
        vers = "vers",
        clar = "clar",
        tbnn = "tbnn",
        equi = "equi",
        conv = "conv",
    ),
)

function solve_les(setup, models)
    (; D, l, n_les, backend, visc, cfl) = setup
    grid = Grid{D}(; l, n = n_les, backend)

    data = joinpath(setup.outdir, "data.jld2") |> load_object
    files = get_upostfiles(setup)

    u_les = data.inputs[1]

    # Solve LES for each model
    u_model = vectorfield(grid)
    for key in keys(models)
        @info "Solving LES with $(key)"
        flush(stderr)
        model = models[key]

        # Do a short model warmup (so that compilation does not get included in timing)
        twarm = [0.0, 1.0e-6]
        foreach(copyto!, u_model, u_les)
        solve_les!(u_model; times = twarm, grid, visc, model, cfl)

        # Solve LES
        foreach(copyto!, u_model, u_les)
        t = time()
        snapshots = solve_les!(u_model; data.times, grid, visc, model, cfl)
        t = time() - t

        # Save results
        save_object(files[key], (; data.times, u = snapshots, timing = t))
    end
    return
end

function solve_les!(u; times, grid, visc, model, cfl)
    cache = getcache(grid)
    if !isnothing(model)
        # Allocate velocity gradient for closure
        cache = (;
            cache...,
            G = tensorfield_nonsym(grid),
            GG = spacetensorfield_nonsym(grid),
        )
    end

    shells = energy_shells(grid, [1, 2], u)

    # Storage for states on CPU
    states = fill(map(Array, u), 0)

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

            maintain_shell_energy!(u, shells)

            if j % 1 == 0
                e = energy(u)
                # @info join(
                #     [
                #         "j = $j",
                #         "t = $(round(t; sigdigits = 4))",
                #         "Δt = $(round(Δt; sigdigits = 4))",
                #         "energy = $(round(e; sigdigits = 4))",
                #     ],
                #     ",\t",
                # )
                flush(stderr)
                forever = Δt < 1.0e-8
                boom = e > 1.0e5
                if forever || boom
                    forever && @warn "This will never finish"
                    boom && @warn "Boom!"
                    flush(stderr)
                    return states
                end
            end
            j += 1
        end

        # Store current state
        foreach(u -> apply!(twothirds!, grid, (u, grid)), u)
        push!(states, map(Array, u))
    end

    return states
end

function get_les_statistics(setup, keys)
    (; D, l, n_les, backend, visc) = setup

    data = joinpath(setup.outdir, "data.jld2") |> load_object
    files = get_upostfiles(setup)

    g = Grid{D}(; l, n = n_les, backend)
    dissfield_les = KernelAbstractions.zeros(backend, typeof(l), ndrange(g))
    stuff = spectral_stuff(g)
    u_ref = data.inputs
    u_les_gpu = vectorfield(g)
    u_ref_gpu = vectorfield(g)
    return map(keys) do k
        f = files[k]
        @info "Reading $(f)"
        flush(stderr)
        u_les = f |> load_object |> x -> x.u
        e_post = map(u_les, u_ref) do u_les, u_ref
            foreach(copyto!, u_les_gpu, u_les)
            foreach(copyto!, u_ref_gpu, u_ref)
            foreach(u -> apply!(twothirds!, g, (u, g)), u_les_gpu)
            foreach(u -> apply!(twothirds!, g, (u, g)), u_ref_gpu)
            foreach(copyto!, u_les, u_les_gpu)
            foreach(copyto!, u_ref, u_ref_gpu)
            u_les = stack(u_les)
            u_ref = stack(u_ref)
            norm(u_les - u_ref) / norm(u_ref)
        end
        s = map(u_les) do u_les
            foreach(copyto!, u_les_gpu, u_les)
            spec = spectrum(u_les_gpu, g, stuff)
            return spec.s
        end
        k => (; e_post, s)
    end |> NamedTuple
end

function getdissipation(g, u, m)
    G = getgradient(u, g)
    τ = unstack_symtensor(m(G), g)
    S = strain_from_gradient(G, g)
    return contract_dissipation(τ, S, g)
end
