# LES time integration: right-hand side with closure, solver, and statistics.

"Compute LES right-hand side with closure model (put force in `du`)."
function les!(du, u, grid, cache; model, visc)
    D = dim(grid)
    (; plan, σ, vi_vj, v, G) = cache

    # Coarse DNS stress
    stress!(σ, vi_vj, v, u, plan, visc, grid)

    # Closure model stress (in physical space)
    apply!(vectorgradient!, grid, (G, u, grid))
    fac = get_fft_fac(grid)
    GG = map(G) do G
        apply!(twothirds!, grid, (G, grid))
        res = plan \ G
        res .*= fac
        return res
    end
    y = model(u, GG)

    # Add closure stress to existing stress (in spectral space)
    for (i, σ) in enumerate(σ)
        # Use vi_vj and du.x as temp storage
        copyto!(vi_vj, selectdim(y, D + 1, i))
        mul!(du.x, plan, vi_vj)
        @. σ += du.x / fac
    end

    # Final force
    apply!(tensordivergence!, grid, (du, σ, grid))

    return
end

function solve_les(; data, setup, models, files)
    (; D, l, n_les, backend, visc, Δ, cfl) = setup
    grid = Grid{D}(; l, n = n_les, backend)

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
    backend = get_backend(u.x)
    cache = getcache(grid)
    if !isnothing(model)
        # Allocate velocity gradient for closure
        cache = (; cache..., G = tensorfield_nonsym(grid))
    end

    # Shell stuff
    s = getshells(grid, [1, 2])
    shells = map(s) do s
        inds = s.inds[1] |> adapt(backend)
        energyinds = vcat(s.inds[1], s.inds[2]) |> adapt(backend)
        eref = sum(u -> sum(abs2, view(u, energyinds)), u) / 2
        (; inds, energyinds, eref)
    end

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

            # Maintain energy
            for s in shells
                eshell = sum(u -> sum(abs2, view(u, s.energyinds)), u) / 2
                foreach(u -> (view(u, s.inds) .*= sqrt(s.eref / eshell)), u)
            end

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

function get_les_statistics(setup, data, files)
    (; D, l, n_les, backend, visc) = setup
    g = Grid{D}(; l, n = n_les, backend)
    dissfield_les = KernelAbstractions.zeros(backend, typeof(l), ndrange(g))
    stuff = spectral_stuff(g)
    u_ref = data.inputs
    u_les_gpu = vectorfield(g)
    u_ref_gpu = vectorfield(g)
    return map(files) do f
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
        (; e_post, s)
    end
end

function getdissipation(g, u, m)
    D = dim(g)
    G = getgradient(u, g)
    τ = m(G)
    S = (; G.xx, G.yy, xy = (G.xy .+ G.yx) ./ 2)
    return if D == 2
        xx, yy, xy = 1, 2, 3
        τ = (;
            xx = selectdim(τ, D + 1, xx),
            yy = selectdim(τ, D + 1, yy),
            xy = selectdim(τ, D + 1, xy),
        )
        @. τ.xx * S.xx + τ.yy * S.yy + 2 * τ.xy * S.xy
    elseif D == 3
        error()
    end
end
