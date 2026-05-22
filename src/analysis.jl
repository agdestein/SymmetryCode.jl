# Post-hoc evaluation of LES closures: SFS prediction series, KDE densities,
# a-priori errors, equivariance commutation errors, dissipation comparisons,
# and Q-R invariant joint distributions.

"Compute distribution of tensor components and dissipation coefficients."
function predict_sfs(setup, models)
    (; outdir, D, l, n_les, backend) = setup

    data = joinpath(setup.outdir, "data.jld2") |> load_object

    g = Grid{D}(; l, n = n_les, backend)
    u = vectorfield(g)
    plan = plan_rfft(spacescalarfield(g))
    A = tensorfield_nonsym(g)
    AA = spacetensorfield_nonsym(g)

    for (key, m) in pairs(models)
        @info "Computing SFS for $(key)"
        flush(stderr)
        τ_series = map(data.inputs) do ucpu
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
            unstack_symtensor(m(u, AA), g) |> cpu_device()
        end
        save_object("$(outdir)/sfs_$(key).jld2", τ_series)
    end
    return
end

"Compute distribution of tensor components and dissipation coefficients."
function compute_densities(setup, modelkeys)
    (; outdir, name, D, l, n_les, backend, Δ) = setup

    data = joinpath(setup.outdir, "data.jld2") |> load_object

    g = Grid{D}(; l, n = n_les, backend)
    u = vectorfield(g)
    τ = spacetensorfield(g)
    dissfield = spacescalarfield(g)
    τhat = scalarfield(g)
    plan = plan_rfft(τ.xx)
    A = tensorfield_nonsym(g)
    AA = spacetensorfield_nonsym(g)

    for mkey in [:ref, modelkeys...]
        @info "Computing SFS quantities for $(mkey)"
        flush(stderr)

        # Get SFS series
        τ_series = if mkey == :ref
            # Reference is still in spectral space
            map(data.outputs) do τcpu
                for (τ, τcpu) in zip(τ, τcpu)
                    copyto!(τhat, τcpu)
                    apply!(twothirds!, g, (τhat, g))
                    to_phys!(τ, τhat, plan, g)
                end
                τ |> cpu_device()
            end
        else
            # LES SFS are already in physical space
            load_object("$(outdir)/sfs_$(mkey).jld2")
        end

        # Get series of the three fields of interest
        fields = map(eachindex(τ_series)) do i
            @info "Snapshot $(i) of $(length(τ_series))"
            ucpu = data.inputs[i]
            τcpu = τ_series[i]

            # Velocity gradient and strain rate (physical space)
            foreach(copyto!, u, ucpu)
            apply!(vectorgradient!, g, (A, u, g))
            for (AA, A) in zip(AA, A)
                apply!(twothirds!, g, (A, g))
                to_phys!(AA, A, plan, g)
            end
            S = strain_from_gradient(AA, g)

            # SFS: copy, dealias, and make trace-free
            foreach(copyto!, τ, τcpu)
            for τ in τ
                dealias_phys!(τ, τhat, plan, g)
            end
            make_tracefree!(τ, g)

            # Extract components and dissipation
            τxx = τ.xx |> cpu_device()
            τxy = τ.xy |> cpu_device()
            dissfield .= contract_dissipation(τ, S, g)
            diss = dissfield |> cpu_device()

            (; xx = τxx, xy = τxy, diss)
        end

        # Compute kernel density estimates and save
        for fkey in [:xx, :xy, :diss]
            @info "Computing $(fkey)-density for $(mkey)"
            flush(stderr)
            samples = stack(fields) do f
                f[fkey]
            end
            estimate = samples |> vec |> kde
            save_object(
                "$(outdir)/kde_$(mkey)_$(fkey).jld2",
                (; estimate.x, estimate.density),
            )
        end
    end
    return nothing
end

"""
Compare predicted SFS stresses against the filtered DNS target snapshot-wise.

The no-model baseline is defined as zero SFS stress, giving relative error one
and zero cross-correlation by construction.
"""
function apriori_error(setup, modelkeys)
    (; D, l, n_les, backend) = setup

    data = joinpath(setup.outdir, "data.jld2") |> load_object

    g = Grid{D}(; l, n = n_les, backend)

    τ = spacetensorfield(g)
    τhat = scalarfield(g)
    plan = plan_rfft(τ.xx)

    τ_ref = map(data.outputs) do τcpu
        for (τ, τcpu) in zip(τ, τcpu)
            copyto!(τhat, τcpu)
            apply!(twothirds!, g, (τhat, g))
            to_phys!(τ, τhat, plan, g)
        end
        τ |> cpu_device()
    end

    map(modelkeys) do key
        @info "Computing a-priori errors for $(key)"
        flush(stderr)

        # For no-model, the SFS is zero
        key == :nomo && return :nomo => (; relerr = 1.0, crosscor = 0.0)

        # Otherwise, load LES SFS
        τ_les = load_object("$(setup.outdir)/sfs_$(key).jld2")
        series = map(zip(τ_les, τ_ref)) do (τ_les, τ_ref)
            a = stack(τ_ref)
            b = stack(τ_les)

            # Make trace free
            bdiag = selectdim(b, D + 1, 1:D)
            trace = sum(bdiag; dims = D + 1)
            @. bdiag -= trace / D

            # Metrics
            bb, aa = b .- mean(b), a .- mean(a)
            relerr = norm(b - a) / norm(a)
            crosscor = dot(bb, aa) / sqrt(dot(bb, bb) * dot(aa, aa))
            (; relerr, crosscor)
        end
        key => (;
            relerr = mean(s -> s.relerr, series),
            crosscor = mean(s -> s.crosscor, series),
        )
    end |> NamedTuple
end

"""
Measure whether each closure commutes with every octahedral transformation.

For each group element this compares `R(model(G))` with `model(R(G))`, using
the physical-space transformation helpers from `symmetry.jl`.
"""
function apriori_equivariance_error(; u, setup, models)
    (; D, l, n_les, backend) = setup
    (; elements, permutations, signs) = group_stuff(D)
    g = Grid{D}(; l, n = n_les, backend)
    u_ref = u.ref |> adapt(backend)
    G = getgradient(u_ref, g)
    errors = map(keys(models)) do key
        @info "Computing a-priori equi errors for $(key)"
        m = models[key]
        mG = m(G)
        mG_split = unstack_symtensor(mG, g)
        err = map(elements) do e
            ip, is = e
            p, s = permutations[ip], signs[is]
            rG = transform_tensor_nonsym(G, g, (p, s))
            mrG = m(rG)
            rmG_split = transform_tensor(mG_split, g, (p, s))
            rmG = stack(rmG_split)
            norm(rmG - mrG) / norm(mrG)
        end
        key => err
    end
    errors |> NamedTuple
end

function get_dissipation_errors(; setup, u_dns, models)
    (; D, l, n_dns, n_les, backend, Δ) = setup
    g_dns = Grid{D}(; l, n = n_dns, backend)
    g_les = Grid{D}(; l, n = n_les, backend)
    ubar = vectorfield(g_les)
    for (ubar, u_dns) in zip(ubar, u_dns)
        apply!(cutoff!, g_les, (ubar, u_dns))
        isnothing(Δ) || apply!(gaussianfilter!, g_les, (ubar, Δ, g_les))
    end
    τhat = sfs(u_dns, g_dns, g_les, Δ)
    τ = spacetensorfield(g_les)
    plan = getplan(g_les)
    for (τ, τhat) in zip(τ, τhat)
        to_phys!(τ, τhat, plan, g_les)
    end
    G = getgradient(ubar, g_les)
    S = strain_from_gradient(G, g_les)
    τ_les = map(m -> unstack_symtensor(m(G), g_les), models)
    τ_all = (; ref = τ, τ_les...)
    diss = map(τ -> contract_dissipation(τ, S, g_les), τ_all)
    return map(median, NamedTuple(diss))
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
function compute_qr(setup, modelkeys)
    (; D, l, n_les, backend) = setup

    data = joinpath(setup.outdir, "data.jld2") |> load_object

    g = Grid{D}(; l, n = n_les, backend)
    Ghat = scalarfield(g)
    G = spacetensorfield_nonsym(g)
    q = spacescalarfield(g)
    r = spacescalarfield(g)
    u = vectorfield(g)
    plan = plan_rfft(G.xx)

    t_kol = mean(x -> x.t_kol, data.statistics_les)

    upostfiles = get_upostfiles(setup)

    for k in modelkeys
        @info "Computing Q-R for $(k)"
        u_series = if k == :ref
            data.inputs
        else
            load_object(upostfiles[k]).u
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
        file = "$(setup.outdir)/qr_$(k).jld2"
        @info "Saving Q-R density to $(file)"
        save_object(file, (; dens.x, dens.y, dens.density))
    end
    return nothing
end
