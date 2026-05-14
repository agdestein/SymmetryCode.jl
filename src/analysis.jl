# Post-hoc evaluation of LES closures: SFS prediction series, KDE densities,
# a-priori errors, equivariance commutation errors, dissipation comparisons,
# and Q-R invariant joint distributions.

"Compute distribution of tensor components and dissipation coefficients."
function predict_sfs(setup, data, models)
    (; outdir, D, l, n_les, backend, Δ) = setup
    g = Grid{D}(; l, n = n_les, backend)
    u = vectorfield(g)
    τ = spacetensorfield(g)
    τhat = scalarfield(g)
    plan = plan_rfft(τ.xx)
    A = tensorfield_nonsym(g)
    AA = spacetensorfield_nonsym(g)

    fac = get_fft_fac(g)
    for (key, m) in pairs(models)
        @info "Computing SFS for $(key)"
        flush(stderr)
        τ_series = map(data.inputs) do ucpu
            GC.gc()
            CUDA.reclaim()

            # Gradient and strain-rate
            foreach(copyto!, u, ucpu)
            apply!(vectorgradient!, g, (A, u, g))
            for (AA, A) in zip(AA, A)
                apply!(twothirds!, g, (A, g))
                ldiv!(AA, plan, A) # Inverse RFFT
                AA .*= fac
            end

            # Prediction by LES model
            y = m(AA)
            τ_les = if D == 2
                xx, yy, xy = 1, 2, 3
                (; xx = view(y, :, :, xx), yy = view(y, :, :, yy), xy = view(y, :, :, xy))
            elseif D == 3
                xx, yy, zz = 1, 2, 3
                xy, yz, zx = 4, 5, 6
                (;
                    xx = view(y, :, :, :, xx),
                    yy = view(y, :, :, :, yy),
                    zz = view(y, :, :, :, zz),
                    xy = view(y, :, :, :, xy),
                    yz = view(y, :, :, :, yz),
                    zx = view(y, :, :, :, zx),
                )
            end
            τ_les |> cpu_device()
        end
        save_object("$(outdir)/sfs_$(key).jld2", τ_series)
    end
    return
end

"Compute distribution of tensor components and dissipation coefficients."
function compute_densities(setup, data, modelkeys)
    (; outdir, name, D, l, n_les, backend, Δ) = setup
    g = Grid{D}(; l, n = n_les, backend)
    u = vectorfield(g)
    τ = spacetensorfield(g)
    dissfield = spacescalarfield(g)
    τhat = scalarfield(g)
    plan = plan_rfft(τ.xx)
    A = tensorfield_nonsym(g)
    AA = spacetensorfield_nonsym(g)

    fac = get_fft_fac(g)
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
                    ldiv!(τ, plan, τhat)
                    τ .*= fac
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

            # Gradient and strain-rate
            foreach(copyto!, u, ucpu)
            apply!(vectorgradient!, g, (A, u, g))
            for (AA, A) in zip(AA, A)
                apply!(twothirds!, g, (A, g))
                ldiv!(AA, plan, A) # Inverse RFFT
                AA .*= fac
            end
            G = AA

            S = if D == 2
                (; G.xx, G.yy, xy = (G.xy .+ G.yx) ./ 2)
            elseif D == 3
                (;
                    G.xx,
                    G.yy,
                    G.zz,
                    xy = (G.xy .+ G.yx) ./ 2,
                    yz = (G.yz .+ G.zy) ./ 2,
                    zx = (G.zx .+ G.xz) ./ 2,
                )
            end

            # SFS
            foreach(copyto!, τ, τcpu)

            # Remove ghost modes and make trace-free
            for τ in τ
                plan = plan_rfft(τ)
                mul!(τhat, plan, τ)
                apply!(twothirds!, g, (τhat, g))
                ldiv!(τ, plan, τhat)
            end
            traces = @. (τ[1] + τ[2] + τ[3]) / 3
            τ[1] .-= traces
            τ[2] .-= traces
            τ[3] .-= traces

            # for S in S
            #     plan = plan_rfft(S)
            #     temp = plan * S
            #     apply!(twothirds!, g, (temp, g))
            #     ldiv!(S, plan, temp)
            # end

            # Extract components and dissipation
            τxx = τ.xx |> cpu_device()
            τxy = τ.xy |> cpu_device()
            if D == 2
                @. dissfield = τ.xx * S.xx + τ.yy * S.yy + 2 * τ.xy * S.xy
            else
                @. dissfield =
                    τ.xx * S.xx +
                    τ.yy * S.yy +
                    τ.zz * S.zz +
                    2 * (τ.xy * S.xy + τ.yz * S.yz + τ.zx * S.zx)
            end
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

function apriori_error(setup, data, modelkeys)
    (; D, l, n_les, backend) = setup
    g = Grid{D}(; l, n = n_les, backend)

    τ = spacetensorfield(g)
    τhat = scalarfield(g)
    plan = plan_rfft(τ.xx)

    fac = get_fft_fac(g)

    τ_ref = map(data.outputs) do τcpu
        for (τ, τcpu) in zip(τ, τcpu)
            copyto!(τhat, τcpu)
            apply!(twothirds!, g, (τhat, g))
            ldiv!(τ, plan, τhat)
            τ .*= fac
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
        mG_split = if D == 2
            xx, yy, xy = 1, 2, 3
            (;
                xx = selectdim(mG, D + 1, xx),
                yy = selectdim(mG, D + 1, yy),
                xy = selectdim(mG, D + 1, xy),
            )
        else
            xx, yy, zz = 1, 2, 3
            xy, yz, zx = 4, 5, 6
            (;
                xx = selectdim(mG, D + 1, xx),
                yy = selectdim(mG, D + 1, yy),
                zz = selectdim(mG, D + 1, zz),
                xy = selectdim(mG, D + 1, xy),
                yz = selectdim(mG, D + 1, yz),
                zx = selectdim(mG, D + 1, zx),
            )
        end
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
        isnothing(Δ) || apply!(gaussianfilter!, g_les, (ubar, Δ, nshell + 1, g_les))
    end
    τhat = sfs(u_dns, g_dns, g_les, Δ)
    τ = spacetensorfield(g_les)
    plan = getplan(g_les)
    for (τ, τhat) in zip(τ, τhat)
        # apply!(twothirds!, g_les, (τhat, g_les))
        ldiv!(τ, plan, τhat)
        fac = get_fft_fac(g_les)
        τ .*= fac
    end
    G = getgradient(ubar, g_les)
    if D == 2
        S = (; G.xx, G.yy, xy = (G.xy .+ G.yx) ./ 2)
    elseif D == 3
        S = (;
            G.xx,
            G.yy,
            G.zz,
            xy = (G.xy .+ G.yx) ./ 2,
            yz = (G.yz .+ G.zy) ./ 2,
            zx = (G.zx .+ G.xz) ./ 2,
        )
    end
    τ_les = map(models) do m
        y = m(G)
        if D == 2
            xx, yy, xy = 1, 2, 3
            (; xx = view(y, :, :, xx), yy = view(y, :, :, yy), xy = view(y, :, :, xy))
        elseif D == 3
            xx, yy, zz = 1, 2, 3
            xy, yz, zx = 4, 5, 6
            (;
                xx = view(y, :, :, :, xx),
                yy = view(y, :, :, :, yy),
                zz = view(y, :, :, :, zz),
                xy = view(y, :, :, :, xy),
                yz = view(y, :, :, :, yz),
                zx = view(y, :, :, :, zx),
            )
        end
    end
    τ_all = (; ref = τ, τ_les...)
    diss = map(τ_all) do τ
        d = if D == 2
            @. τ.xx * S.xx + τ.yy * S.yy + 2 * τ.xy * S.xy
        else
            @. τ.xx * S.xx +
                τ.yy * S.yy +
                τ.zz * S.zz +
                2 * (τ.xy * S.xy + τ.yz * S.yz + τ.zx * S.zx)
        end
        d
    end
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

function compute_qr(setup, data, upostfiles)
    (; D, l, n_les, backend) = setup
    g = Grid{D}(; l, n = n_les, backend)
    Ghat = scalarfield(g)
    G = spacetensorfield_nonsym(g)
    q = spacescalarfield(g)
    r = spacescalarfield(g)
    u = vectorfield(g)
    plan = plan_rfft(G.xx)

    t_kol = mean(x -> x.t_kol, data.statistics_les)

    modelkeys = [:ref, keys(upostfiles)...]

    fac = get_fft_fac(g)

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
                ldiv!(G[ij], plan, Ghat) # Inverse RFFT
                G[ij] .*= fac
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
