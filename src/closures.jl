# Classical LES closure models. Each kernel writes a stress tensor τ_{ij}
# (in physical space, packed as flattened symmetric components) from a velocity
# gradient G_{ij}. The `create_*` wrappers return a closure usable by `les!`.

@kernel function clark_kernel!(ττ, GG, Δ, g::Grid{2})
    I = @index(Global, Cartesian)
    G = @SMatrix [GG.xx[I] GG.xy[I]; GG.yx[I] GG.yy[I]]
    τ = Δ^2 / 12 * G * G'
    xx, yy, xy = 1, 2, 3
    ττ[I, xx] = τ[1, 1]
    ττ[I, yy] = τ[2, 2]
    ττ[I, xy] = τ[1, 2]
end

@kernel function clark_kernel!(ττ, GG, Δ, g::Grid{3})
    I = @index(Global, Cartesian)
    G = @SMatrix [
        GG.xx[I] GG.xy[I] GG.xz[I]
        GG.yx[I] GG.yy[I] GG.yz[I]
        GG.zx[I] GG.zy[I] GG.zz[I]
    ]
    τ = Δ^2 / 12 * G * G'
    xx, yy, zz, xy, yz, zx = 1, 2, 3, 4, 5, 6
    ττ[I, xx] = τ[1, 1]
    ττ[I, yy] = τ[2, 2]
    ττ[I, zz] = τ[3, 3]
    ττ[I, xy] = τ[1, 2]
    ττ[I, yz] = τ[2, 3]
    ττ[I, zx] = τ[3, 1]
end

create_clark(Δ, g) = function clark(u, G)
    τ = stack(spacetensorfield(g))
    apply!(clark_kernel!, g, (τ, G, Δ, g); ndrange = space_ndrange(g))
    return τ
end

@kernel function smagorinsky_kernel!(ττ, GG, CS, Δ, g::Grid{2})
    I = @index(Global, Cartesian)
    G = @SMatrix [GG.xx[I] GG.xy[I]; GG.yx[I] GG.yy[I]]
    S = (G + G') / 2
    nu = CS^2 * Δ^2 * sqrt(2 * sum(abs2, S))
    τ = -2 * nu * S
    xx, yy, xy = 1, 2, 3
    ττ[I, xx] = τ[1, 1]
    ττ[I, yy] = τ[2, 2]
    ττ[I, xy] = τ[1, 2]
end

@kernel function smagorinsky_kernel!(ττ, GG, CS, Δ, g::Grid{3})
    I = @index(Global, Cartesian)
    G = @SMatrix [
        GG.xx[I] GG.xy[I] GG.xz[I]
        GG.yx[I] GG.yy[I] GG.yz[I]
        GG.zx[I] GG.zy[I] GG.zz[I]
    ]
    S = (G + G') / 2
    nu = CS^2 * Δ^2 * sqrt(2 * sum(abs2, S))
    τ = -2 * nu * S
    xx, yy, zz, xy, yz, zx = 1, 2, 3, 4, 5, 6
    ττ[I, xx] = τ[1, 1]
    ττ[I, yy] = τ[2, 2]
    ττ[I, zz] = τ[3, 3]
    ττ[I, xy] = τ[1, 2]
    ττ[I, yz] = τ[2, 3]
    ττ[I, zx] = τ[3, 1]
end

@kernel function verstappen_kernel!(ττ, GG, C, Δ, g::Grid{3})
    I = @index(Global, Cartesian)
    G = @SMatrix [
        GG.xx[I] GG.xy[I] GG.xz[I]
        GG.yx[I] GG.yy[I] GG.yz[I]
        GG.zx[I] GG.zy[I] GG.zz[I]
    ]
    S = (G + G') / 2
    q = tr(S * S) / 2
    r = tr(S * S * S) / 3
    nu = C^2 * Δ^2 * abs(r) / q
    τ = -2 * nu * S
    xx, yy, zz, xy, yz, zx = 1, 2, 3, 4, 5, 6
    ττ[I, xx] = τ[1, 1]
    ττ[I, yy] = τ[2, 2]
    ττ[I, zz] = τ[3, 3]
    ττ[I, xy] = τ[1, 2]
    ττ[I, yz] = τ[2, 3]
    ττ[I, zx] = τ[3, 1]
end

function create_smagorinsky(CS, Δ, g)
    function smagorinsky(u, G)
        τ = stack(spacetensorfield(g))
        apply!(smagorinsky_kernel!, g, (τ, G, CS, Δ, g); ndrange = space_ndrange(g))
        return τ
    end
    return smagorinsky
end

function create_verstappen(C, Δ, g)
    D = dim(g)
    @assert D == 3 "Q-R model is only defined in 3D"
    function verstappen(u, G)
        τ = stack(spacetensorfield(g))
        apply!(verstappen_kernel!, g, (τ, G, C, Δ, g); ndrange = space_ndrange(g))
        return τ
    end
    return verstappen
end

@kernel function smagorinsky_tensor!(τ, S, Δ, g::Grid{2})
    I = @index(Global, Cartesian)
    Sxx, Syy, Sxy = S.xx[I], S.yy[I], S.xy[I]
    S2 =
        Sxx^2 +
        Syy^2 +
        2 * Sxy^2
    nu = Δ^2 * sqrt(2 * S2)
    xx, yy, xy = 1, 2, 3
    τ[xx][I] = -2 * nu * Sxx
    τ[yy][I] = -2 * nu * Syy
    τ[xy][I] = -2 * nu * Sxy
end

@kernel function smagorinsky_tensor!(τ, S, Δ, g::Grid{3})
    I = @index(Global, Cartesian)
    Sxx, Syy, Szz = S.xx[I], S.yy[I], S.zz[I]
    Sxy, Syz, Szx = S.xy[I], S.yz[I], S.zx[I]
    S2 =
        Sxx^2 +
        Syy^2 +
        Szz^2 +
        2 * Sxy^2 +
        2 * Syz^2 +
        2 * Szx^2
    nu = Δ^2 * sqrt(2 * S2)
    xx, yy, zz, xy, yz, zx = 1, 2, 3, 4, 5, 6
    τ[xx][I] = -2 * nu * Sxx
    τ[yy][I] = -2 * nu * Syy
    τ[zz][I] = -2 * nu * Szz
    τ[xy][I] = -2 * nu * Sxy
    τ[yz][I] = -2 * nu * Syz
    τ[zx][I] = -2 * nu * Szx
end

@kernel function smagorinsky_coefficient!(c, M, L, g::Grid{2})
    I = @index(Global, Cartesian)
    Mxx, Myy, Mxy = M.xx[I], M.yy[I], M.xy[I]
    Lxx, Lyy, Lxy = L.xx[I], L.yy[I], L.xy[I]

    # Make L trace-free
    trace = (Lxx + Lyy) / 2
    Lxx -= trace
    Lyy -= trace

    # Dot products
    ML = Mxx * Lxx + Myy * Lyy + 2 * Mxy * Lxy
    MM = Mxx * Mxx + Myy * Myy + 2 * Mxy * Mxy

    # Least squares fit
    # c[I] = -ML / MM
    c[I] = max(-ML / MM, 0) # Clip
end

@kernel function smagorinsky_coefficient!(c, M, L, g::Grid{3})
    I = @index(Global, Cartesian)
    Mxx, Myy, Mzz, Mxy, Myz, Mzx = M.xx[I], M.yy[I], M.zz[I], M.xy[I], M.yz[I], M.zx[I]
    Lxx, Lyy, Lzz, Lxy, Lyz, Lzx = L.xx[I], L.yy[I], L.zz[I], L.xy[I], L.yz[I], L.zx[I]

    # Make L trace-free
    trace = (Lxx + Lyy + Lzz) / 3
    Lxx -= trace
    Lyy -= trace
    Lzz -= trace

    # Dot products
    ML = Mxx * Lxx + Myy * Lyy + Mzz * Lzz + 2 * Mxy * Lxy + 2 * Myz * Lyz + 2 * Mzx * Lzx
    MM = Mxx * Mxx + Myy * Myy + Mzz * Mzz + 2 * Mxy * Mxy + 2 * Myz * Myz + 2 * Mzx * Mzx

    # Least squares fit
    # c[I] = -ML / MM
    c[I] = max(-ML / MM, 0)
end

function create_dynamic_smagorinsky(Δ, g)
    # Allocate arrays (these can probably be fewer with some clever re-use)
    space = spacescalarfield(g) # Temporary spatial field
    spect = scalarfield(g) # Temporary spectral field
    Shat = tensorfield(g) # Strain-rate
    S = spacetensorfield(g) # Strain-rate
    L = spacetensorfield(g) # Non-linearity commutator
    M = spacetensorfield(g) # Smagorinsky-tensor commutator
    m1 = spacetensorfield(g) # Original Smagorinsky-tensor
    m2 = spacetensorfield(g) # Double-filter Smagorinsky tensor
    τ = similar(space, space_ndrange(g)..., tensordim(g))
    c = spacescalarfield(g) # Dynamic Smagorinsky coefficient field
    utilde = vectorfield(g) # Test-filtered ubar (effectively double-filtered)
    v = spacevectorfield(g)
    σ = tensorfield(g)
    σtilde = tensorfield(g)
    plan = plan_rfft(c)

    D = dim(g)
    Δtilde = 2 * Δ # Filter width of test filter (single Gaussian, twice the original width)
    Δdouble = sqrt(Δtilde^2 + Δ^2) # Filter width of double-Gaussian (original + test)
    fac = get_fft_fac(g)

    # Model takes spectral ubar as input
    function model(u, G)
        # Test-filter u
        for (utilde, u) in zip(utilde, u)
            copyto!(utilde, u)
            apply!(gaussianfilter!, g, (utilde, Δtilde, g))
            apply!(twothirds!, g, (utilde, g))
        end

        # Nonlinearities and L
        nonlinearity!(σ, space, v, u, plan, g)
        nonlinearity!(σtilde, space, v, utilde, plan, g)
        for σ in σ
            apply!(gaussianfilter!, g, (σ, Δtilde, g))
            apply!(twothirds!, g, (σ, g))
        end
        for (L, σ, σtilde) in zip(L, σ, σtilde)
            ldiv!(space, plan, σ)
            space ./= fac
            copyto!(L, space)
            ldiv!(space, plan, σtilde)
            space .*= fac
            L .-= space
            # Now L = tilde(σ) - σ(tilde)
        end

        # Original strainrate
        apply!(strainrate!, g, (Shat, u, g))
        for (S, Shat) in zip(S, Shat)
            apply!(twothirds!, g, (Shat, g))
            ldiv!(S, plan, Shat) # Inverse RFFT
            S .*= fac # FFT factor
        end

        # Original Smagorinsky tensor
        apply!(smagorinsky_tensor!, g, (m1, S, Δ, g))

        # Filter original tensor m1, put result in M because we need unfiltered m1 for later
        for (m1, M) in zip(m1, M)
            mul!(spect, plan, m1) # RFFT
            apply!(twothirds!, g, (spect, g))
            ldiv!(m1, plan, spect) # Inverse RFFT
            mul!(spect, plan, m1) # RFFT
            apply!(gaussianfilter!, g, (spect, Δtilde, g)) # Test filter
            apply!(twothirds!, g, (spect, g))
            ldiv!(M, plan, spect) # Inverse RFFT
        end

        # Tensor applied to test-filtered strainrate
        for (S, Shat) in zip(S, Shat)
            apply!(gaussianfilter!, g, (Shat, Δtilde, g)) # Test filter
            apply!(twothirds!, g, (Shat, g))
            ldiv!(S, plan, Shat) # Inverse RFFT
            S .*= fac # FFT factor
        end
        apply!(smagorinsky_tensor!, g, (m2, S, Δdouble, g)) # Note: width from double-filter

        # Compute commutator between smagtensor and test filter, put result in M
        for (M, m2) in zip(M, m2)
            M .-= m2
        end

        # Estimate Smagorinsky coefficients from smag-commutator M and nonlinearity-commutator L
        apply!(smagorinsky_coefficient!, g, (c, M, L, g))

        # Multiply original Smagorinsky tensor with coefficients
        for m1 in m1
            m1 .*= c
        end

        # This now contains the correctly weighted original Smagorinsky tensor
        for (i, m1) in enumerate(m1)
            copyto!(selectdim(τ, D + 1, i), m1)
        end
        return τ
    end
    return model
end
