# Classical LES closure models. Each kernel writes a stress tensor τ_{ij}
# (in physical space, packed as flattened symmetric components) from a velocity
# gradient G_{ij}. The `create_*` wrappers return a closure usable by `les!`.

"Read a non-symmetric tensor field at index `I` as a local SMatrix."
@inline tensorat(GG, I, ::Grid{2}) = @SMatrix [GG.xx[I] GG.xy[I]; GG.yx[I] GG.yy[I]]
@inline tensorat(GG, I, ::Grid{3}) = @SMatrix [
    GG.xx[I] GG.xy[I] GG.xz[I]
    GG.yx[I] GG.yy[I] GG.yz[I]
    GG.zx[I] GG.zy[I] GG.zz[I]
]

"Write the upper triangle of a symmetric SMatrix into the packed array `ττ` at `I`."
@inline function store_symtensor!(ττ, I, τ, ::Grid{2})
    ττ[I, 1] = τ[1, 1]
    ττ[I, 2] = τ[2, 2]
    ττ[I, 3] = τ[1, 2]
    return nothing
end
@inline function store_symtensor!(ττ, I, τ, ::Grid{3})
    ττ[I, 1] = τ[1, 1]
    ττ[I, 2] = τ[2, 2]
    ττ[I, 3] = τ[3, 3]
    ττ[I, 4] = τ[1, 2]
    ττ[I, 5] = τ[2, 3]
    ττ[I, 6] = τ[3, 1]
    return nothing
end

"Allocate a packed symmetric-tensor field and dispatch `kernel!` over physical space."
function run_closure_kernel(kernel!, args, g)
    τ = stack(spacetensorfield(g))
    apply!(kernel!, g, (τ, args..., g); ndrange = space_ndrange(g))
    return τ
end

@kernel function clark_kernel!(ττ, GG, Δ, g)
    I = @index(Global, Cartesian)
    G = tensorat(GG, I, g)
    τ = Δ^2 / 12 * G * G'
    store_symtensor!(ττ, I, τ, g)
end

@kernel function smagorinsky_kernel!(ττ, GG, CS, Δ, g)
    I = @index(Global, Cartesian)
    G = tensorat(GG, I, g)
    S = (G + G') / 2
    nu = CS^2 * Δ^2 * sqrt(2 * sum(abs2, S))
    τ = -2 * nu * S
    store_symtensor!(ττ, I, τ, g)
end

@kernel function verstappen_kernel!(ττ, GG, C, Δ, g::Grid{3})
    I = @index(Global, Cartesian)
    G = tensorat(GG, I, g)
    S = (G + G') / 2
    q = tr(S * S) / 2
    r = tr(S * S * S) / 3
    nu = C^2 * Δ^2 * abs(r) / q
    τ = -2 * nu * S
    store_symtensor!(ττ, I, τ, g)
end

create_clark(Δ, g) = (u, G) -> run_closure_kernel(clark_kernel!, (G, Δ), g)
create_smagorinsky(CS, Δ, g) =
    (u, G) -> run_closure_kernel(smagorinsky_kernel!, (G, CS, Δ), g)

function create_verstappen(C, Δ, g)
    @assert dim(g) == 3 "Q-R model is only defined in 3D"
    return (u, G) -> run_closure_kernel(verstappen_kernel!, (G, C, Δ), g)
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

@kernel function smagorinsky_ml_mm!(ml, mm, M, L, g::Grid{2})
    I = @index(Global, Cartesian)
    Mxx, Myy, Mxy = M.xx[I], M.yy[I], M.xy[I]
    Lxx, Lyy, Lxy = L.xx[I], L.yy[I], L.xy[I]

    # Make L trace-free
    trace = (Lxx + Lyy) / 2
    Lxx -= trace
    Lyy -= trace

    ml[I] = Mxx * Lxx + Myy * Lyy + 2 * Mxy * Lxy
    mm[I] = Mxx * Mxx + Myy * Myy + 2 * Mxy * Mxy
end

@kernel function smagorinsky_ml_mm!(ml, mm, M, L, g::Grid{3})
    I = @index(Global, Cartesian)
    Mxx, Myy, Mzz, Mxy, Myz, Mzx = M.xx[I], M.yy[I], M.zz[I], M.xy[I], M.yz[I], M.zx[I]
    Lxx, Lyy, Lzz, Lxy, Lyz, Lzx = L.xx[I], L.yy[I], L.zz[I], L.xy[I], L.yz[I], L.zx[I]

    # Make L trace-free
    trace = (Lxx + Lyy + Lzz) / 3
    Lxx -= trace
    Lyy -= trace
    Lzz -= trace

    ml[I] = Mxx * Lxx + Myy * Lyy + Mzz * Lzz + 2 * Mxy * Lxy + 2 * Myz * Lyz + 2 * Mzx * Lzx
    mm[I] = Mxx * Mxx + Myy * Myy + Mzz * Mzz + 2 * Mxy * Mxy + 2 * Myz * Myz + 2 * Mzx * Mzx
end

"""
Create the dynamic Smagorinsky closure using a box-averaged Lilly coefficient.

The model allocates its scratch buffers once and returns a closure of `(u, G)`;
`G` is accepted for the shared closure interface but the dynamic model computes
the strain fields it needs from `u` and its test-filtered copy.
"""
function create_dynamic_smagorinsky(Δ, g)
    space = spacescalarfield(g)
    spect = scalarfield(g)
    Shat = tensorfield(g)
    S = spacetensorfield(g)       # Strain rate at grid-filter level
    Stilde = spacetensorfield(g)  # Strain rate at test-filter level
    L = spacetensorfield(g)       # Germano resolved-stress commutator
    M = spacetensorfield(g)       # Smagorinsky-tensor commutator
    m1 = spacetensorfield(g)      # Smagorinsky tensor at grid level
    m2 = spacetensorfield(g)      # Smagorinsky tensor at combined-filter level
    τ = similar(space, space_ndrange(g)..., tensordim(g))
    ml = spacescalarfield(g)      # Pointwise M:L for Lilly volume averaging
    mm = spacescalarfield(g)      # Pointwise M:M for Lilly volume averaging
    utilde = vectorfield(g)       # Test-filtered velocity (spectral)
    v = spacevectorfield(g)
    σ = tensorfield(g)
    σtilde = tensorfield(g)
    plan = plan_rfft(ml)

    D = dim(g)
    Δtilde = 2 * Δ                   # Test filter width
    Δdouble = sqrt(Δtilde^2 + Δ^2)   # Combined grid + test filter width

    function model(u, G)
        # Test-filtered velocity (spectral)
        for (utilde, u) in zip(utilde, u)
            copyto!(utilde, u)
            apply!(gaussianfilter!, g, (utilde, Δtilde, g))
            apply!(twothirds!, g, (utilde, g))
        end

        # Germano L = tilde(uu) - tilde(u) tilde(u), in physical space
        nonlinearity!(σ, space, v, u, plan, g)
        nonlinearity!(σtilde, space, v, utilde, plan, g)
        for σ in σ
            apply!(gaussianfilter!, g, (σ, Δtilde, g))
            apply!(twothirds!, g, (σ, g))
        end
        for (L, σ, σtilde) in zip(L, σ, σtilde)
            to_phys!(L, σ, plan, g)
            to_phys!(space, σtilde, plan, g)
            L .-= space
        end

        # Physical strain rate at grid (S) and combined-filter (Stilde) levels
        apply!(strainrate!, g, (Shat, u, g))
        for (S, Stilde, Shat) in zip(S, Stilde, Shat)
            apply!(twothirds!, g, (Shat, g))
            to_phys!(S, Shat, plan, g)
            apply!(gaussianfilter!, g, (Shat, Δtilde, g))
            apply!(twothirds!, g, (Shat, g))
            to_phys!(Stilde, Shat, plan, g)
        end

        # Smagorinsky tensors at the two filter levels
        apply!(smagorinsky_tensor!, g, (m1, S, Δ, g); ndrange = space_ndrange(g))
        apply!(smagorinsky_tensor!, g, (m2, Stilde, Δdouble, g); ndrange = space_ndrange(g))

        # M = tilde(m1) - m2; m1 is dealiased in place so it can be reused at the end
        for (m1, M) in zip(m1, M)
            dealias_phys!(m1, spect, plan, g)
            copyto!(M, m1)
            test_filter_phys!(M, spect, plan, Δtilde, g)
        end
        for (M, m2) in zip(M, m2)
            M .-= m2
        end

        # Dynamic coefficient via global Lilly average: c = -<ML>/<MM>, clipped.
        # Valid because the laptop/turbulator/snellius setups are homogeneous on
        # the periodic box, so spatial mean is the right statistical operation.
        apply!(smagorinsky_ml_mm!, g, (ml, mm, M, L, g); ndrange = space_ndrange(g))
        sum_ml = sum(ml)
        sum_mm = sum(mm)
        c = ifelse(iszero(sum_mm), zero(sum_mm), max(-sum_ml / sum_mm, zero(sum_mm)))
        for m1 in m1
            m1 .*= c
        end
        for (i, m1) in enumerate(m1)
            copyto!(selectdim(τ, D + 1, i), m1)
        end
        return τ
    end
    return model
end
