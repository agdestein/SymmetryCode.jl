# Network architectures for the learned closures.
#
# All three nets (`equivariant_net`, `mlp`, `tbnn_net`) are *pointwise* MLPs
# in physical space: they predict τ at each grid point from local features,
# with no spatial mixing. They are built via Lux's `Conv` primitive with a
# 1×1 (2D) or 1×1×1 (3D) kernel — not for spatial convolution, but so all
# three share the same layer-shape convention and `equivariant_net` can use
# the Conv weight layout that its group-weight-tying requires.
#
# - `equivariant_net` is a *group* convolution over the symmetry group of
#   the cube (dihedral D₄ in 2D, octahedral O_h in 3D); each layer holds
#   |G| copies of unconstrained weights, tied by `project_*`.
# - `mlp` is the non-equivariant baseline: identical architecture without
#   weight projection.
# - `tbnn_net` predicts coefficients of the trace-free tensor basis from
#   the gradient invariants (Pope-style TBNN).

"""
Build a group convolution over the symmetry group of the cube — the
dihedral group D₄ in 2D, the octahedral group O_h in 3D (both of order
|G| = `length(elements)`).

A group convolution is realized here as a pointwise MLP whose channel
count is a multiple of |G|: each layer holds |G| copies of unconstrained
weights, and `project_*` ties them via the group projectors from
`symmetry.jl` so the layer commutes with every group element. The
returned `ps` are the unconstrained basis; Lux sees the projected weights
only after they pass through `project`.
"""
function equivariant_net(setup, nchan)
    (; D, backend, train_setup) = setup
    dev = adapt(backend)
    # dev = identity
    rng = Xoshiro(train_setup.seed)
    T = train_setup.precision
    f = T === Float32 ? f32 : f64
    nten = D^2
    (; elements) = group_stuff(D)
    (; r_lift, r_sink, r_mid) = get_weight_projectors(D)
    nreg = length(elements)
    e_lift = eigen(r_lift / nreg; sortby = -).vectors[:, 1:nten]
    e_mid = eigen(r_mid / nreg; sortby = -).vectors[:, 1:nreg]
    e_sink = eigen(r_sink / nreg; sortby = -).vectors[:, 1:nten]
    proj_lift = Dense(nten => nten * nreg; use_bias = false)
    proj_sink = Dense(nten => nreg * nten; use_bias = false)
    proj_mid = Dense(nreg => nreg^2; use_bias = false)
    proj_lift_ps, _ = Lux.setup(rng, proj_lift) |> f |> dev
    copyto!(proj_lift_ps.weight, e_lift)
    proj_sink_ps, _ = Lux.setup(rng, proj_sink) |> f |> dev
    copyto!(proj_sink_ps.weight, e_sink)
    proj_mid_ps, _ = Lux.setup(rng, proj_mid) |> f |> dev
    copyto!(proj_mid_ps.weight, e_mid)
    # Pointwise: kernel = 1 in every spatial dim. The Conv primitive is used
    # for the weight layout that the group projection needs; the spatial
    # mixing happens upstream (the input is the velocity gradient).
    kern = ntuple(Returns(1), D)
    function project_lift(ps)
        w, b = ps.weight, ps.bias
        _, c_out = size(w)
        w = proj_lift(w, proj_lift_ps, (;)) |> first
        w = reshape(w, nreg, nten, c_out)
        w = permutedims(w, (2, 1, 3))
        weight = reshape(w, kern..., nten, nreg * c_out)
        bias = reshape(repeat(reshape(b, 1, :), nreg), :)
        return (; weight, bias)
    end
    function project_mid(ps)
        w, b = ps.weight, ps.bias
        _, c_out, c_in = size(w)
        w = reshape(w, nreg, :)
        w = proj_mid(w, proj_mid_ps, (;)) |> first
        w = reshape(w, nreg, nreg, c_out, c_in)
        w = permutedims(w, (2, 4, 1, 3))
        weight = reshape(w, kern..., nreg * c_in, nreg * c_out)
        bias = reshape(repeat(reshape(b, 1, :), nreg), :)
        return (; weight, bias)
    end
    function project_sink(ps)
        w = ps.weight
        _, c_in = size(w)
        w = proj_sink(w, proj_sink_ps, (;)) |> first
        w = reshape(w, nten, nreg, c_in)
        w = permutedims(w, (2, 3, 1))
        weight = reshape(w, kern..., nreg * c_in, nten)
        return (; weight)
    end
    function project(ps)
        lift, mids..., sink, symm = ps
        return (;
            lift = project_lift(lift),
            map(project_mid, mids)...,
            sink = project_sink(sink),
            symm,
        )
    end
    net = Chain(;
        lift = Conv(kern, nten => nreg * nchan[1], gelu),
        map(
            i ->
            Symbol(:mid_, i) =>
                Conv(kern, nreg * nchan[i] => nreg * nchan[i + 1], gelu),
            1:(length(nchan) - 1),
        )...,
        sink = Conv(kern, nreg * nchan[end] => nten; use_bias = false),
        symm = WrappedFunction() do σ
            # Symmetrize *and* remove the trace, so the prediction lives in
            # the same deviatoric space as the (trace-free) DNS target.
            if D == 2
                xx = selectdim(σ, 3, 1:1)
                yy = selectdim(σ, 3, 4:4)
                xy = (selectdim(σ, 3, 2:2) + selectdim(σ, 3, 3:3)) / 2
                t = (xx + yy) / 2
                cat(xx - t, yy - t, xy; dims = 3)
            else
                xx = selectdim(σ, 4, 1:1)
                yy = selectdim(σ, 4, 5:5)
                zz = selectdim(σ, 4, 9:9)
                xy = (selectdim(σ, 4, 2:2) + selectdim(σ, 4, 4:4)) / 2
                yz = (selectdim(σ, 4, 6:6) + selectdim(σ, 4, 8:8)) / 2
                zx = (selectdim(σ, 4, 3:3) + selectdim(σ, 4, 7:7)) / 2
                t = (xx + yy + zz) / 3
                cat(xx - t, yy - t, zz - t, xy, yz, zx; dims = 4)
            end
        end,
    )
    net |> display
    ps =
        (;
        lift = (;
            weight = glorot_uniform(rng, T, nten, nchan[1]),
            bias = zeros(T, nchan[1]),
        ),
        map(
            i ->
            Symbol(:mid_, i) => (;
                weight = glorot_uniform(rng, T, nreg, nchan[i + 1], nchan[i]),
                bias = zeros(T, nchan[i + 1]),
            ),
            1:(length(nchan) - 1),
        )...,
        sink = (; weight = glorot_uniform(rng, T, nten, nchan[end])),
        symm = (;),
    ) |> dev
    st = map(Returns((;)), ps)
    return (; project, net, ps, st)
end

"""
Plain pointwise MLP — the non-equivariant baseline. Same depth and per-layer
widths as `equivariant_net`, but with no weight projection so each layer is
free to learn arbitrary mixing across channels. With `same_as_equi=true`,
the hidden widths are multiplied by |G| so the parameter count matches and
the comparison isolates the effect of equivariance from raw capacity.
"""
function mlp(setup, nchan; same_as_equi)
    (; D, backend, train_setup) = setup
    dev = adapt(backend)
    # dev = identity
    rng = Xoshiro(train_setup.seed)
    f = train_setup.precision === Float32 ? f32 : f64
    nt_nonsym = D^2
    nt = D == 2 ? 3 : 6
    (; elements) = group_stuff(D)
    nreg = if same_as_equi
        length(elements)
    else
        1
    end
    # Pointwise MLP via Lux's Conv primitive (1×1 kernel) so the layer
    # shape matches equivariant_net exactly — only the weight projection
    # is missing here.
    kern = ntuple(Returns(1), D)
    net = Chain(;
        lift = Conv(kern, nt_nonsym => nreg * nchan[1], gelu),
        map(
            i ->
            Symbol(:mid_, i) =>
                Conv(kern, nreg * nchan[i] => nreg * nchan[i + 1], gelu),
            1:(length(nchan) - 1),
        )...,
        sink = Conv(kern, nreg * nchan[end] => nt; use_bias = false),
        symm = WrappedFunction() do σ
            # Remove the trace so the prediction lives in the same
            # deviatoric space as the (trace-free) DNS target.
            if D == 2
                xx = selectdim(σ, 3, 1:1)
                yy = selectdim(σ, 3, 2:2)
                xy = selectdim(σ, 3, 3:3)
                t = (xx + yy) / 2
                cat(xx - t, yy - t, xy; dims = 3)
            else
                xx = selectdim(σ, 4, 1:1)
                yy = selectdim(σ, 4, 2:2)
                zz = selectdim(σ, 4, 3:3)
                xy = selectdim(σ, 4, 4:4)
                yz = selectdim(σ, 4, 5:5)
                zx = selectdim(σ, 4, 6:6)
                t = (xx + yy + zz) / 3
                cat(xx - t, yy - t, zz - t, xy, yz, zx; dims = 4)
            end
        end,
    )
    net |> display
    ps, st = Lux.setup(rng, net) |> f |> dev
    project = identity # No projection
    return (; project, net, ps, st)
end

"""
Wrap a trained pointwise network (`equivariant_net` or `mlp`) into a
closure with solver-facing units.

The dataloader trained the network on normalized gradients and normalized
stress. This wrapper repeats that normalization at inference and scales the
prediction back to physical `τ` before `les!` transforms it to spectral space.
"""
function fullchain(setup, net, project, ps, st, Δ)
    (; D) = setup
    ps = project(ps)

    # x is the VGT
    function model(u, x)
        x = stack(x) # Convert named tuple to array
        s = size(x)
        x = reshape(x, s..., 1) # Add singleton sample dimension
        A2 = sum(abs2, x; dims = D + 1) # VGT squared norm
        @. x /= (sqrt(A2) + eps(eltype(x))) # Normalize input gradient
        y = net(x, ps, st) |> first
        @. y *= Δ^2 * A2 # Scale output with dimensional stuff
        return reshape(y, s[1:D]..., :) # Remove singleton sample dimension
    end
    return model
end

# --- Tensor-basis neural network (TBNN) ---

# 2D basis
@inline nbasis(::Grid{2}) = 2 # Number of entries below
@inline getbasis(::Grid{2}, S, R) = (
    deviator(S),
    deviator(S * R - R * S),
    # deviator(S * S),
    # deviator(R * R),
)

# # Second-order only basis:
# # https://arc.aiaa.org/doi/10.2514/6.2022-0595
# @inline nbasis(::Grid{3}) = 5
# @inline getbasis(::Grid{3}, S, R) =
#     (deviator(S), deviator(S * S), deviator(R * R), deviator(S * R - R * S))

# New reduced basis:
# https://arc.aiaa.org/doi/10.2514/6.2022-0595
@inline nbasis(::Grid{3}) = 7
@inline getbasis(::Grid{3}, S, R) = (
    deviator(S),
    deviator(S * S),
    deviator(R * R),
    deviator(S * R - R * S),
    deviator(R * S * R),
    deviator(R * S * S - S * S * R),
    deviator(R * S * R * R - R * R * S * R),
)

# # Pope's basis:
# @inline nbasis(::Grid{3}) = 10
# @inline getbasis(::Grid{3}, S, R) = (
#     deviator(S),
#     deviator(S * R - R * S),
#     deviator(S * S),
#     deviator(R * R),
#     deviator(R * S * S - S * S * R),
#     deviator(S * R * R + R * R * S),
#     deviator(R * S * R * R - R * R * S * R),
#     deviator(S * R * S * S - S * S * R * S),
#     deviator(R * R * S * S + S * S * R * R),
#     deviator(R * S * S * R * R - R * R * S * S * R),
# )

# 2D invariants
@inline ninvariant(::Grid{2}) = 2
@inline getinvariants(::Grid{2}, S, R) = tr(S * S), tr(R * R)

# 3D invariants
@inline ninvariant(::Grid{3}) = 5
@inline getinvariants(::Grid{3}, S, R) =
    tr(S * S), tr(R * R), tr(S * S * S), tr(S * R * R), tr(S * S * R * R)

# # 3D invariants
# @inline ninvariant(::Grid{3}) = 6
# @inline getinvariants(::Grid{3}, S, R) =
#     tr(S * S), tr(R * R), tr(S * S * S), tr(S * R * R), tr(S * S * R * R), tr(S * S * R * R * S * R)

"Compute deviatoric part of a tensor."
@inline deviator(σ::SMatrix{2, 2}) = σ - tr(σ) / 2 * one(σ)
@inline deviator(σ::SMatrix{3, 3}) = σ - tr(σ) / 3 * one(σ)

@kernel function tb_kernel!(invariants, basis, a2, grads, g::Grid{2})
    nb, ni = nbasis(g), ninvariant(g)
    I = @index(Global, Cartesian)
    Gxx, Gyx, Gxy, Gyy = grads.xx[I], grads.yx[I], grads.xy[I], grads.yy[I]
    A = @SMatrix [Gxx Gxy; Gyx Gyy]
    A2 = sum(abs2, A)
    a2[I] = A2
    A = A / (sqrt(A2) + eps(eltype(A))) # Normalize gradient
    S, R = (A + A') / 2, (A - A') / 2
    i, b = getinvariants(g, S, R), getbasis(g, S, R)
    for iinv in Base.OneTo(ni)
        invariants[I, iinv] = i[iinv]
    end
    for ibas in Base.OneTo(nb)
        # Flatten symmetric 2x2 basis tensor to [xx, yy, xy]. The basis is
        # left O(1); the Δ^2 * |A|^2 factor that maps coeffs*basis to the
        # physical stress is applied in `tbnn` / the dataloader instead.
        basis[I, 1, ibas] = b[ibas][1, 1]
        basis[I, 2, ibas] = b[ibas][2, 2]
        basis[I, 3, ibas] = b[ibas][1, 2]
    end
end

@kernel function tb_kernel!(invariants, basis, a2, grads, g::Grid{3})
    ni, nb = ninvariant(g), nbasis(g)
    I = @index(Global, Cartesian)
    Axx, Axy, Axz = grads.xx[I], grads.xy[I], grads.xz[I]
    Ayx, Ayy, Ayz = grads.yx[I], grads.yy[I], grads.yz[I]
    Azx, Azy, Azz = grads.zx[I], grads.zy[I], grads.zz[I]
    A = @SMatrix [Axx Axy Axz; Ayx Ayy Ayz; Azx Azy Azz]
    A2 = sum(abs2, A)
    a2[I] = A2
    A = A / (sqrt(A2) + eps(eltype(A))) # Normalize gradient
    S, R = (A + A') / 2, (A - A') / 2
    i, b = getinvariants(g, S, R), getbasis(g, S, R)
    for iinv in Base.OneTo(ni)
        invariants[I, iinv] = i[iinv]
    end
    for ibas in Base.OneTo(nb)
        # Flatten symmetric 3x3 basis tensor to [xx, yy, zz, xy, yz, zx].
        # The basis is left O(1); the Δ^2 * |A|^2 factor that maps
        # coeffs*basis to the physical stress is applied in `tbnn` /
        # the dataloader instead.
        basis[I, 1, ibas] = b[ibas][1, 1]
        basis[I, 2, ibas] = b[ibas][2, 2]
        basis[I, 3, ibas] = b[ibas][3, 3]
        basis[I, 4, ibas] = b[ibas][1, 2]
        basis[I, 5, ibas] = b[ibas][2, 3]
        basis[I, 6, ibas] = b[ibas][3, 1]
    end
end

"""
Build normalized TBNN inputs from a physical-space velocity gradient.

Returns invariants, basis tensors, and the unnormalized `|∇u|^2`; callers use
the latter to restore the physical `Δ^2 |∇u|^2` stress scale.
"""
function build_tensorbasis(grad, g)
    T = typeof(g.l)
    nx, nb, ni, nt = space_ndrange(g), nbasis(g), ninvariant(g), tensordim(g)
    basis = KernelAbstractions.zeros(g.backend, T, nx..., nt, nb)
    invariants = KernelAbstractions.zeros(g.backend, T, nx..., ni)
    a2 = KernelAbstractions.zeros(g.backend, T, nx...)
    apply!(tb_kernel!, g, (invariants, basis, a2, grad, g); ndrange = nx)
    return invariants, basis, a2
end

function getgradient(u, g)
    D = dim(g)
    A = tensorfield_nonsym(g)
    AA = spacetensorfield_nonsym(g)
    apply!(vectorgradient!, g, (A, u, g))
    plan = plan_rfft(AA.xx)
    fac = get_fft_fac(g)
    for (AA, A) in zip(AA, A)
        apply!(twothirds!, g, (A, g))
        ldiv!(AA, plan, A) # Inverse RFFT
        AA .*= fac
    end
    return AA
end

"""
Build the TBNN coefficient network from a vector of hidden-layer widths
`nchan`. Mirrors `equivariant_net` / `mlp`: the input layer maps the
invariants to `nchan[1]`, middle layers map `nchan[i] => nchan[i+1]`, and
the (bias-free) output layer maps `nchan[end]` to the basis coefficients.
"""
function tbnn_net(setup, nchan)
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    # Pointwise MLP via 1×1 Conv, same convention as `equivariant_net` / `mlp`.
    kern = ntuple(Returns(1), setup.D)
    return Chain(
        Conv(kern, ninvariant(g) => nchan[1], gelu),
        map(
            i -> Conv(kern, nchan[i] => nchan[i + 1], gelu),
            1:(length(nchan) - 1),
        )...,
        Conv(kern, nchan[end] => nbasis(g); use_bias = false),
    )
end

tbnn(net, ps, st, Δ, g) = function model(u, A)
    nx = space_ndrange(g)
    nt = tensordim(g)
    nb = nbasis(g)

    # Invariants and (O(1)) basis tensors are built in solver precision; the
    # net weights are upcast to Float64 in create_model, so the forward pass
    # runs uniformly in Float64.
    invariants, basis, a2 = build_tensorbasis(A, g)
    invariants = reshape(invariants, size(invariants)..., 1) # One sample
    w = net(invariants, ps, st) |> first

    # Basis contraction
    b = reshape(basis, :, nt, nb)
    w = reshape(w, :, 1, nb)
    b .*= w
    m = reshape(sum(b; dims = 3), nx..., nt)

    # Map the normalized prediction back to the physical sub-filter stress
    a2 = reshape(a2, nx..., 1)
    @. m = m * Δ^2 * a2
    return m
end
