# LES filters operating on RFFT-shaped spectral arrays.

@inline function cutoff_index(nbar, n, i, is1)
    imax = div(nbar, 2) + is1
    isneg = i > imax
    return ifelse(isneg, n - nbar + i, i) # Negative wavenumbers count backwards
end
@inline cutoff_index(nbar, n, I::CartesianIndex{2}) = CartesianIndex(
    (
        cutoff_index(nbar, n, I.I[1], true),
        cutoff_index(nbar, n, I.I[2], false),
    ),
)
@inline cutoff_index(nbar, n, I::CartesianIndex{3}) = CartesianIndex(
    (
        cutoff_index(nbar, n, I.I[1], true),
        cutoff_index(nbar, n, I.I[2], false),
        cutoff_index(nbar, n, I.I[3], false),
    ),
)

@kernel function cutoff!(ubar, u)
    nbar = size(ubar, 2)
    n = size(u, 2)
    I = @index(Global, Cartesian)
    J = cutoff_index(nbar, n, I)
    ubar[I] = u[J]
end

@kernel function inverse_cutoff!(u, ubar)
    nbar = size(ubar, 2)
    n = size(u, 2)
    I = @index(Global, Cartesian)
    J = cutoff_index(nbar, n, I)
    u[J] = ubar[I]
end

@kernel function gaussianfilter!(u, Δ, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = wavenumber_full(g, I)
    k2 = kx^2 + ky^2
    w = exp(-Δ^2 * k2 / 24)
    nshell = 2
    w = ifelse(k2 < (nshell + 1)^2, one(w), w) # Don't filter forced shells
    u[I] *= w
end
@kernel function gaussianfilter!(u, Δ, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumber_full(g, I)
    k2 = kx^2 + ky^2 + kz^2
    w = exp(-Δ^2 * k2 / 24)
    nshell = 2
    kbound2 = (2π / g.l * (nshell + 1))^2
    w = ifelse(k2 < kbound2, one(w), w) # Don't filter forced shells
    u[I] *= w
end
