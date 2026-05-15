# "Pseudo-spectral solver for the 3D incompressible Navier-Stokes equations."
# module Solver

# using AbstractFFTs
# using Adapt
# using FFTW
# using LinearAlgebra
# using KernelAbstractions
# using Random

"Cartesian grid of a squared/cubic periodic domain."
struct Grid{D, T, B}
    "Domain side length."
    l::T

    "Number of grid points in each dimension (should be even)."
    n::Int

    """
    KernelAbstractions.jl hardware backend.
    For Nvidia GPUs, do `using CUDA` and set to `CUDABackend()`.
    """
    backend::B

    "Kernel work group size."
    workgroupsize::Int

    function Grid{D}(; l, n, backend = CPU(), workgroupsize = 64) where {D}
        @assert n % 2 == 0 "Only even number of grid points supported."
        return new{D, typeof(l), typeof(backend)}(l, n, backend, workgroupsize)
    end
end

"Physical dimension."
@inline dim(::Grid{D}) where {D} = D

"Number of components in symmetric tensor."
@inline tensordim(::Grid{2}) = 3
@inline tensordim(::Grid{3}) = 6

"Grid spacing."
@inline spacing(g::Grid) = g.l / g.n

"Volume of a grid cell."
@inline volume(g::Grid{D}) where {D} = (g.l / g.n)^D

# Like fftfreq, but with proper type
@inline fftfreq_int(g::Grid, i::Int) = i - 1 - ifelse(i <= (g.n + 1) >> 1, 0, g.n)

"Integer wavenumber vector."
@inline wavenumber_int(g::Grid, I::CartesianIndex{2}) = I.I[1] - 1, fftfreq_int(g, I.I[2])
@inline wavenumber_int(g::Grid, I::CartesianIndex{3}) =
    I.I[1] - 1, fftfreq_int(g, I.I[2]), fftfreq_int(g, I.I[3])

"""
Get wavenumber with dimensional part as ``2 \\pi k / l``,
where ``k \\ \\in \\mathbb{Z}^3`` is an integer-wavenumber-vector.
"""
@inline wavenumber_full(g::Grid, I::CartesianIndex{2}) =
    pi / g.l * 2 * (I.I[1] - 1), pi / g.l * 2 * fftfreq_int(g, I.I[2])
@inline wavenumber_full(g::Grid, I::CartesianIndex{3}) = (
    pi / g.l * 2 * (I.I[1] - 1),
    pi / g.l * 2 * fftfreq_int(g, I.I[2]),
    pi / g.l * 2 * fftfreq_int(g, I.I[3]),
)

@inline function squared_wavenumber_int(g::Grid{2}, I)
    kx, ky = wavenumber_int(g, I)
    return kx^2 + ky^2
end
@inline function squared_wavenumber_int(g::Grid{3}, I)
    kx, ky, kz = wavenumber_int(g, I)
    return kx^2 + ky^2 + kz^2
end

@inline squared_wavenumber_full(g, I) = (pi / g.l * 2)^2 * squared_wavenumber_int(g, I)

"Range for RFFT sized spectral arrays."
ndrange((; n)::Grid{2}) = div(n, 2) + 1, n
ndrange((; n)::Grid{3}) = div(n, 2) + 1, n, n

"Range for physical space arrays."
space_ndrange(g::Grid) = ntuple(Returns(g.n), dim(g))

get_fft_fac(g::Grid) = (g.n / g.l)^dim(g)

"Inverse RFFT with the solver's `(n/l)^D` scaling. Returns `phys`."
to_phys!(phys, spec, plan, g) =
    (ldiv!(phys, plan, spec); phys .*= get_fft_fac(g); phys)

"Forward RFFT with the solver's `(n/l)^D` scaling. Returns `spec`."
to_spec!(spec, phys, plan, g) =
    (mul!(spec, plan, phys); spec ./= get_fft_fac(g); spec)

"Round-trip a physical scalar through 2/3-truncation. `spec` is scratch."
dealias_phys!(phys, spec, plan, g) = (
    mul!(spec, plan, phys);
    apply!(twothirds!, g, (spec, g));
    ldiv!(phys, plan, spec);
    phys
)

"Apply Gaussian test filter (with 2/3 truncation) to a physical scalar in place."
test_filter_phys!(phys, spec, plan, Δ, g) = (
    mul!(spec, plan, phys);
    apply!(gaussianfilter!, g, (spec, Δ, g));
    apply!(twothirds!, g, (spec, g));
    ldiv!(phys, plan, spec);
    phys
)

"""
Apply KernelAbstractions kernel over given ndrange
(defaults to RFFT spectral array range).
"""
function apply!(kernel!, grid, args; ndrange = ndrange(grid))
    (; backend, workgroupsize) = grid
    kernel!(backend, workgroupsize)(args...; ndrange)
    KernelAbstractions.synchronize(backend)
    return nothing
end

"Scalar-valued spectral field (RFFT-sized)."
scalarfield(g::Grid{D, T}) where {D, T} =
    KernelAbstractions.zeros(g.backend, Complex{T}, ndrange(g))

"Vector-valued spectral field."
vectorfield(g::Grid{2}) = (; x = scalarfield(g), y = scalarfield(g))
vectorfield(g::Grid{3}) = (; x = scalarfield(g), y = scalarfield(g), z = scalarfield(g))

"Symmetric-tensor-valued spectral field."
tensorfield(g::Grid{2}) = (; xx = scalarfield(g), yy = scalarfield(g), xy = scalarfield(g))
tensorfield(g::Grid{3}) = (;
    xx = scalarfield(g),
    yy = scalarfield(g),
    zz = scalarfield(g),
    xy = scalarfield(g),
    yz = scalarfield(g),
    zx = scalarfield(g),
)

"Non-symmetric-tensor-valued spectral field."
tensorfield_nonsym(g::Grid{2}) =
    (; xx = scalarfield(g), yx = scalarfield(g), xy = scalarfield(g), yy = scalarfield(g))
tensorfield_nonsym(g::Grid{3}) = (;
    xx = scalarfield(g),
    yx = scalarfield(g),
    zx = scalarfield(g),
    xy = scalarfield(g),
    yy = scalarfield(g),
    zy = scalarfield(g),
    xz = scalarfield(g),
    yz = scalarfield(g),
    zz = scalarfield(g),
)

"Scalar-valued physical-space field."
spacescalarfield(g::Grid{D, T}) where {D, T} =
    KernelAbstractions.zeros(g.backend, T, ntuple(Returns(g.n), D))

"Vector-valued physical-space field."
spacevectorfield(g::Grid{2}) = (; x = spacescalarfield(g), y = spacescalarfield(g))
spacevectorfield(g::Grid{3}) =
    (; x = spacescalarfield(g), y = spacescalarfield(g), z = spacescalarfield(g))

"Symmetric-tensor-valued physical-space field."
spacetensorfield(g::Grid{2}) =
    (; xx = spacescalarfield(g), yy = spacescalarfield(g), xy = spacescalarfield(g))
spacetensorfield(g::Grid{3}) = (;
    xx = spacescalarfield(g),
    yy = spacescalarfield(g),
    zz = spacescalarfield(g),
    xy = spacescalarfield(g),
    yz = spacescalarfield(g),
    zx = spacescalarfield(g),
)

"Non-symmetric-tensor-valued physical-space field."
spacetensorfield_nonsym(g::Grid{2}) = (;
    xx = spacescalarfield(g),
    yx = spacescalarfield(g),
    xy = spacescalarfield(g),
    yy = spacescalarfield(g),
)
spacetensorfield_nonsym(g::Grid{3}) = (;
    xx = spacescalarfield(g),
    yx = spacescalarfield(g),
    zx = spacescalarfield(g),
    xy = spacescalarfield(g),
    yy = spacescalarfield(g),
    zy = spacescalarfield(g),
    xz = spacescalarfield(g),
    yz = spacescalarfield(g),
    zz = spacescalarfield(g),
)

"Subtract the trace of a symmetric tensor field in-place."
function make_tracefree!(τ, g::Grid{2})
    trace = @. (τ.xx + τ.yy) / 2
    @. τ.xx -= trace
    @. τ.yy -= trace
    return τ
end
function make_tracefree!(τ, g::Grid{3})
    trace = @. (τ.xx + τ.yy + τ.zz) / 3
    @. τ.xx -= trace
    @. τ.yy -= trace
    @. τ.zz -= trace
    return τ
end

"""
Unpack a stacked symmetric-tensor array (shape `(n,..,n,tensordim)`)
into a NamedTuple of views indexed by component.
"""
unstack_symtensor(y, ::Grid{2}) =
    (; xx = selectdim(y, 3, 1), yy = selectdim(y, 3, 2), xy = selectdim(y, 3, 3))
unstack_symtensor(y, ::Grid{3}) = (;
    xx = selectdim(y, 4, 1),
    yy = selectdim(y, 4, 2),
    zz = selectdim(y, 4, 3),
    xy = selectdim(y, 4, 4),
    yz = selectdim(y, 4, 5),
    zx = selectdim(y, 4, 6),
)

"Make vector field solenoidal."
@kernel function project!(u, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = wavenumber_full(g, I)
    ux, uy = u.x[I], u.y[I]
    p = (kx * ux + ky * uy) / (kx * kx + ky * ky)
    p = ifelse(I.I == (1, 1), zero(p), p) # Leave constant mode intact
    u.x[I] = ux - kx * p
    u.y[I] = uy - ky * p
end
@kernel function project!(u, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumber_full(g, I)
    ux, uy, uz = u.x[I], u.y[I], u.z[I]
    p = (kx * ux + ky * uy + kz * uz) / (kx * kx + ky * ky + kz * kz)
    p = ifelse(I.I == (1, 1, 1), zero(p), p) # Leave constant mode intact
    u.x[I] = ux - kx * p
    u.y[I] = uy - ky * p
    u.z[I] = uz - kz * p
end

"Set ghost components to zero."
@kernel function twothirds!(u, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = wavenumber_int(g, I)
    kcut = div(g.n, 3)
    nonzero = (kx, ky) != (0, 0)
    ix = abs(kx) ≤ kcut
    iy = abs(ky) ≤ kcut
    u[I] = ifelse(nonzero & ix & iy, u[I], zero(eltype(u)))
end
@kernel function twothirds!(u, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumber_int(g, I)
    kcut = div(g.n, 3)
    nonzero = (kx, ky, kz) != (0, 0, 0)
    ix = abs(kx) ≤ kcut
    iy = abs(ky) ≤ kcut
    iz = abs(kz) ≤ kcut
    u[I] = ifelse(nonzero & ix & iy & iz, u[I], zero(eltype(u)))
end

"""
Enforce Hermitian symmetry `S(k) = conj(S(-k))` on the `kx=0` and `kx=kmax`
planes of an RFFT array. The other `kx` planes are already independent and
left alone. Required after filling the spectral array with `randn!`, which
breaks the symmetry and causes the IRFFT to silently discard part of the
energy on those planes.
"""
@kernel function symmetrize!(u, g::Grid{2})
    I = @index(Global, Cartesian)
    n = g.n
    kmax = div(n, 2)
    ix, iy = I.I
    on_sym = (ix == 1) | (ix == kmax + 1)
    iy_self = (iy == 1) | (iy == kmax + 1)
    if on_sym
        if iy_self
            u[I] = complex(real(u[I]))
        elseif iy > kmax + 1
            u[I] = conj(u[ix, n + 2 - iy])
        end
    end
end
@kernel function symmetrize!(u, g::Grid{3})
    I = @index(Global, Cartesian)
    n = g.n
    kmax = div(n, 2)
    ix, iy, iz = I.I
    on_sym = (ix == 1) | (ix == kmax + 1)
    iy_self = (iy == 1) | (iy == kmax + 1)
    iz_self = (iz == 1) | (iz == kmax + 1)
    piy = ifelse(iy_self, iy, n + 2 - iy)
    piz = ifelse(iz_self, iz, n + 2 - iz)
    if on_sym
        if iy_self & iz_self
            u[I] = complex(real(u[I]))
        elseif (iy > piy) | ((iy == piy) & (iz > piz))
            u[I] = conj(u[ix, piy, piz])
        end
    end
end

getplan(grid) = plan_rfft(spacescalarfield(grid))

function nonlinearity!(σ, vi_vj, v, u, plan, g::Grid)
    D = dim(g)
    temp = σ.xx # Use σ.xx as temporary complex storage
    fac = get_fft_fac(g)
    for i in 1:D
        copyto!(temp, u[i])
        apply!(twothirds!, g, (temp, g)) # Zero out high wavenumbers
        ldiv!(v[i], plan, temp) # Inverse transform
        v[i] .*= fac # FFT factor
    end
    symbols = if D == 2
        [(:x, :x), (:y, :y), (:x, :y)]
    elseif D == 3
        [(:x, :x), (:y, :y), (:z, :z), (:x, :y), (:y, :z), (:z, :x)]
    end
    for (i, j) in symbols
        ij = Symbol(i, j)
        @. vi_vj = v[i] * v[j]
        mul!(σ[ij], plan, vi_vj)
        σ[ij] ./= fac
    end
    return nothing
end

@kernel function viscosity!(σ, u, visc, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = wavenumber_full(g, I)
    ux, uy = u.x[I], u.y[I]
    σ.xx[I] -= 2 * visc * im * kx * ux
    σ.yy[I] -= 2 * visc * im * ky * uy
    σ.xy[I] -= visc * im * (ky * ux + kx * uy)
end
@kernel function viscosity!(σ, u, visc, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumber_full(g, I)
    ux, uy, uz = u.x[I], u.y[I], u.z[I]
    σ.xx[I] -= 2 * visc * im * kx * ux
    σ.yy[I] -= 2 * visc * im * ky * uy
    σ.zz[I] -= 2 * visc * im * kz * uz
    σ.xy[I] -= visc * im * (ky * ux + kx * uy)
    σ.yz[I] -= visc * im * (kz * uy + ky * uz)
    σ.zx[I] -= visc * im * (kx * uz + kz * ux)
end

function stress!(σ, vi_vj, v, u, plan, visc, g::Grid)
    # foreach(s -> fill!(s, 0), σ)
    nonlinearity!(σ, vi_vj, v, u, plan, g)
    return apply!(viscosity!, g, (σ, u, visc, g))
end

@kernel function vectordivergence!(div, u, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = wavenumber_full(g, I)
    div[I] = im * kx * u.x[I] + im * ky * u.y[I]
end
@kernel function vectordivergence!(div, u, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumber_full(g, I)
    div[I] = im * kx * u.x[I] + im * ky * u.y[I] + im * kz * u.z[I]
end

@kernel function tensordivergence!(div, σ, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = wavenumber_full(g, I)
    div.x[I] = -im * kx * σ.xx[I] - im * ky * σ.xy[I]
    div.y[I] = -im * kx * σ.xy[I] - im * ky * σ.yy[I]
end
@kernel function tensordivergence!(div, σ, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumber_full(g, I)
    div.x[I] = -im * kx * σ.xx[I] - im * ky * σ.xy[I] - im * kz * σ.zx[I]
    div.y[I] = -im * kx * σ.xy[I] - im * ky * σ.yy[I] - im * kz * σ.yz[I]
    div.z[I] = -im * kx * σ.zx[I] - im * ky * σ.yz[I] - im * kz * σ.zz[I]
end

peak_profile(k; kpeak) = k^4 * exp(-2 * (k / kpeak)^2)
linear_profile_2D(k) = (k > 0) * k^(-3)
linear_profile_3D(k) = (k > 0) * k^(-5 / 3)

"Taylor-Green vortex."
function taylorgreen(g::Grid{2}, plan; doproject = true)
    (; l, n, backend) = g
    h = l / n
    x = range(h / 2, l - h / 2, n) |> Array |> adapt(backend)
    y = reshape(x, 1, :)
    v = spacescalarfield(g)
    fac = get_fft_fac(g)
    #! format: off
    @. v =  -sinpi(2x / l) * cospi(2y / l); ux = plan * v; ux ./= fac
    @. v = cospi(2x / l) * sinpi(2y / l); uy = plan * v; uy ./= fac
    #! format: on
    v = nothing
    u = (; x = ux, y = uy)
    doproject && apply!(project!, g, (u, g))
    return u
end
function taylorgreen(g::Grid{3}, plan; doproject = true)
    (; l, n, backend) = g
    h = l / n
    x = range(h / 2, l - h / 2, n) |> Array |> adapt(backend)
    y = reshape(x, 1, :)
    z = reshape(x, 1, 1, :)
    v = spacescalarfield(g)
    fac = get_fft_fac(g)
    #! format: off
    @. v =  sinpi(2x / l) * cospi(2y / l) * sinpi(2z / l) / 2; ux = plan * v; ux ./= fac
    @. v = -cospi(2x / l) * sinpi(2y / l) * sinpi(2z / l) / 2; uy = plan * v; uy ./= fac
    #! format: on
    v = nothing
    uz = zero(ux)
    u = (; x = ux, y = uy, z = uz)
    doproject && apply!(project!, g, (u, g))
    return u
end

"""
Make random velocity field with prescribed energy spectrum profile.
Additional kwargs are passed to `profile(k; kwargs...)`.
"""
function randomfield(profile, grid; totalenergy = 1, rng = Random.default_rng(), kwargs...)
    # Mask for active wavenumbers: kleft ≤ k < kleft + 1
    # Do everything squared to avoid floats
    @kernel function mask!(mask, kleft, g)
        I = @index(Global, Cartesian)
        k2 = squared_wavenumber_int(g, I)
        mask[I] = kleft^2 ≤ k2 < (kleft + 1)^2
    end

    # Create random field and make it divergence free.
    # `randn!` on the complex spectral array does not produce a Hermitian-
    # symmetric field on the kx=0 / kx=kmax planes, so fix that before any
    # transform back to physical space silently averages it away.
    u = vectorfield(grid)
    foreach(u -> randn!(rng, u), u)
    foreach(u -> apply!(symmetrize!, grid, (u, grid)), u)
    apply!(project!, grid, (u, grid))

    # RFFT exploits conjugate symmetry, so we only need half the modes
    kmax = div(grid.n, 2)
    T = typeof(grid.l)

    # Allocate arrays
    E = similar(u.x, T, ndrange(grid)...)
    Emask = similar(E)
    mask = similar(E, Bool)

    # Set ghost cells to zero
    foreach(u -> apply!(twothirds!, grid, (u, grid)), u)

    # Compute energy
    if dim(grid) == 2
        @. E = (abs2(u.x) + abs2(u.y)) / 2
    else
        @. E = (abs2(u.x) + abs2(u.y) + abs2(u.z)) / 2
    end

    # Maximum partially resolved wavenumber (sqrt(dim) * kmax)
    kdiag = floor(Int, sqrt(dim(grid)) * kmax)
    # k23 = round(Int, 2 / 3 * kmax)

    # Sum of shell weights
    totalprofile = sum(k -> profile(k; kwargs...), 0:kdiag)

    # Adjust energy in each partially resolved shell [k, k+1)
    for k in 0:kdiag
        apply!(mask!, grid, (mask, k, grid)) # Shell mask
        @. Emask = mask * E
        Eshell = sum(Emask) + sum(selectdim(Emask, 1, 2:kmax)) # Current energy in shell
        E0 = totalenergy * profile(k; kwargs...) / totalprofile # Desired energy in shell
        factor = Eshell > 0 ? sqrt(E0 / Eshell) : zero(E0) # E = u^2 / 2
        for u in u
            @. u = ifelse(mask, factor * u, u)
        end
    end

    # Set ghost cells to zero for no surprises
    foreach(u -> apply!(twothirds!, grid, (u, grid)), u)

    # The velocity now has
    # the correct spectrum,
    # random phase shifts,
    # random orientations,
    # and is also divergence free.
    return u
end

function energy(u)
    kmax = size(u.x, 1) - 1
    return sum(u -> sum(abs2, u) + sum(abs2, selectdim(u, 1, 2:kmax)), u) / 2
end

@kernel function z_vort_kernel!(vort, u, grid)
    I = @index(Global, Cartesian)
    k = wavenumber_full(grid, I)
    kx, ky = k[1], k[2]
    ux, uy = u.x[I], u.y[I]
    vort[I] = -im * ky * ux + im * kx * uy
end

function z_vort!(spacevort, vort, u, plan, grid)
    apply!(z_vort_kernel!, grid, (vort, u, grid))
    apply!(twothirds!, grid, (vort, grid)) # Zero out high wavenumbers
    ldiv!(spacevort, plan, vort)
    spacevort .*= get_fft_fac(grid) # FFT factor
    return nothing
end

function spectral_stuff(grid; npoint = nothing)
    (; l, backend) = grid
    T = typeof(l)

    n = grid.n
    kmax = div(n, 2)

    k2 = map(I -> squared_wavenumber_int(grid, I), CartesianIndices(ndrange(grid)))
    k2 = reshape(k2, :)

    # Output query points (evenly log-spaced, but only integer wavenumbers)
    kcut = div(2 * kmax, 3)
    # kcut = kmax
    if isnothing(npoint)
        kuse = 1:kcut
    else
        kuse = logrange(T(1), T(kcut), npoint)
        kuse = sort(unique(round.(Int, kuse)))
    end

    shells = getshells(grid, kuse)
    inds = map(s -> vcat(s.inds...), shells) # Include conjugate indices

    # Put indices on GPU
    inds = map(adapt(backend), inds)

    return (; shells = inds, k = 2π / l * kuse)
end

function spectrum(u, grid, stuff = spectral_stuff(grid))
    (; shells, k) = stuff
    s = map(shells) do shell
        sum(u -> sum(abs2, view(u, shell)) / 2, u)
    end
    return (; k, s)
end

"Sum of squared modulus that also accounts for missing modes in RFFT."
getenergy(u) = sum(abs2, u) + sum(abs2, selectdim(u, 1, 2:(size(u, 1) - 1)))

"Sum of that also accounts for missing modes in RFFT."
spectralsum(f, u) = sum(f, u) + sum(f, selectdim(u, 1, 2:(size(u, 1) - 1)))

"Dot product of that also accounts for missing modes in RFFT."
spectraldot(u, v) =
    dot(u, v) + dot(selectdim(u, 1, 2:(size(u, 1) - 1)), selectdim(v, 1, 2:(size(v, 1) - 1)))

@kernel function vectorgradient!(G, u, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = wavenumber_full(g, I)
    ux, uy = u.x[I], u.y[I]
    G.xx[I] = im * kx * ux
    G.xy[I] = im * ky * ux
    G.yx[I] = im * kx * uy
    G.yy[I] = im * ky * uy
end
@kernel function vectorgradient!(G, u, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumber_full(g, I)
    ux, uy, uz = u.x[I], u.y[I], u.z[I]
    G.xx[I] = im * kx * ux
    G.xy[I] = im * ky * ux
    G.xz[I] = im * kz * ux
    G.yx[I] = im * kx * uy
    G.yy[I] = im * ky * uy
    G.yz[I] = im * kz * uy
    G.zx[I] = im * kx * uz
    G.zy[I] = im * ky * uz
    G.zz[I] = im * kz * uz
end

@kernel function derivative!(du, u, j, g::Grid)
    I = @index(Global, Cartesian)
    kj = wavenumber_full(g, I)[j]
    du[I] = im * kj * u[I]
end

"Symmetric part of a velocity-gradient tensor field (physical space)."
strain_from_gradient(G, ::Grid{2}) = (; G.xx, G.yy, xy = (G.xy .+ G.yx) ./ 2)
strain_from_gradient(G, ::Grid{3}) = (;
    G.xx,
    G.yy,
    G.zz,
    xy = (G.xy .+ G.yx) ./ 2,
    yz = (G.yz .+ G.zy) ./ 2,
    zx = (G.zx .+ G.xz) ./ 2,
)

"Pointwise double-dot `τ : S` for symmetric tensors (sub-grid dissipation)."
contract_dissipation(τ, S, ::Grid{2}) =
    @. τ.xx * S.xx + τ.yy * S.yy + 2 * τ.xy * S.xy
contract_dissipation(τ, S, ::Grid{3}) =
    @. τ.xx * S.xx +
       τ.yy * S.yy +
       τ.zz * S.zz +
       2 * (τ.xy * S.xy + τ.yz * S.yz + τ.zx * S.zx)

@kernel function strainrate!(S, u, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = wavenumber_full(g, I)
    ux, uy = u.x[I], u.y[I]
    S.xx[I] = im * kx * ux
    S.yy[I] = im * ky * uy
    S.xy[I] = im * (ky * ux + kx * uy) / 2
end
@kernel function strainrate!(S, u, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumber_full(g, I)
    ux, uy, uz = u.x[I], u.y[I], u.z[I]
    S.xx[I] = im * kx * ux
    S.yy[I] = im * ky * uy
    S.zz[I] = im * kz * uz
    S.xy[I] = im * (ky * ux + kx * uy) / 2
    S.yz[I] = im * (kz * uy + ky * uz) / 2
    S.zx[I] = im * (kx * uz + kz * ux) / 2
end

@kernel function dissipation_kernel!(diss, u, visc, g::Grid{2})
    I = @index(Global, Cartesian)
    k2 = squared_wavenumber_full(g, I)
    u2 = abs2(u.x[I]) + abs2(u.y[I])
    diss[I] = visc * k2 * u2
end
@kernel function dissipation_kernel!(diss, u, visc, g::Grid{3})
    I = @index(Global, Cartesian)
    k2 = squared_wavenumber_full(g, I)
    u2 = abs2(u.x[I]) + abs2(u.y[I]) + abs2(u.z[I])
    diss[I] = visc * k2 * u2
end
function get_dissipation!(diss, u, visc, g)
    apply!(dissipation_kernel!, g, (diss, u, visc, g))
    return d = sum(diss) + sum(selectdim(diss, 1, 2:(size(diss, 1) - 1)))
end

function turbulence_statistics(u, visc, g, dissfield = similar(u.x, typeof(g.l)))
    D = dim(g)
    foreach(u -> apply!(twothirds!, g, (u, g)), u) # Empty ghost modes
    e = sum(getenergy, u) / 2
    uavg = sqrt(2 * e / D)
    diss = get_dissipation!(dissfield, u, visc, g)
    l_kol = (visc^3 / diss)^(1 / 4)
    l_tay = sqrt(15 * visc / diss) * uavg
    l_int = uavg^3 / diss
    t_int = l_int / uavg
    t_tay = l_tay / uavg
    t_kol = visc / diss |> sqrt
    Re_int = l_int * uavg / visc
    Re_tay = l_tay * uavg / visc
    Re_kol = l_kol * uavg / visc
    return (; uavg, diss, l_int, l_tay, l_kol, t_int, t_tay, t_kol, Re_int, Re_tay, Re_kol)
end

function forwardeuler!(f!, u, cache, grid, Δt; args...)
    (; du) = cache
    f!(du, u, grid; args...)
    for i in 1:dim(grid)
        axpy!(Δt, du[i], u[i])
    end
    return apply!(project!, grid, (u, grid))
end

"Adams-Bashforth-Crank-Nicolson time stepping."
@kernel function abcn_kernel!(u, du, du_old, Δt, visc, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = wavenumber_full(g, I)
    a = Δt / 2 * visc * (kx^2 + ky^2)
    u.x[I] = (1 - a) / (1 + a) * u.x[I] + Δt / (1 + a) * (3 * du.x[I] - du_old.x[I]) / 2
    u.y[I] = (1 - a) / (1 + a) * u.y[I] + Δt / (1 + a) * (3 * du.y[I] - du_old.y[I]) / 2
end
@kernel function abcn_kernel!(u, du, du_old, Δt, visc, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumber_full(g, I)
    a = Δt / 2 * visc * (kx^2 + ky^2 + kz^2)
    u.x[I] = (1 - a) / (1 + a) * u.x[I] + Δt / (1 + a) * (3 * du.x[I] - du_old.x[I]) / 2
    u.y[I] = (1 - a) / (1 + a) * u.y[I] + Δt / (1 + a) * (3 * du.y[I] - du_old.y[I]) / 2
    u.z[I] = (1 - a) / (1 + a) * u.z[I] + Δt / (1 + a) * (3 * du.z[I] - du_old.z[I]) / 2
end

function abcn!(u, cache, Δt, visc, grid; firststep)
    (; σ, vi_vj, v, plan, du, ustart) = cache
    du_old = ustart # use same name as wray3 cache
    nonlinearity!(σ, vi_vj, v, u, plan, grid)
    apply!(tensordivergence!, grid, (du, σ, grid))
    firststep && foreach(copyto!, du_old, du) # Forward-Euler for first step
    apply!(abcn_kernel!, grid, (u, du, du_old, Δt, visc, grid))
    apply!(project!, grid, (u, grid))
    return foreach(copyto!, du_old, du)
end

function propose_timestep(u, grid, visc, cache)
    (; vi_vj, du, v, plan) = cache
    D = dim(grid)
    for i in 1:D
        copyto!(du[i], u[i])
        apply!(twothirds!, grid, (du[i], grid)) # Zero out ghost modes
        ldiv!(v[i], plan, du[i]) # ldiv! overwrites input...
        v[i] .*= get_fft_fac(grid) # FFT factor
    end
    D == 2 && @. vi_vj = sqrt(v.x^2 + v.y^2)
    D == 3 && @. vi_vj = sqrt(v.x^2 + v.y^2 + v.z^2)
    vmax = maximum(vi_vj)
    h = grid.l / grid.n
    Δt_conv = h / vmax
    Δt_diff = h^2 / D / 2 / visc
    return min(Δt_conv, Δt_diff)
end

function convectiondiffusion!(du, u, grid, cache; visc)
    (; plan, σ, vi_vj, v) = cache
    stress!(σ, vi_vj, v, u, plan, visc, grid)
    return apply!(tensordivergence!, grid, (du, σ, grid))
end

"Pre-allocate temporary arrays and RFFT plan for time stepping."
getcache(grid) = (;
    ustart = vectorfield(grid),
    du = vectorfield(grid),
    σ = tensorfield(grid),
    vi_vj = spacescalarfield(grid),
    v = spacevectorfield(grid),
    plan = getplan(grid),
)

"Perform time step using Wray's third-order scheme."
function wray3!(f!, u, Δt, grid, cache; args...)
    (; ustart, du) = cache
    T = eltype(u.x)
    D = dim(grid)

    # Low-storage Butcher tableau:
    # c1 | 0             ⋯   0
    # c2 | a1  0         ⋯   0
    # c3 | b1 a2  0      ⋯   0
    # c4 | b1 b2 a3  0   ⋯   0
    #  ⋮ | ⋮   ⋮  ⋮  ⋱   ⋱   ⋮
    # cn | b1 b2 b3  ⋯ an-1  0
    # ---+--------------------
    #    | b1 b2 b3  ⋯ bn-1 an
    #
    # Note the definition of (ai)i.
    # They are shifted to simplify the for-loop.
    # TODO: Make generic by passing a, b, c as inputs
    a = T(8 / 15), T(5 / 12), T(3 / 4)
    b = T(1 / 4), T(0)
    c = T(0), T(8 / 15), T(2 / 3)
    nstage = length(a)

    # Update current solution
    foreach(copyto!, ustart, u)

    for i in 1:nstage
        f!(du, u, grid, cache; args...)

        # Compute u = project(ustart + Δt * a[i] * du)
        i == 1 || foreach(copyto!, u, ustart) # Skip first iter
        for j in 1:D
            axpy!(a[i] * Δt, du[j], u[j])
        end
        apply!(project!, grid, (u, grid))

        # Compute ustart = ustart + Δt * b[i] * du
        # Skip last iter
        i < nstage && for j in 1:D
            axpy!(b[i] * Δt, du[j], ustart[j])
        end
    end

    return u
end

"Ornstein-Uhlenbeck forcing setup."
function ouforcer(grid, kcut)
    kmax = div(grid.n, 2)
    D = dim(grid)
    k2 = if D == 2
        kx = 0:kmax # For RFFT, the x-wavenumbers are 0:kmax
        ky = reshape(map(i -> fftfreq_int(grid, i), 1:grid.n), 1, :) # Normal FFT wavenumbers
        @. kx^2 + ky^2
    else
        kx = 0:kmax # For RFFT, the x-wavenumbers are 0:kmax
        ky = reshape(map(i -> fftfreq_int(grid, i), 1:grid.n), 1, :) # Normal FFT wavenumbers
        kz = reshape(map(i -> fftfreq_int(grid, i), 1:grid.n), 1, 1, :) # Normal FFT wavenumbers
        @. kx^2 + ky^2 + kz^2
    end
    k2 = reshape(k2, :)
    iuse = findall(k2 -> 0 < k2 < kcut^2, k2) # Exclude 0-th mode
    kuse = k2[iuse]
    nuse = length(iuse)
    x = complex(grid.l)
    b = KernelAbstractions.zeros(grid.backend, typeof(x), nuse, D)
    bold = zero(b)
    return (; iuse, kuse, b, bold)
end

"Get indices of wavenumbers with `|k| < kband`."
function getband(grid, kband)
    kmax = div(grid.n, 2)
    D = dim(grid)

    # Get squared wavenumbers in an RFFT-shaped array
    k2 = map(I -> squared_wavenumber_int(grid, I), CartesianIndices(ndrange(grid)))

    # Flatten since we work with linear RFFT indices
    k2 = reshape(k2, :)

    # Indices in band (without zero mode)
    inds = filter(i -> 0 < k2[i] < kband^2, eachindex(k2))

    # We need to adapt the band for RFFT.
    # Consider the following example:
    #
    # julia> n, kmax = 8, 4;
    # julia> u = randn(n, n, n);
    # julia> f = fft(u); r = rfft(u);
    # julia> sum(abs2, f)
    # 275142.33506202063
    # julia> sum(abs2, r) + sum(abs2, selectdim(r, 1, 2:kmax))
    # 275142.3350620207
    #
    # To compute the energy of the FFT, we need an additional term for RFFT.
    # The second term sums over all the x-indices except for 1 and kmax + 1.
    # We thus need to add indices to account for the conjugate symmetry in RFFT.
    # For an RFFT array r of size (kmax + 1, n, n), we have the linear index relation
    # r[i] == r[x, y, z]
    # if
    # i == x + (y - 1) * (kmax + 1) + (z - 1) * (kmax + 1) * n.
    # We therefore need to exclude the indices:
    # (x == 1), i.e. (i % (kmax + 1) == 1), and
    # (x == kmax + 1), i.e. (i % (kmax + 1) == 0).
    # We only keep i if (i % (kmax + 1) > 1).
    conjinds = filter(j -> j % (kmax + 1) > 1, inds)

    return (; inds, conjinds, k2 = k2[inds])
end

"Get indices and wavenumbers for the `i`-th shell (`k` such that `i ≤ |k| < i + 1`)."
function getshells(grid, shells)
    kmax = div(grid.n, 2)
    D = dim(grid)

    # Get squared wavenumbers in an RFFT-shaped array
    k2 = map(I -> squared_wavenumber_int(grid, I), CartesianIndices(ndrange(grid)))

    # Flatten since we work with linear RFFT indices
    k2 = reshape(k2, :)

    isort = sortperm(k2) # Permutation for sorting the wavenumbers
    k2sort = k2[isort]

    # Get linear RFFT indices and corresponding waveumbers for each shell
    return map(shells) do i
        # Since the wavenumbers are sorted, we just need to find the start and stop of each shell.
        # The linear indices for that shell is then given by the permutation in that range.
        jstart = findfirst(≥(i^2), k2sort)
        jstop = findfirst(≥((i + 1)^2), k2sort)
        isnothing(jstop) && (jstop = length(k2sort) + 1) # findfirst may return nothing
        jstop -= 1
        inds = isort[jstart:jstop] # Linear indices of the i-th shell

        # We need to adapt the shells for RFFT.
        # Consider the following example:
        #
        # julia> n, kmax = 8, 4;
        # julia> u = randn(n, n, n);
        # julia> f = fft(u); r = rfft(u);
        # julia> sum(abs2, f)
        # 275142.33506202063
        # julia> sum(abs2, r) + sum(abs2, selectdim(r, 1, 2:kmax))
        # 275142.3350620207
        #
        # To compute the energy of the FFT, we need an additional term for RFFT.
        # The second term sums over all the x-indices except for 1 and kmax + 1.
        # We thus need to add indices to account for the conjugate symmetry in RFFT.
        # For an RFFT array r of size (kmax + 1, n, n), we have the linear index relation
        # r[i] == r[x, y, z]
        # if
        # i == x + (y - 1) * (kmax + 1) + (z - 1) * (kmax + 1) * n.
        # We therefore need to exclude the indices:
        # (x == 1), i.e. (i % (kmax + 1) == 1), and
        # (x == kmax + 1), i.e. (i % (kmax + 1) == 0).
        # We only keep i if (i % (kmax + 1) > 1).
        conjinds = filter(j -> j % (kmax + 1) > 1, inds)

        (; shell = i, inds = (inds, conjinds), k2 = (k2[inds], k2[conjinds]))
    end
end

"""
Precompute per-shell index sets and reference energies for the
forcing-via-energy-clamp pattern. `u` is the reference (initial) field.
"""
energy_shells(grid, shells, u) = map(getshells(grid, shells)) do s
    inds = s.inds[1] |> adapt(grid.backend)
    energyinds = vcat(s.inds[1], s.inds[2]) |> adapt(grid.backend)
    eref = sum(u -> sum(abs2, view(u, energyinds)), u) / 2
    (; inds, energyinds, eref)
end

"""
Rescale each shell of `u` to match the reference energy stored in `shells`
(output of `energy_shells`).
"""
function maintain_shell_energy!(u, shells)
    for s in shells
        eshell = sum(u -> sum(abs2, view(u, s.energyinds)), u) / 2
        foreach(u -> (view(u, s.inds) .*= sqrt(s.eref / eshell)), u)
    end
    return u
end

# end
