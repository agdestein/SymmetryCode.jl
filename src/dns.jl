# Truncation-aware DNS: the single-device design of the DistributedNavierStokes
# spectral solver, ported to `Grid{D}`. Opt-in; the full-array solver in
# `solver.jl` (`wray3!`, `convectiondiffusion!`, `propose_timestep`) is unchanged.
#
# 1. Storage. The state holds only the modes the 2/3 rule retains: kx ∈ 0:kcut
#    on the RFFT axis and |k| ≤ kcut on each full axis, kcut = n ÷ 3, in
#    "compact" order (odd-size fftfreq, m = 2kcut + 1). A compact array is
#    (kcut + 1, m) in 2D and (kcut + 1, m, m) in 3D: 0.44× resp. 0.30× of the
#    RFFT array, so every spectral-space pass reads that much less.
# 2. Transforms. Zero padding happens inside the pipeline: a compact field is
#    padded into one RFFT-sized scratch array and backward-transformed
#    without cuFFT's input-protecting copy (the scratch is destroyed); forward
#    transforms truncate and normalize in one kernel. The physical velocity is
#    dealiased by construction, so no `twothirds!` pass is ever needed.
# 3. Right-hand side. The products v_i v_j are transformed one at a time and
#    their divergence −i k_j σ̂_ij is accumulated straight into `du`; the
#    stress tensor is never stored.
# 4. Time stepping. Wray's low-storage RK3 with an integrating factor: a
#    diagonal linear damping rate L(k²) (viscosity, hypo- or hyperviscosity)
#    is advanced exactly, φᵢ = exp(−L(k²)(c_{i+1} − c_i)Δt) on both
#    registers, and has no time-step limit. The projection is fused into the
#    stage update (it commutes with φ), so the state stays solenoidal.
# 5. Time step. Δt = cfl·√3/(kmax·max_x Σ_c |v_c|), the fraction `cfl` of the
#    frozen-velocity stability boundary of RK3 on the imaginary axis (√3),
#    with kmax = 2π·kcut/l; the velocity maximum comes from the first stage's
#    physical velocity, which is in hand anyway (no extra transforms).
#
# Conventions follow `solver.jl`: û = F[u]/n^D, NamedTuples of component
# arrays, 2D and 3D kernels as separate methods. Unlike `twothirds!`, the
# compact layout keeps the k = 0 mode; a mean-free initial field stays
# mean-free, since the nonlinear term vanishes there and the forcing helpers
# never touch it. Use `pad!` / `truncate!` to hand a state to the full-array
# tools (filters, `sfs!`, spectra, `turbulence_statistics`).

"Truncation wavenumber and compact length of `g`'s retained-mode layout."
@inline compactsizes(g::Grid) = (kcut = div(g.n, 3), m = 2 * div(g.n, 3) + 1)

"Compact-array size (RFFT axis first)."
compact_ndrange(g::Grid{2}) = (k = compactsizes(g); (k.kcut + 1, k.m))
compact_ndrange(g::Grid{3}) = (k = compactsizes(g); (k.kcut + 1, k.m, k.m))

"Integer wavenumber of compact index `j` on a full axis."
@inline compactfreq(j, m, kcut) = ifelse(j ≤ kcut + 1, j - 1, j - 1 - m)

"Full-axis (fftfreq-order) index of compact index `j`."
@inline fullindex(j, n, m, kcut) = ifelse(j ≤ kcut + 1, j, j + (n - m))

"Is full-axis index `j` a retained mode? (`|fftfreq| ≤ kcut`)"
@inline retained(j, n, kcut) = (j ≤ kcut + 1) | (j ≥ n - kcut + 1)

"Scalar spectral field on the compact layout."
compactfield(g::Grid{D, T}) where {D, T} =
    KernelAbstractions.zeros(g.backend, Complex{T}, compact_ndrange(g))

"Vector spectral field on the compact layout."
compactvectorfield(g::Grid{2}) = (; x = compactfield(g), y = compactfield(g))
compactvectorfield(g::Grid{3}) =
    (; x = compactfield(g), y = compactfield(g), z = compactfield(g))

"Viscous damping rate `visc·k²` (the default linear operator)."
struct ViscousRate{T}
    visc::T
end
@inline (r::ViscousRate)(k2) = r.visc * k2

# ---------------------------------------------------------------- FFT plans

# cuFFT plans carry a hidden field-sized buffer each: the R2C plan's serves
# only `ldiv!` (never used here), and the C2R plan's protects the input of
# cuFFT's destructive out-of-place transform by copying it first. Our
# backward input is pipeline scratch, so the copy is skipped by executing
# the plan directly, and both buffers are freed. The executor is internal
# cuFFT API; without it the C2R plan keeps its buffer and the public `mul!`
# is used (correct, one extra copy).

const CUFFT_DIRECT =
    isdefined(CUDA.CUFFT, :unsafe_execute_external_batches!) ?
    CUDA.CUFFT.unsafe_execute_external_batches! :
    isdefined(CUDA.CUFFT, :unsafe_execute!) ? CUDA.CUFFT.unsafe_execute! : nothing

"Free an FFT plan's hidden staging buffer where it is never needed."
trim_plan!(p) = p
function trim_plan!(p::CUDA.CUFFT.CuFFTPlan{T}) where {T}
    if !isnothing(p.buffer) && (T <: Complex || !isnothing(CUFFT_DIRECT))
        CUDA.unsafe_free!(p.buffer)
    end
    return p
end

"Backward real FFT `y ← plan * x`, where `x` is scratch and is destroyed."
brfft_scratch!(y, plan, x) = mul!(y, plan, x)
function brfft_scratch!(y::CuArray, plan::CUDA.CUFFT.CuFFTPlan, x::CuArray)
    isnothing(CUFFT_DIRECT) ? mul!(y, plan, x) : CUFFT_DIRECT(plan, x, y)
    return y
end

# ---------------------------------------------------------------- setup

"""
    d = dnssetup(g; visc = 0, linop = nothing)

Truncation-aware DNS context on grid `g`: FFT plans, the pipeline scratch
(one RFFT-sized spectral array, the physical velocity and one product
buffer) and the linear damping rate `linop(k²)` that the integrating factor
applies exactly. `linop` is any isbits callable on squared physical
wavenumbers (default `visc·k²`); it is never evaluated at k = 0, which is
left undamped. Compute steep rates (e.g. `ν₈ k¹⁶`) in a type wide enough
not to overflow: with `Float32`, evaluate in `Float64` and convert.
"""
function dnssetup(g::Grid{D, T}; visc = zero(T), linop = nothing) where {D, T}
    (; kcut, m) = compactsizes(g)
    3kcut ≤ g.n || error("aliasing: 3kcut ≤ n is required")
    w = spacescalarfield(g)
    v = spacevectorfield(g)
    â = scalarfield(g)
    rplan = trim_plan!(plan_rfft(w))
    bplan = trim_plan!(plan_brfft(â, g.n))
    L = isnothing(linop) ? ViscousRate(T(visc)) : linop
    return (; g, kcut, m, kfac = T(2π) / g.l, linop = L, w, v, â, rplan, bplan)
end

"Two compact registers for [`dns_step!`](@ref)."
dnscache(d) = (; ustart = compactvectorfield(d.g), du = compactvectorfield(d.g))

# ---------------------------------------------------------------- layout kernels

@kernel function dnspad!(â, uc, kcut, m, g::Grid{2})
    i, j = @index(Global, NTuple)
    n = g.n
    keep = (i ≤ kcut + 1) & retained(j, n, kcut)
    jc = ifelse(j ≤ kcut + 1, j, j - (n - m))
    @inbounds â[i, j] = keep ? uc[i, jc] : zero(eltype(â))
end
@kernel function dnspad!(â, uc, kcut, m, g::Grid{3})
    i, j, k = @index(Global, NTuple)
    n = g.n
    keep = (i ≤ kcut + 1) & retained(j, n, kcut) & retained(k, n, kcut)
    jc = ifelse(j ≤ kcut + 1, j, j - (n - m))
    kc = ifelse(k ≤ kcut + 1, k, k - (n - m))
    @inbounds â[i, j, k] = keep ? uc[i, jc, kc] : zero(eltype(â))
end

@kernel function dnstruncate!(uc, â, s, kcut, m, g::Grid{2})
    i, j = @index(Global, NTuple)
    @inbounds uc[i, j] = s * â[i, fullindex(j, g.n, m, kcut)]
end
@kernel function dnstruncate!(uc, â, s, kcut, m, g::Grid{3})
    i, j, k = @index(Global, NTuple)
    n = g.n
    @inbounds uc[i, j, k] = s * â[i, fullindex(j, n, m, kcut), fullindex(k, n, m, kcut)]
end

"""
Full RFFT field `u` ← compact field `uc`, zero outside the retained modes
(the inverse of [`truncate!`](@ref) on 2/3-truncated fields). Works on
scalars and on NamedTuples of components.
"""
pad!(u::NamedTuple, uc::NamedTuple, d) = (foreach((a, b) -> pad!(a, b, d), u, uc); u)
pad!(u, uc, d) = (apply!(dnspad!, d.g, (u, uc, d.kcut, d.m, d.g)); u)

"Compact field `uc` ← the retained modes of full RFFT field `u`."
truncate!(uc::NamedTuple, u::NamedTuple, d) =
    (foreach((a, b) -> truncate!(a, b, d), uc, u); uc)
truncate!(uc, u, d) = (
    apply!(
        dnstruncate!, d.g, (uc, u, one(real(eltype(uc))), d.kcut, d.m, d.g);
        ndrange = size(uc),
    ); uc
)

# ---------------------------------------------------------------- transforms

"Physical field `v` ← compact spectral `uc` (the scratch `d.â` is destroyed)."
function dns_to_phys!(v, uc, d)
    apply!(dnspad!, d.g, (d.â, uc, d.kcut, d.m, d.g))
    brfft_scratch!(v, d.bplan, d.â)
    return v
end
dns_to_phys!(v::NamedTuple, uc::NamedTuple, d) =
    (foreach((a, b) -> dns_to_phys!(a, b, d), v, uc); v)

"Compact spectral `uc` ← physical field `w` (û = F[w]/n^D, truncated)."
function dns_to_spec!(uc, w, d)
    T = real(eltype(uc))
    mul!(d.â, d.rplan, w)
    s = one(T) / T(get_fft_fac(d.g))
    apply!(dnstruncate!, d.g, (uc, d.â, s, d.kcut, d.m, d.g); ndrange = size(uc))
    return uc
end

# ---------------------------------------------------------------- right-hand side

# Accumulate the divergence of one transformed product σ̂_ab into `du`:
# du_a (+)= −i k_b σ̂_ab and, off the diagonal, du_b (+)= −i k_a σ̂_ab. The
# product is read from the RFFT scratch at the retained positions and
# normalized on the way (one kernel instead of truncate + divergence).

@kernel function dnsdivacc!(dua, dub, â, a, b, init, s, kcut, m, kfac, g::Grid{2})
    i, j = @index(Global, NTuple)
    @inbounds begin
        σ = s * â[i, fullindex(j, g.n, m, kcut)]
        k = (kfac * (i - 1), kfac * compactfreq(j, m, kcut))
        ca = -im * k[b] * σ
        dua[i, j] = init ? ca : dua[i, j] + ca
        if a != b
            cb = -im * k[a] * σ
            dub[i, j] = init ? cb : dub[i, j] + cb
        end
    end
end
@kernel function dnsdivacc!(dua, dub, â, a, b, init, s, kcut, m, kfac, g::Grid{3})
    i, j, l = @index(Global, NTuple)
    n = g.n
    @inbounds begin
        σ = s * â[i, fullindex(j, n, m, kcut), fullindex(l, n, m, kcut)]
        k = (kfac * (i - 1), kfac * compactfreq(j, m, kcut), kfac * compactfreq(l, m, kcut))
        ca = -im * k[b] * σ
        dua[i, j, l] = init ? ca : dua[i, j, l] + ca
        if a != b
            cb = -im * k[a] * σ
            dub[i, j, l] = init ? cb : dub[i, j, l] + cb
        end
    end
end

"Products (a, b, initialize) in accumulation order: diagonal first."
dnsproducts(::Grid{2}) = ((1, 1, true), (2, 2, true), (1, 2, false))
dnsproducts(::Grid{3}) =
    ((1, 1, true), (2, 2, true), (3, 3, true), (1, 2, false), (2, 3, false), (3, 1, false))

"""
Convective right-hand side `du = −i k_j F[v_i v_j]` on the compact layout
(unprojected; [`dns_step!`](@ref) projects). Viscosity is not included: the
integrating factor applies it. The physical velocity stays in `d.v`.
"""
function dns_rhs!(du, uh, d)
    (; g, w, v, â, rplan) = d
    T = real(eltype(du.x))
    dns_to_phys!(v, uh, d)
    s = one(T) / T(get_fft_fac(g))
    for (a, b, init) in dnsproducts(g)
        @. w = v[a] * v[b]
        mul!(â, rplan, w)
        args = (du[a], du[b], â, a, b, init, s, d.kcut, d.m, d.kfac, g)
        apply!(dnsdivacc!, g, args; ndrange = size(du.x))
    end
    return du
end

# ---------------------------------------------------------------- time stepping

# Wray RK3 in low-storage form (as `wray3!`): a, b, and dc = c_{i+1} − c_i.
const WRAY3 = (a = (8 / 15, 5 / 12, 3 / 4), b = (1 / 4, 0.0, 0.0), dc = (8 / 15, 2 / 15, 1 / 3))

@kernel function dnsifstage!(uh, ust, src, du, a, b, dcΔt, Δt, writestart, linop, kcut, m, kfac, g::Grid{2})
    i, j = @index(Global, NTuple)
    @inbounds begin
        kx, ky = kfac * (i - 1), kfac * compactfreq(j, m, kcut)
        kk = kx * kx + ky * ky
        zero_k = kk == 0
        φ = zero_k ? one(kk) : exp(-linop(kk) * dcΔt)
        dx, dy = du.x[i, j], du.y[i, j]
        p = zero_k ? zero(dx) : (kx * dx + ky * dy) / kk
        dx, dy = dx - kx * p, dy - ky * p
        sx, sy = src.x[i, j], src.y[i, j]
        uh.x[i, j] = φ * (sx + a * Δt * dx)
        uh.y[i, j] = φ * (sy + a * Δt * dy)
        if writestart
            ust.x[i, j] = φ * (sx + b * Δt * dx)
            ust.y[i, j] = φ * (sy + b * Δt * dy)
        end
    end
end
@kernel function dnsifstage!(uh, ust, src, du, a, b, dcΔt, Δt, writestart, linop, kcut, m, kfac, g::Grid{3})
    i, j, l = @index(Global, NTuple)
    @inbounds begin
        kx = kfac * (i - 1)
        ky = kfac * compactfreq(j, m, kcut)
        kz = kfac * compactfreq(l, m, kcut)
        kk = kx * kx + ky * ky + kz * kz
        zero_k = kk == 0
        φ = zero_k ? one(kk) : exp(-linop(kk) * dcΔt)
        dx, dy, dz = du.x[i, j, l], du.y[i, j, l], du.z[i, j, l]
        p = zero_k ? zero(dx) : (kx * dx + ky * dy + kz * dz) / kk
        dx, dy, dz = dx - kx * p, dy - ky * p, dz - kz * p
        sx, sy, sz = src.x[i, j, l], src.y[i, j, l], src.z[i, j, l]
        uh.x[i, j, l] = φ * (sx + a * Δt * dx)
        uh.y[i, j, l] = φ * (sy + a * Δt * dy)
        uh.z[i, j, l] = φ * (sz + a * Δt * dz)
        if writestart
            ust.x[i, j, l] = φ * (sx + b * Δt * dx)
            ust.y[i, j, l] = φ * (sy + b * Δt * dy)
            ust.z[i, j, l] = φ * (sz + b * Δt * dz)
        end
    end
end

@kernel function dnsproject!(u, kcut, m, kfac, g::Grid{2})
    i, j = @index(Global, NTuple)
    @inbounds begin
        kx, ky = kfac * (i - 1), kfac * compactfreq(j, m, kcut)
        kk = kx * kx + ky * ky
        ux, uy = u.x[i, j], u.y[i, j]
        p = kk == 0 ? zero(ux) : (kx * ux + ky * uy) / kk
        u.x[i, j] = ux - kx * p
        u.y[i, j] = uy - ky * p
    end
end
@kernel function dnsproject!(u, kcut, m, kfac, g::Grid{3})
    i, j, l = @index(Global, NTuple)
    @inbounds begin
        kx = kfac * (i - 1)
        ky = kfac * compactfreq(j, m, kcut)
        kz = kfac * compactfreq(l, m, kcut)
        kk = kx * kx + ky * ky + kz * kz
        ux, uy, uz = u.x[i, j, l], u.y[i, j, l], u.z[i, j, l]
        p = kk == 0 ? zero(ux) : (kx * ux + ky * uy + kz * uz) / kk
        u.x[i, j, l] = ux - kx * p
        u.y[i, j, l] = uy - ky * p
        u.z[i, j, l] = uz - kz * p
    end
end

"Make the compact velocity `uh` solenoidal (leaves k = 0 intact)."
dns_project!(uh, d) =
    (apply!(dnsproject!, d.g, (uh, d.kcut, d.m, d.kfac, d.g); ndrange = size(uh.x)); uh)

"Maximum over the grid of the pointwise component sum Σ_c |v_c| (one fused reduction)."
vsummax(v::NamedTuple{(:x, :y)}) = mapreduce((a, b) -> abs(a) + abs(b), max, v.x, v.y)
vsummax(v::NamedTuple{(:x, :y, :z)}) =
    mapreduce((a, b, c) -> abs(a) + abs(b) + abs(c), max, v.x, v.y, v.z)

"Velocity scale of the stability bound for the compact state `uh` (one inverse transform)."
dns_vmax(uh, d) = vsummax(dns_to_phys!(d.v, uh, d))

"""
    Δt = dns_timestep(vmax, d; cfl = 0.85)

The fraction `cfl` of RK3's frozen-velocity stability boundary for the
convective term, `cfl·√3/(kmax·vmax)` with `kmax = 2π·kcut/l` and `vmax`
the maximum of Σ_c |v_c| (from [`dns_step!`](@ref) or [`dns_vmax`](@ref)).
Keep `cfl < 1`: `vmax` lags one step and the analysis freezes the velocity.
The linear damping needs no limit (integrating factor).
"""
function dns_timestep(vmax, d; cfl = 0.85)
    T = typeof(d.kfac)
    return T(cfl) * sqrt(T(3)) / (d.kfac * d.kcut * (T(vmax) + eps(T)))
end

"""
    vmax = dns_step!(uh, Δt, d, cache = dnscache(d); rhs! = nothing)

One IF-RK3 step of the compact state `uh` (solenoidal; see
[`dns_project!`](@ref)). `rhs!(du, uh, d)`, if given, adds to the convective
right-hand side at every stage (e.g. a forcing); additions are projected
with it. Returns the stability velocity scale of the stage-1 velocity
(time tₙ), for the next step's [`dns_timestep`](@ref).
"""
function dns_step!(uh, Δt, d, cache = dnscache(d); rhs! = nothing)
    (; ustart, du) = cache
    (; g) = d
    T = real(eltype(uh.x))
    Δt = T(Δt)
    vmax = zero(T)
    for i in 1:3
        dns_rhs!(du, uh, d)
        isnothing(rhs!) || rhs!(du, uh, d)
        i == 1 && (vmax = vsummax(d.v))
        src = i == 1 ? uh : ustart # stage 1 reads the state itself: no copy
        args = (
            uh, ustart, src, du, T(WRAY3.a[i]), T(WRAY3.b[i]), T(WRAY3.dc[i]) * Δt, Δt,
            i < 3, d.linop, d.kcut, d.m, d.kfac, g,
        )
        apply!(dnsifstage!, g, args; ndrange = size(uh.x))
    end
    return vmax
end

# ---------------------------------------------------------------- diagnostics

"Mean kinetic energy ⟨½|u|²⟩ of a compact field (kx > 0 counted twice; no Nyquist row)."
compact_energy(u) = sum(c -> sum(abs2, c) + sum(abs2, selectdim(c, 1, 2:size(c, 1))), u) / 2

"""
Shell index sets and reference energies on the compact layout, in the form
`maintain_shell_energy!` takes (shell s: s² ≤ |k_int|² < (s+1)²). `uh` is
the reference (initial) field.
"""
function compact_energy_shells(g::Grid, shells, uh)
    (; kcut, m) = compactsizes(g)
    CI = CartesianIndices(compact_ndrange(g))
    k2 = map(I -> (I.I[1] - 1)^2 + sum(j -> compactfreq(j, m, kcut)^2, Base.tail(I.I)), CI)
    k2 = reshape(k2, :)
    dbl = reshape(map(I -> I.I[1] > 1, CI), :)
    return map(shells) do s
        sel = findall(q -> s^2 ≤ q < (s + 1)^2, k2)
        inds = sel |> adapt(g.backend)
        energyinds = vcat(sel, filter(i -> dbl[i], sel)) |> adapt(g.backend)
        eref = sum(u -> sum(abs2, view(u, energyinds)), uh) / 2
        (; inds, energyinds, eref)
    end
end
