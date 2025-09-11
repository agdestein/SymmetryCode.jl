if false
    include("src/SymmetryCode.jl")
    using .SymmetryCode
    using .SymmetryCode.Spectral
end

using Adapt
using CUDA, cuDNN
using FFTW
using KernelAbstractions
using KernelDensity
using LinearAlgebra
using Lux
using MLUtils
using Random
using Seneca
using StaticArrays
using Statistics
using SymmetryCode
using SymmetryCode.Spectral
using WGLMakie
using Zygote
lines([1,2,3])

dns_aid()

setup = (; visc = 5e-7, D = 2, l = 1.0, n_dns = 1024, n_les = 128, backend = CUDABackend())

data = create_data(setup; nstep = 100, nsubstep = 100, rng = Xoshiro(0));

let
    (; visc, D, n_dns, n_les, backend) = setup
    g_dns = Grid{D}(; setup.l, n = n_dns, backend)
    g_les = Grid{D}(; setup.l, n = n_les, backend)
    rng = Xoshiro(0)
    u = randomfield(g_dns; rng, kpeak = 5)
    foreach(u -> randn!(rng, u), u)
    c_dns = getcache(g_dns)
    c_les = getcache(g_les)
    ubar = vectorfield(g_les)
    σbar1 = tensorfield(g_les)
    σbar2 = tensorfield(g_les)
    τ = tensorfield(g_les)
    sfs!(τ, σbar1, σbar2, ubar, u, c_dns, c_les, g_dns, g_les)
    τ.xx .|> abs |> extrema
end

data[2][end].xx .|> abs |> extrema

data[1][1].x |> size
data[2][1].xx |> size
data[2][end].xy .|> abs |> extrema

let
    (; D) = setup
    g = Grid{D}(; setup.l, n = setup.n_les, setup.backend)
    τ = data[1][end][1] |> adapt(setup.backend)
    τxx = spacescalarfield(g)
    plan = plan_rfft(τxx)
    # apply!(twothirds!, g, (τ, g))
    ldiv!(τxx, plan, τ)
    τxx .*= g.n^D
    sum(abs2, τxx) / prod(size(τxx)) |> display
    fig, ax, hm = τxx |> Array |> heatmap
    Colorbar(fig[1, 2], hm)
    display(fig)
end;

grad = let
    (; D) = setup
    g = Grid{D}(; setup.l, n = setup.n_les, setup.backend)
    u = data[1][1] |> adapt(setup.backend)
    G = tensorfield_nonsym(g)
    GG = spacetensorfield_nonsym(g)
    # apply!(twothirds!, g, (u[1], g))
    # apply!(twothirds!, g, (u[2], g))
    apply!(vectorgradient!, g, (G, u, g))
    plan = plan_rfft(GG[1])
    for (GG, G) in zip(GG, G)
        apply!(twothirds!, g, (G, g))
        ldiv!(GG, plan, G) # Inverse RFFT
        GG .*= g.n^D # FFT factor
    end
    fig, ax, hm = GG[3] |> Array |> heatmap
    Colorbar(fig[1, 2], hm)
    display(fig)
    GG
end;

grad |> pairs

let
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    net_stuff = equivariant_net(setup, [10, 10, 20, 20])
    (; project, net, ps, st) = net_stuff
    ps = project(ps)
    x = stack(grad)
    (; elements, permutations, signs) = group_stuff(setup.D)
    i = 6
    ip, is = elements[i]
    p, s = permutations[ip], signs[is]
    xx = tensorfield_to_smatrix(grad)
    rxx = transform_tensor(xx, (p, s))
    rx = smatrix_to_tensorfield(rxx) |> stack
    nx = net(reshape(x, :, 4, 1), ps, st) |> first
    nrx = net(reshape(rx, :, 4, 1), ps, st) |> first
    nx = hcat(nx[:, 1, :], nx[:, 3, :], nx[:, 3, :], nx[:, 2, :]) # Desymmetrize
    nrx = hcat(nrx[:, 1, :], nrx[:, 3, :], nrx[:, 3, :], nrx[:, 2, :]) # Desymmetrize
    nxx = reshape(nx, g.n, g.n, 4)
    nxx = ntuple(i -> view(nxx, :, :, i), 4)
    nxx = tensorfield_to_smatrix(nxx)
    rnxx = transform_tensor(nxx, (p, s))
    rnx = smatrix_to_tensorfield(rnxx) |> stack
    rnx = reshape(rnx, :, 4)
    @show norm(rnx)
    norm(nrx - rnx) / norm(rnx)
end

using ComponentArrays
# net_stuff = equivariant_net(setup, [24, 32, 32, 32]);
net_stuff = cnn(setup, [32, 32, 32, 32]);
net_stuff = (; net_stuff..., ps = NamedTuple(0.1 * ComponentArray(net_stuff.ps |> adapt(CPU())) |> adapt(setup.backend)))
ps, st = train(;
    setup,
    dataloader = create_dataloader(setup, data; batchsize = 10),
    nepoch = 500,
    learning_rate = 1e-3,
    net_stuff,
);
(; net, project) = net_stuff;

net_stuff.ps.mid_1.weight |> std
net_stuff.project(net_stuff.ps).mid_1.weight |> std

model = fullchain(setup, net, project, ps, st)
inference_post(; setup, model)

data_test = create_data(setup; nstep = 100, nsubstep = 100, rng = Xoshiro(321));
# dataloader_test = create_dataloader(setup, data_test; batchsize = 1)
dataloader_test = create_dataloader(setup, data; batchsize = 1)

let
    n = setup.n_les
    D = setup.D
    dev = adapt(setup.backend)
    data = dataloader_test |> collect
    x = getindex.(data, 1) |> stack |> x -> reshape(x, n^D, 4, :) |> dev
    y = getindex.(data, 2) |> stack |> x -> reshape(x, n^D, 3, :) |> dev
    y |> std
end

let
    mx = model(x)
    @show MSELoss()(mx, y)
    @show norm(mx - y) / norm(y)
    i = 1
    dy = y[:, i, :] |> vec |> Array |> kde |> x -> (; x.x, x.density)
    dmx = mx[:, i, :] |> vec |> Array |> kde |> x -> (; x.x, x.density)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "τ₁₁", ylabel = "Density", yscale = log10)
    lines!(ax, dy.x, dy.density; label = "Reference")
    # lines!(ax, dmx.x / sqrt(n), dmx.density * sqrt(n); label = "Prediction")
    lines!(ax, dmx.x, dmx.density; label = "Prediction")
    ylims!(ax, 1e-2, 1e3)
    axislegend(ax)
    fig |> display
    nothing
end

let
    n = setup.n_les
    y = dataloader_test.data[2][:, 1, 17]
    reshape(y, n, n) |> heatmap
end

for (x, y) in dataloader_test
    @show y |> mean
    @show y |> std
    break
end

# 1. Put model in solver
# 2. Dissipation coefficient
# 3. A-posteriori errors/spectra
# 4. TBNN
# 5. Forcing
# 6. 3D
