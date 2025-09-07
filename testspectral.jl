if false
    include("src/SymmetryCode.jl")
    using .SymmetryCode
    using .SymmetryCode.Spectral
end

using CUDA, cuDNN
using FFTW
using KernelAbstractions
using LinearAlgebra
using Lux
using MLUtils
using Random
using Seneca
using StaticArrays
using SymmetryCode
using SymmetryCode.Spectral
using WGLMakie
using Zygote

dns_aid()

setup = (; visc = 4e-4, D = 2, l = 1.0, n_dns = 1024, n_les = 128, backend = CUDABackend())

data = create_data(setup);

data[1].x |> size
data[2].xx |> size

data[2].xx .|> abs |> extrema

grad = let
    (; D) = setup
    g = Grid{D}(; setup.l, n = setup.n_les, setup.backend)
    u = data[1]
    G = tensorfield_nonsym(g)
    GG = map(G -> spacescalarfield(g), G)
    apply!(vectorgradient!, g, (G, u, g))
    plan = plan_rfft(GG.xx)
    for (GG, G) in zip(GG, G)
        ldiv!(GG, plan, G) # Inverse RFFT
        GG .*= g.n^D # FFT factor
    end
    # irfft(u.x, g.n) |> Array |> heatmap
    # GG.yx |> Array |> heatmap
    GG
end

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
    nxx = reshape(nx, g.n, g.n, 4)
    nxx = ntuple(i -> view(nxx, :, :, i), 4)
    nxx = tensorfield_to_smatrix(nxx)
    rnxx = transform_tensor(nxx, (p, s))
    rnx = smatrix_to_tensorfield(rnxx) |> stack
    rnx = reshape(rnx, :, 4)
    norm(nrx - rnx) / norm(rnx)
end

dataloader = let
    batchsize = 64
    (; D) = setup
    g = Grid{D}(; setup.l, n = setup.n_les, setup.backend)
    u, τ = data
    G = tensorfield_nonsym(g)
    GG = map(G -> spacescalarfield(g), G)
    ττ = map(G -> spacescalarfield(g), τ)
    apply!(vectorgradient!, g, (G, u, g))
    plan = plan_rfft(GG.xx)
    for (GG, G, ττ, τ) in zip(GG, G, ττ, τ)
        ldiv!(GG, plan, G) # Inverse RFFT
        ldiv!(ττ, plan, copy(τ)) # Inverse RFFT
        GG .*= g.n^D # FFT factor
        ττ .*= g.n^D # FFT factor
    end
    GG = stack(GG)
    ττ = stack(ττ)
    x = reshape(GG, :, D^2) |> cpu_device()
    y = reshape(ττ, :, tensordim(g)) |> cpu_device()
    x = permutedims(x, (2, 1))
    y = permutedims(y, (2, 1))
    x = reshape(x, 1, size(x)...)
    y = reshape(y, 1, size(y)...)
    DataLoader((x, y); batchsize, shuffle = true, partial = false)
end

net_stuff = equivariant_net(setup, [10, 10, 20, 20]);

ps, st = train(; setup, dataloader, nepoch = 50, learning_rate = 1e-3, net_stuff)

# net_stuff = (; net_stuff.net, net_stuff.project, ps, st)

ps.lift.weight

using Statistics
for (x, y) in dataloader
    @show x |> mean
    @show x |> std
    break
end
