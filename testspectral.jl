if false
    include("src/SymmetryCode.jl")
    using .SymmetryCode
    using .SymmetryCode.Spectral
end

using CUDA, cuDNN
using FFTW
using KernelAbstractions
using LinearAlgebra
using Random
using Seneca
using SymmetryCode
using SymmetryCode.Spectral
using WGLMakie

dns_aid()

setup = (;
    visc = 4e-4,
    D = 2,
    l = 1.0,
    n_dns = 1024,
    n_les = 128,
    backend = CUDABackend()
)

data = create_data(setup);

data[1].x |> size
data[2].xx |> size

data[2].xx .|> abs |> extrema

net_stuff = equivariant_net(setup, [10, 10, 20, 20]);

grad = let
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    u = data[1]
    G = tensorfield_nonsym(g)
    GG = map(G -> spacescalarfield(g), G)
    apply!(vectorgradient!, g, (G, u, g))
    plan = plan_rfft(GG.xx)
    for (GG, G) in zip(GG, G)
        ldiv!(GG, plan, G) # Inverse RFFT
        GG .*= g.n^3 # FFT factor
    end
    # irfft(u.x, g.n) |> Array |> heatmap
    # GG.yx |> Array |> heatmap
    GG
end

grad |> pairs

let
    net_stuff = equivariant_net(setup, [10, 10, 20, 20]);
    (; project, net, ps, st) = net_stuff
    ps = project(ps)
    x = stack(grad)
    x = reshape(x, :, 4, 1) # 4 channels, 1 sample
    net(x, ps, st)
end;



function transform_tensor(t, (p, s))
    t = permutedims(t, p)
    dims = (findall(==(-1), s)...,)
    m = roto_reflection_matrix(p, s)
    t = map(t -> m * t * m', t)
end
