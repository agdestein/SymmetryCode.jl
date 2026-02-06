let
    s = group_stuff(3)
    # map(display, s.mats)
    m = @SMatrix [
        -1 0 0
        0 1 0
        0 0 -1
    ]
    findfirst(isequal(m), s.mats)
end

let
    m = @SMatrix [
        -1 0 0
        0 -1 0
        0 0 -1
    ]
    x = @SMatrix [
        11 12 13
        21 22 23
        31 32 33
    ]
    m * x * m'
end

let
    l = 10.0
    Δ = l / 20
    n = 512
    x = range(-l / 2, l / 2, n + 1)[1:(end - 1)]
    y = @. sqrt(6 / pi / Δ^2) * exp(-6 * x^2 / Δ^2)
    yhat = rfft(y) * l / n
    kmax = div(n, 2)
    k = 0:kmax
    yref = @. exp(-Δ^2 * (2 * pi * k / l)^2 / 24)
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10)
    lines!(ax, 2 * pi / l * k[2:end], abs.(yhat[2:end]); label = "FFT")
    lines!(ax, 2 * pi / l * k[2:end], abs.(yref[2:end]); label = "Theory")
    vlines!(ax, 2 * pi / Δ)
    vlines!(ax, 2 * pi / l)
    ylims!(1.0e-10, 1.0e2)
    save("$(plotdir)/filterkernel.pdf", fig; backend = CairoMakie)
    fig
end

let
    l = 1.0
    n_les = 64
    ncut = 2 * div(n_les, 3)
    Δ = 2 * l / n_les
    k = 2 * pi / l * (1:32)
    w = @. exp(-Δ^2 * k^2 / 24)
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10)
    kol = k .^ (-5 / 3)
    lines!(ax, k, kol)
    lines!(ax, k, w .* kol)
    vlines!(ax, 2 * pi / l * div(n_les, 3))
    fig
end

setup = let
    l = 1.0
    n_les = 256
    Δ = 4 * l / n_les
    (; visc = 5.0e-5, D = 2, l = 1.0, n_dns = 1024, n_les, Δ, backend = CUDABackend())
end

dns = create_dns(setup; t_warmup = 0.5, cfl = 0.35, rng = Xoshiro(0));

let
    g_dns = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    g_les = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    u = dns
    ubar = vectorfield(g_les)
    for (ubar, u) in zip(ubar, u)
        apply!(cutoff!, g_les, (ubar, u))
        apply!(gaussianfilter!, g_les, (ubar, setup.Δ, g_les))
    end
    D = dim(g_dns)
    stat = turbulence_statistics(u, setup.visc, g_dns)
    s_dns = spectrum(u, g_dns)
    s_les = spectrum(ubar, g_les)
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10)
    k = [2, g_dns.n / 8]
    if D == 2
        kolmo = @. 2.0e0 * stat.diss^(2 / 3) * k^(-3)
        escale = stat.diss^(-2 / 3) * stat.l_kol^(-3)
    elseif D == 3
        kolmo = @. 5.0e-1 * stat.diss^(2 / 3) * k^(-5 / 3)
        escale = stat.diss^(-2 / 3) * stat.l_kol^(-5 / 3)
    end
    kscale = stat.l_kol
    lines!(ax, kscale * s_dns.k, escale * s_dns.s)
    lines!(ax, kscale * s_les.k, escale * s_les.s)
    lines!(kscale * k, escale * kolmo)
    # ylims!(1e-7, 1)
    fig
end

data = let
    d = create_data(setup; nstep = 1000, nsubstep = 100, rng = Xoshiro(0), setup.Δ)
    filename = joinpath(outdir, "data-$(setup.n_les).jld2")
    jldsave(filename; data = d)
    load(filename, "data")
end

let
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    plan = plan_rfft(spacescalarfield(g))
    data[2][end].xx |> adapt(setup.backend) |> x -> (plan \ x) * g.n^3 |> median |> display
    error()
end

let
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    u = data[1][1] |> adapt(setup.backend)
    i, b = Spectral.build_tensorbasis(u, g)
    b[:, :, :, :, 4]
end

d = create_dataloader_tbnn(setup, data; batchsize = 5, rng = Xoshiro(0));

d

first(d)[1] |> size
first(d)[1][:, :, 11, 1]

let
    (; D) = setup
    g = Grid{D}(; setup.l, n = setup.n_les, setup.backend)
    T = typeof(g.l)
    (; elements, permutations, signs, mats) = group_stuff(setup.D)
    i = 6
    ip, is = elements[i]
    p, s = permutations[ip], signs[is]
    u = spacevectorfield(g)
    rng = Xoshiro(0)
    foreach(u -> randn!(rng, u), u)
    u_sa = map(SVector{D, T}, u...)
    ru_sa = transform_vector(u_sa, (p, s))
    ru = (; x = getindex.(ru_sa, 1), y = getindex.(ru_sa, 2))
    uhat = map(rfft, u)
    ruhat = map(rfft, ru)
    Gu = getgradient(uhat, g)
    Gru = getgradient(ruhat, g)
    nGu = m_tbnn(Gu)
    nGru = m_tbnn(Gru)
    M = SMatrix{2, 2, T, 4}
    nGu = M.(
        selectdim(nGu, 3, 1),
        selectdim(nGu, 3, 3),
        selectdim(nGu, 3, 3),
        selectdim(nGu, 3, 2),
    )
    nGru = M.(
        selectdim(nGru, 3, 1),
        selectdim(nGru, 3, 3),
        selectdim(nGru, 3, 3),
        selectdim(nGru, 3, 2),
    )
    rnGu = transform_tensor(nGu, (p, s))
    rnGu = smatrix_to_tensorfield(rnGu) |> stack
    nGru = smatrix_to_tensorfield(nGru) |> stack
    norm(nGru - rnGu) / norm(rnGu)
end

let
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    net = Chain(
        Conv((1, 1), Spectral.ninvariant(g) => 10, gelu),
        Conv((1, 1), 10 => 20, gelu),
        Conv((1, 1), 20 => Spectral.nbasis(g); use_bias = false),
    )
    ps, st = Lux.setup(Xoshiro(0), net) |> f64 |> adapt(setup.backend)
    model = tbnn(net, ps, st, g)
    u = data[1][1] |> adapt(setup.backend)
    model(u)
end

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
    τ = data[2][end][1] |> adapt(setup.backend)
    τxx = spacescalarfield(g)
    plan = plan_rfft(τxx)
    apply!(twothirds!, g, (τ, g))
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
    (; elements, permutations, signs) = group_stuff(setup.D)
    i = 6
    ip, is = elements[i]
    p, s = permutations[ip], signs[is]
    u = spacevectorfield(g)
    foreach(randn!, u)
    u = map(rfft, u)
    # u = vectorfield(g)
    # foreach(randn!, u)
    grad = getgradient(u, g)
    x = stack(grad)
    xx = tensorfield_to_smatrix(grad)
    rxx = transform_tensor(xx, (p, s))
    rx = smatrix_to_tensorfield(rxx) |> stack
    nx = net(reshape(x, size(x)..., 1), ps, st) |> first
    nrx = net(reshape(rx, size(rx)..., 1), ps, st) |> first
    nx = cat(
        selectdim(nx, 3, 1),
        selectdim(nx, 3, 3),
        selectdim(nx, 3, 3),
        selectdim(nx, 3, 2);
        dims = 3,
    ) # Desymmetrize
    nrx = cat(
        selectdim(nrx, 3, 1),
        selectdim(nrx, 3, 3),
        selectdim(nrx, 3, 3),
        selectdim(nrx, 3, 2);
        dims = 3,
    ) # Desymmetrize
    nx = nx
    nx = ntuple(i -> view(nx, :, :, i), 4)
    nx = tensorfield_to_smatrix(nx)
    rnx = transform_tensor(nx, (p, s))
    rnx = smatrix_to_tensorfield(rnx) |> stack
    norm(nrx - rnx) / norm(rnx)
end

let
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    (; elements, permutations, signs, mats) = group_stuff(setup.D)
    i = 6
    ip, is = elements[i]
    # mats[i] |> display; error()
    p, s = permutations[ip], signs[is]
    u = spacevectorfield(g)
    foreach(randn!, u)
    # u[1][1:1] .= 1
    uu = vectorfield_to_svector(u)
    ruu = transform_vector(uu, (p, s))
    ru = svector_to_vectorfield(ruu)
    uhat = map(rfft, u)
    ruhat = map(rfft, ru)
    foreach(u -> (u ./= g.n^dim(g)), uhat)
    foreach(u -> (u ./= g.n^dim(g)), ruhat)
    Gu = getgradient(uhat, g)
    Gru = getgradient(ruhat, g)
    Gu = stack(Gu)
    Gru = stack(Gru)
    GGu = ntuple(i -> view(Gu, :, :, i), 4)
    GGu = tensorfield_to_smatrix(GGu)
    rGGu = transform_tensor(GGu, (p, s))
    rGu = smatrix_to_tensorfield(rGGu) |> stack
    norm(rGu - Gru) / norm(Gru)
    # rGu
    # Gru
end

let
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    net_stuff = equivariant_net(setup, [10, 10, 20, 20])
    (; project, net, ps, st) = net_stuff
    ps = project(ps)
    (; elements, permutations, signs) = group_stuff(setup.D)
    i = 6
    ip, is = elements[i]
    p, s = permutations[ip], signs[is]
    u = spacevectorfield(g)
    foreach(randn!, u)
    uu = vectorfield_to_svector(u)
    ruu = transform_vector(uu, (p, s))
    ru = svector_to_vectorfield(ruu)
    uhat = map(rfft, u)
    ruhat = map(rfft, ru)
    foreach(u -> (u ./= g.n^dim(g)), uhat)
    foreach(u -> (u ./= g.n^dim(g)), ruhat)
    G = getgradient(uhat, g)
    rG = getgradient(ruhat, g)
    G = stack(G)
    rG = stack(rG)
    nG = net(reshape(G, :, 4, 1), ps, st) |> first
    nrG = net(reshape(rG, :, 4, 1), ps, st) |> first
    nG = hcat(nG[:, 1, :], nG[:, 3, :], nG[:, 3, :], nG[:, 2, :]) # Desymmetrize
    nrG = hcat(nrG[:, 1, :], nrG[:, 3, :], nrG[:, 3, :], nrG[:, 2, :]) # Desymmetrize
    nGG = reshape(nG, g.n, g.n, 4)
    nGG = ntuple(i -> view(nGG, :, :, i), 4)
    nGG = tensorfield_to_smatrix(nGG)
    rnGG = transform_tensor(nGG, (p, s))
    rnG = smatrix_to_tensorfield(rnGG) |> stack
    rnG = reshape(rnG, :, 4)
    norm(nrG - rnG) / norm(rnG)
end

m_tbnn = let
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    net = Chain(
        Conv((1, 1), Spectral.ninvariant(g) => 64, gelu),
        Conv((1, 1), 64 => 64, gelu),
        Conv((1, 1), 64 => 128, gelu),
        Conv((1, 1), 128 => Spectral.nbasis(g); use_bias = false),
    ) # 13_056 parameters
    net |> display
    ps, st = Lux.setup(Xoshiro(0), net) |> f64 |> adapt(setup.backend)
    for l in ps
        l.weight .*= 0.1
    end
    ps, st = train(;
        loss = create_loss_tbnn(g),
        setup,
        dataloader = create_dataloader_tbnn(setup, data; batchsize = 5, rng = Xoshiro(0)),
        nepoch = 10,
        learning_rate = 1.0e-3,
        net_stuff = (; net, ps, st),
    )
    file = joinpath(outdir, "ps-tbnn-$(setup.n_les).jld2")
    jldsave(file; ps = ps |> cpu_device())
    # ps = load(file, "ps") |> adapt(setup.backend)
    m_tbnn = tbnn(net, ps, st, g)
end;

m_equi = let
    net_stuff = equivariant_net(
        setup,
        [16, 24, 24, 32], # 14112 actual params
    )
    st = net_stuff.st
    ps, st = train(;
        loss = create_loss(net_stuff.project),
        setup,
        dataloader = create_dataloader(setup, data; batchsize = 5),
        nepoch = 10,
        learning_rate = 1.0e-3,
        net_stuff,
    )
    file = joinpath(outdir, "ps_equi.jld2")
    jldsave(file; ps = ps |> cpu_device())
    # ps = load(file, "ps") |> adapt(setup.backend)
    # ps |> cpu_device() |> ComponentArray |> length |> display
    (; net, project) = net_stuff
    fullchain(setup, net, project, ps, st)
end;

m_conv = let
    net_stuff = cnn(
        setup,
        [24, 64, 64, 128]; # 14_587 parameters
        same_as_equi = false,
    )
    for ps in net_stuff.ps
        # Initialize weights are too large
        hasfield(typeof(ps), :weight) && (ps.weight .*= 0.1)
    end
    st = net_stuff.st
    ps, st = train(;
        loss = create_loss(net_stuff.project),
        setup,
        dataloader = create_dataloader(setup, data; batchsize = 5),
        nepoch = 10,
        learning_rate = 1.0e-3,
        net_stuff,
    )
    file = joinpath(outdir, "ps_conv.jld2")
    jldsave(file; ps = ps |> cpu_device())
    # ps = load(file, "ps") |> adapt(setup.backend)
    (; net, project) = net_stuff
    fullchain(setup, net, project, ps, st)
end;

let
    g_dns = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    ustart = randomfield(g_dns; rng = Xoshiro(123), kpeak = 5)
    models = (; tbnn = m_tbnn, conv = m_conv, equi = m_equi)
    u, u_les = inference_post(; ustart, setup, models, setup.Δ, tstop = 1.0e-1)
end

u_dns, u_les = let
    g_dns = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    ustart = randomfield(g_dns; rng = Xoshiro(123), kpeak = 5)
    inference_post(;
        ustart,
        setup,
        models = (; tbnn = m_tbnn, conv = m_conv, equi = m_equi),
        setup.Δ,
        tstop = 1.0e-1,
    )
end;

get_errors(setup, u_les)

let
    g_dns = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    g_les = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    D = dim(g_dns)
    stat = turbulence_statistics(u_dns, setup.visc, g_dns)
    s = spectrum(u_dns, g_dns)
    s_les = map(u -> spectrum(u, g_les), u_les)
    fig = Figure()
    ax = Makie.Axis(fig[1, 1]; xscale = log10, yscale = log10)
    k = [2, g_dns.n / 8]
    if D == 2
        kolmo = @. 2.0e0 * stat.diss^(1 / 3) * k^(-3)
        escale = stat.diss^(-2 / 3) * stat.l_kol^(-3)
    elseif D == 3
        kolmo = @. 5.0e-1 * stat.diss^(2 / 3) * k^(-5 / 3)
        escale = stat.diss^(-2 / 3) * stat.l_kol^(-5 / 3)
    end
    kscale = stat.l_kol
    # lines!(ax, kscale * s.k, escale * s.s)
    # lines!(kscale * k, escale * kolmo)
    for (key, val) in pairs(s_les)
        lines!(ax, kscale * val.k, escale * val.s; label = string(key))
    end
    axislegend(ax)
    # ylims!(1e-7, 1)
    fig
end

let
    g_dns = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    g_les = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    D = dim(g_dns)
    u_ref = u_les.ref
    τhat = sfs(u_dns, g_dns, g_les, setup.Δ)
    τ = spacetensorfield(g_les)
    plan = Seneca.getplan(g_les)
    for (τ, τhat) in zip(τ, τhat)
        apply!(twothirds!, g_les, (τhat, g_les))
        ldiv!(τ, plan, τhat)
        τ .*= g_les.n^dim(g_les)
    end
    G = getgradient(u_ref, g_les)
    S = (; G.xx, G.yy, xy = (G.xy .+ G.yx) ./ 2)
    models = (; tbnn = m_tbnn, conv = m_conv, equi = m_equi)
    labels = (; ref = "Reference", tbnn = "TBNN", conv = "Conv", equi = "G-Conv")
    τ_les = map(models) do m
        y = m(G)
        if D == 2
            xx, yy, xy = 1, 2, 3
            (; xx = view(y, :, :, xx), yy = view(y, :, :, yy), xy = view(y, :, :, xy))
        elseif D == 3
            error()
        end
    end
    τ_all = (; ref = τ, τ_les...)
    τxx = map(τ -> τ.xx, τ_all)
    τyy = map(τ -> τ.yy, τ_all)
    τxy = map(τ -> τ.xy, τ_all)
    diss = map(τ_all) do τ
        @. τ.xx * S.xx + τ.yy * S.yy + 2 * τ.xy * S.xy
    end
    # τ.xx |> x -> plan \ x |> display; error()
    # τ.xx |> display; error()
    # map(mean, τxx) |> pairs |> display; error()
    # τ.xx |> Array |> heatmap |> display; error()
    # τ.xx |> Array |> display
    # error()
    fig = Figure(; size = (700, 300))
    #
    ax_xx =
        Makie.Axis(fig[1, 1]; xlabel = "xx-component", ylabel = "Density", yscale = log10)
    τxx = map(d -> kde(d |> vec |> Array) |> x -> (; x.x, x.density), τxx)
    for (key, val) in pairs(τxx)
        lines!(ax_xx, val.x / setup.Δ, val.density; label = labels[key])
    end
    ylims!(ax_xx, 3.0e-2, 2.0e2)
    # xlims!(ax_xx, -0.1, 0.4)
    #
    ax_xy = Makie.Axis(fig[1, 2]; xlabel = "xy-component", yscale = log10)
    τxy = map(d -> kde(d |> vec |> Array) |> x -> (; x.x, x.density), τxy)
    for (key, val) in pairs(τxy)
        lines!(ax_xy, val.x / setup.Δ, val.density)
    end
    ylims!(ax_xy, 5.0e-2, 1.0e2)
    # xlims!(ax_xy, -0.2, 0.12)
    #
    ax_diss = Makie.Axis(fig[1, 3]; xlabel = "Dissipation", yscale = log10)
    diss = map(d -> kde(d |> vec |> Array) |> x -> (; x.x, x.density), diss)
    for (key, val) in pairs(diss)
        lines!(
            ax_diss,
            # val.x,
            val.x / setup.Δ,
            val.density,
        )
    end
    ylims!(ax_diss, 1.0e-1, 7.0e1)
    xlims!(ax_diss, -20, 20)
    Legend(fig[0, :], ax_xx; orientation = :horizontal)
    fig
end

let
    g = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    u = u_dns
    s = turbulence_statistics(u, setup.visc, g)
    s |> pairs
end

let
    u = u_dns
    g = Grid{setup.D}(; setup.l, n = setup.n_dns, setup.backend)
    k = div(g.n, 2)
    plan = plan_rfft(spacescalarfield(g))
    plan \ copy(u.y) |> Array |> heatmap
end

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
    ylims!(ax, 1.0e-2, 1.0e3)
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
