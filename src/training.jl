function create_tbnn(setup, data, dotrain)
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    kern = ntuple(Returns(1), setup.D)
    net = Chain(
        Conv(kern, ninvariant(g) => 64, gelu),
        Conv(kern, 64 => 64, gelu),
        Conv(kern, 64 => 128, gelu),
        Conv(kern, 128 => nbasis(g); use_bias = false),
    ) # 13_888 parameters
    # net = Chain(
    #     Conv(kern, Spectral.ninvariant(g) => 16, gelu),
    #     Conv(kern, 16 => 32, gelu),
    #     Conv(kern, 32 => 64, gelu),
    #     Conv(kern, 64 => Spectral.nbasis(g); use_bias = false),
    # ) # 3_200 parameters
    net |> display
    flush(stdout)
    ps, st = Lux.setup(Xoshiro(0), net) |> f64 |> adapt(setup.backend)
    for l in ps
        l.weight .*= 0.1
    end
    file = joinpath(setup.outdir, "ps-tbnn.jld2")
    if dotrain
        @info "Training TBNN"
        flush(stderr)
        (; ps, st, losses_train, losses_valid, timing) = train(;
            loss = create_loss_tbnn(g),
            setup,
            dataloader = create_dataloader_tbnn(
                setup,
                data;
                nsample = 50, # Don't use all the snapshots
                batchsize = 20,
                rng = Xoshiro(0),
            ),
            nepoch = 5,
            learning_rate = 1e-3,
            net_stuff = (; net, ps, st),
        )
        ps = ps |> cpu_device()
        jldsave(file; ps, losses_train, losses_valid, timing)
    end
    ps, losses_train, losses_valid, timing =
        load(file, "ps", "losses_train", "losses_valid", "timing")
    ps = ps |> adapt(setup.backend)
    chain = tbnn(net, ps, st, setup.Δ, g)
    chain, (; losses_train, losses_valid, timing)
end

function create_equi(setup, data, dotrain)
    net_stuff = equivariant_net(
        setup,
        # [12, 16, 16, 24], # 40_328 actual params
        [8, 8, 8, 16], # 12_544 actual params
        # [4, 4, 4, 8], # 3_200 actual params
    )
    st = net_stuff.st
    file = joinpath(setup.outdir, "ps-equi.jld2")
    if dotrain
        @info "Training G-conv"
        flush(stderr)
        (; ps, st, losses_train, losses_valid, timing) = train(;
            loss = create_loss(net_stuff.project),
            setup,
            dataloader = create_dataloader(setup, data; nsample = 50, batchsize = 20),
            nepoch = 5,
            learning_rate = 1e-3,
            net_stuff,
        )
        ps = ps |> cpu_device()
        jldsave(file; ps, losses_train, losses_valid, timing)
    end
    ps, losses_train, losses_valid, timing =
        load(file, "ps", "losses_train", "losses_valid", "timing")
    # ps = net_stuff.ps
    ps = ps |> adapt(setup.backend)
    ps |> cpu_device() |> ComponentArray |> length |> display
    flush(stdout)
    (; net, project) = net_stuff
    chain = fullchain(setup, net, project, ps, st, setup.Δ)
    chain, (; losses_train, losses_valid, timing)
end

function create_conv(setup, data, dotrain)
    net_stuff = cnn(
        setup,
        # [48, 128, 128, 128]; # 40_550 parameters
        [48, 64, 64, 64]; # 12_320 parameters
        # [16, 32, 64]; # 3_200 parameters
        same_as_equi = false,
    )
    for ps in net_stuff.ps
        # Initialize weights are too large
        hasfield(typeof(ps), :weight) && (ps.weight .*= 0.1)
    end
    st = net_stuff.st
    file = joinpath(setup.outdir, "ps-conv.jld2")
    if dotrain
        @info "Training Conv"
        flush(stderr)
        (; ps, st, losses_train, losses_valid, timing) = train(;
            loss = create_loss(net_stuff.project),
            setup,
            dataloader = create_dataloader(setup, data; nsample = 50, batchsize = 20),
            nepoch = 5,
            learning_rate = 1e-3,
            net_stuff,
        )
        ps = ps |> cpu_device()
        jldsave(file; ps, losses_train, losses_valid, timing)
    end
    # ps = net_stuff.ps
    ps, losses_train, losses_valid, timing =
        load(file, "ps", "losses_train", "losses_valid", "timing")
    ps = ps |> adapt(setup.backend)
    ps |> cpu_device() |> ComponentArray |> length |> display
    flush(stdout)
    (; net, project) = net_stuff
    chain = fullchain(setup, net, project, ps, st, setup.Δ)
    chain, (; losses_train, losses_valid, timing)
end

function plot_training(setup, train_tbnn, train_equi, train_conv)
    fig = Figure(; size = (400, 340))
    ax = Axis(
        fig[1, 1];
        # xscale = log10,
        # yscale = log10,
        xlabel = "Iteration",
        ylabel = "Loss",
    )
    lines!(ax, train_tbnn.losses_valid; label = "TBNN")
    lines!(ax, train_equi.losses_valid; label = "G-Conv")
    lines!(ax, train_conv.losses_valid; label = "Conv")
    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        tellheight = true,
        framevisible = false,
        horizontal = true,
        nbanks = 3,
    )
    eps = 0.1
    ylims!(ax, -eps, 1 + eps)
    rowgap!(fig.layout, 5)
    save("$(setup.plotdir)/training.pdf", fig; backend = CairoMakie)
    fig
end
