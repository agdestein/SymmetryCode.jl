# Generic training loop and per-model orchestration (TBNN, G-Conv, Conv).

create_loss(project) = function loss(net, ps, st, (x, y))
    ps = project(ps)
    yhat = net(x, ps, st) |> first
    # l = MSELoss()(yhat, y) # (x, y) pair is already normalized
    l = sum(abs2, yhat - y) / sum(abs2, y)
    return l, st, (;)
end

create_loss_tbnn(g) = function loss(net, ps, st, (x, y))
    D = dim(g)
    nx = size(x)[1:D]
    nt = tensordim(g)
    ni = ninvariant(g)
    nb = nbasis(g)

    # Destructure invariants and basis
    i = selectdim(x, D + 1, 1:ni)
    b = selectdim(x, D + 1, (ni + 1):size(x, D + 1))

    # Compute coefficients
    w = net(i, ps, st) |> first

    # Basis contraction
    w = reshape(w, nx..., 1, nb, :)
    b = reshape(b, nx..., nt, nb, :)
    wb = @. w * b
    m = sum(wb; dims = D + 2)
    m = reshape(m, nx..., nt, :)

    # l = MSELoss()(m, y)
    l = sum(abs2, m - y) / sum(abs2, y)
    return l, st, (;)
end

function train(; loss, setup, dataloader, nepoch, learning_rate, net_stuff)
    (; backend) = setup
    (; net, ps, st) = net_stuff
    ps = deepcopy(ps)
    device = adapt(backend)
    opt = AdamW(learning_rate)
    train_state = Training.TrainState(net, ps, st, opt)
    b_valid = first(dataloader) |> device
    ps_best = deepcopy(ps)
    l_best = Inf
    losses_train = zeros(0)
    losses_valid = zeros(0)

    # Warmup step
    x, y = first(dataloader) |> device
    _, l_train, _, train_state =
        Training.single_train_step!(AutoZygote(), loss, (x, y), train_state)
    l_valid = loss(net, ps, st, b_valid) |> first

    timing = time()
    i = 0
    for iepoch in 1:nepoch, (ibatch, batch) in enumerate(dataloader)
        i += 1
        x, y = batch |> device
        # loss(net, ps, st, (x, y)); error()
        _, l_train, _, train_state =
            Training.single_train_step!(AutoZygote(), loss, (x, y), train_state)
        if ibatch % 1 == 0
            # Check performance on validation batch
            l_valid = loss(net, ps, st, b_valid) |> first

            # Log
            @info join(
                [
                    "iepoch = $iepoch",
                    "ibatch = $ibatch",
                    "loss (valid) = $(round(l_valid; sigdigits = 4))",
                    "loss (train) = $(round(l_train; sigdigits = 4))",
                ],
                ",\t",
            )
            flush(stderr)

            push!(losses_train, l_train)
            push!(losses_valid, l_valid)

            # Keep current best parameters
            if l_valid < l_best
                l_best = l_valid
                ps_best = deepcopy(train_state.parameters)
            end
        end
    end
    timing = time() - timing

    ps = ps_best # Retain best (not last) parameters
    st = train_state.states # Note: If st is non-empty, need to make "best"-mechanism for states
    return (; ps, st, losses_train, losses_valid, timing)
end

function create_tbnn(setup, dotrain)
    (; tbnn_setup) = setup
    data = joinpath(setup.outdir, "data.jld2") |> load_object
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    kern = ntuple(Returns(1), setup.D)
    # TODO: Extract this from `tbnn_setup`
    # net = Chain(
    #     Conv(kern, ninvariant(g) => 64, gelu),
    #     Conv(kern, 64 => 64, gelu),
    #     Conv(kern, 64 => 128, gelu),
    #     Conv(kern, 128 => nbasis(g); use_bias = false),
    # ) # 13_888 parameters
    net = Chain(
        Conv(kern, ninvariant(g) => 16, gelu),
        Conv(kern, 16 => 32, gelu),
        Conv(kern, 32 => 64, gelu),
        Conv(kern, 64 => nbasis(g); use_bias = false),
    ) # 3_200 parameters
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
            learning_rate = 1.0e-3,
            net_stuff = (; net, ps, st),
        )
        ps = ps |> cpu_device()
        jldsave(file; ps, losses_train, losses_valid, timing)
    end
    ps, losses_train, losses_valid, timing =
        load(file, "ps", "losses_train", "losses_valid", "timing")
    ps = ps |> adapt(setup.backend)
    chain = tbnn(net, ps, st, setup.Δ, g)
    return chain, (; losses_train, losses_valid, timing)
end

function create_equi(setup, dotrain)
    (; equi_setup) = setup
    data = joinpath(setup.outdir, "data.jld2") |> load_object
    net_stuff = equivariant_net(setup, equi_setup.layers)
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
            learning_rate = 1.0e-3,
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
    return chain, (; losses_train, losses_valid, timing)
end

function create_conv(setup, dotrain)
    (; conv_setup) = setup
    data = joinpath(setup.outdir, "data.jld2") |> load_object
    net_stuff = cnn(setup, conv_setup.layers; conv_setup.same_as_equi)
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
            learning_rate = 1.0e-3,
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
    return chain, (; losses_train, losses_valid, timing)
end
