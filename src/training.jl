# Generic training loop and per-model orchestration (TBNN, G-Conv, Conv).

create_loss(project) = function loss(net, ps, st, (x, y))
    ps = project(ps)
    yhat = net(x, ps, st) |> first
    # l = MSELoss()(yhat, y) # (x, y) pair is already normalized
    l = sum(abs2, yhat - y) / (sum(abs2, y) + eps(eltype(y)))
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
    l = sum(abs2, m - y) / (sum(abs2, y) + eps(eltype(y)))
    return l, st, (;)
end

function train(; loss, setup, trainloader, valloader, nepoch, learning_rate, net_stuff)
    (; backend) = setup
    (; net, ps, st) = net_stuff
    device = adapt(backend)
    ps_init = deepcopy(ps)

    # Warmup: compile the train step on a throwaway state so the timed
    # loop is measured accurately *and* starts from the initial parameters
    # (the previous code left this warmup update in the real state).
    let ts = Training.TrainState(
            net, deepcopy(ps_init), deepcopy(st), AdamW(learning_rate),
        )
        x, y = first(trainloader) |> device
        Training.single_train_step!(AutoZygote(), loss, (x, y), ts)
    end

    opt = AdamW(learning_rate)
    train_state = Training.TrainState(net, deepcopy(ps_init), st, opt)
    b_valid = first(valloader) |> device
    ps_best = deepcopy(train_state.parameters)
    st_best = deepcopy(train_state.states)
    l_best = Inf
    losses_train = zeros(0)
    losses_valid = zeros(0)

    timing = time()
    for iepoch in 1:nepoch, (ibatch, batch) in enumerate(trainloader)
        x, y = batch |> device
        _, l_train, _, train_state =
            Training.single_train_step!(AutoZygote(), loss, (x, y), train_state)

        # Validate against the *current trained* parameters, not a stale
        # closure-local `ps` (which never tracked the optimizer updates).
        l_valid =
            loss(net, train_state.parameters, train_state.states, b_valid) |> first

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

        # Keep the best-validation parameters together with their states
        if l_valid < l_best
            l_best = l_valid
            ps_best = deepcopy(train_state.parameters)
            st_best = deepcopy(train_state.states)
        end
    end
    timing = time() - timing

    ps = ps_best # Retain best (not last) parameters
    st = st_best  # ... and the states captured at the same time
    return (; ps, st, losses_train, losses_valid, timing)
end

function create_tbnn(setup, dotrain)
    (; tbnn_setup) = setup
    data = joinpath(setup.outdir, "data.jld2") |> load_object
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)
    net = tbnn_net(setup, tbnn_setup.layers)
    net |> display
    flush(stdout)
    ps, st = Lux.setup(Xoshiro(0), net) |> f64 |> adapt(setup.backend)
    file = joinpath(setup.outdir, "ps-tbnn.jld2")
    if dotrain
        @info "Training TBNN"
        flush(stderr)
        trainloader, valloader = create_dataloader_tbnn(
            setup,
            data;
            nsample = 50, # Don't use all the snapshots
            batchsize = 20,
            rng = Xoshiro(0),
        )
        (; ps, st, losses_train, losses_valid, timing) = train(;
            loss = create_loss_tbnn(g),
            setup,
            trainloader,
            valloader,
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
        trainloader, valloader =
            create_dataloader(setup, data; nsample = 50, batchsize = 20)
        (; ps, st, losses_train, losses_valid, timing) = train(;
            loss = create_loss(net_stuff.project),
            setup,
            trainloader,
            valloader,
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
    st = net_stuff.st
    file = joinpath(setup.outdir, "ps-conv.jld2")
    if dotrain
        @info "Training Conv"
        flush(stderr)
        trainloader, valloader =
            create_dataloader(setup, data; nsample = 50, batchsize = 20)
        (; ps, st, losses_train, losses_valid, timing) = train(;
            loss = create_loss(net_stuff.project),
            setup,
            trainloader,
            valloader,
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
