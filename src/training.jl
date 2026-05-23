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

function train(;
        loss, setup, trainloader, valloader, net_stuff,
        checkpointfile = nothing, resume = false,
    )
    (; backend, train_setup) = setup
    (;
        nepoch, learning_rate, log_every, grad_clip, patience,
        warmup_frac, checkpoint_every,
    ) = train_setup
    (; net, ps, st) = net_stuff
    device = adapt(backend)
    opt = OptimiserChain(ClipNorm(grad_clip), AdamW(learning_rate))

    # Warmup: compile the train step on a throwaway state so the timed loop
    # is measured accurately and starts from the initial parameters.
    let ts = Training.TrainState(net, deepcopy(ps), deepcopy(st), opt)
        x, y = first(trainloader) |> device
        Training.single_train_step!(AutoZygote(), loss, (x, y), ts)
    end

    train_state = Training.TrainState(net, deepcopy(ps), st, opt)
    b_valid = first(valloader) |> device
    ps_best = deepcopy(train_state.parameters)
    st_best = deepcopy(train_state.states)
    l_best = Inf
    losses_train = Float64[]
    losses_valid = Float64[]
    start_epoch = 1

    if resume && !isnothing(checkpointfile) && isfile(checkpointfile)
        @info "Resuming training from $(checkpointfile)"
        flush(stderr)
        ck = load(checkpointfile)
        train_state =
            Training.TrainState(net, ck["ps_cur"] |> device, ck["st_cur"], opt)
        ps_best = ck["ps_best"] |> device
        st_best = ck["st_best"]
        l_best = ck["l_best"]
        losses_train = ck["losses_train"]
        losses_valid = ck["losses_valid"]
        start_epoch = ck["epoch"] + 1
    end

    nbatch = length(trainloader)
    total = nepoch * nbatch
    nwarm = max(1, round(Int, warmup_frac * total))
    # Linear warmup then cosine decay to zero.
    schedule(s) =
        s < nwarm ? learning_rate * s / nwarm :
        learning_rate * (1 + cos(π * (s - nwarm) / max(1, total - nwarm))) / 2
    step = (start_epoch - 1) * nbatch
    epochs_since_improve = 0

    timing = time()
    for iepoch in start_epoch:nepoch
        l_best_epoch_start = l_best
        for (ibatch, batch) in enumerate(trainloader)
            step += 1
            Optimisers.adjust!(train_state.optimizer_state, schedule(step))
            x, y = batch |> device
            _, l_train, _, train_state =
                Training.single_train_step!(AutoZygote(), loss, (x, y), train_state)

            # Validate on the held-out set against the current parameters.
            l_valid =
                loss(net, train_state.parameters, train_state.states, b_valid) |>
                first
            push!(losses_train, l_train)
            push!(losses_valid, l_valid)
            if l_valid < l_best
                l_best = l_valid
                ps_best = deepcopy(train_state.parameters)
                st_best = deepcopy(train_state.states)
            end

            if ibatch % log_every == 0
                @info join(
                    [
                        "epoch $(iepoch)/$(nepoch)",
                        "batch $(ibatch)/$(nbatch)",
                        "lr = $(round(schedule(step); sigdigits = 3))",
                        "loss (valid) = $(round(l_valid; sigdigits = 4))",
                        "loss (train) = $(round(l_train; sigdigits = 4))",
                    ],
                    ",\t",
                )
                flush(stderr)
            end

            if !isnothing(checkpointfile) && step % checkpoint_every == 0
                jldsave(
                    checkpointfile;
                    epoch = iepoch - 1, # resume re-runs at most this epoch
                    ps_cur = train_state.parameters |> cpu_device(),
                    st_cur = train_state.states,
                    ps_best = ps_best |> cpu_device(),
                    st_best,
                    l_best,
                    losses_train,
                    losses_valid,
                )
            end
        end

        # Early stopping on the held-out validation loss.
        epochs_since_improve =
            l_best < l_best_epoch_start ? 0 : epochs_since_improve + 1
        if !isnothing(checkpointfile)
            jldsave(
                checkpointfile;
                epoch = iepoch,
                ps_cur = train_state.parameters |> cpu_device(),
                st_cur = train_state.states,
                ps_best = ps_best |> cpu_device(),
                st_best,
                l_best,
                losses_train,
                losses_valid,
            )
        end
        if epochs_since_improve >= patience
            @info "Early stopping: no validation improvement in $(patience) epochs"
            flush(stderr)
            break
        end
    end
    timing = time() - timing

    return (; ps = ps_best, st = st_best, losses_train, losses_valid, timing)
end

"""
Generic training orchestrator shared by all learned closures, so the three
models differ only in architecture/loss, not in training treatment.

`mode`:
- `:scratch` — train from the initial parameters (ignore any checkpoint)
- `:resume`  — continue from `ps-<key>-checkpoint.jld2` if present, else
               train from scratch; periodic checkpoints make it SLURM-safe
- `:skip`    — do not train; just load the persisted parameters

A `Bool` is accepted for backwards compatibility: `true → :resume`,
`false → :skip`.
"""
function create_model(setup, mode; key, buildnet, makeloss, makeloaders, wrap)
    mode = mode isa Bool ? (mode ? :resume : :skip) : mode
    (; outdir) = setup
    file = joinpath(outdir, "ps-$(key).jld2")
    checkpointfile = joinpath(outdir, "ps-$(key)-checkpoint.jld2")
    net_stuff = buildnet()
    g = Grid{setup.D}(; setup.l, n = setup.n_les, setup.backend)

    if mode != :skip
        @info "Training $(key)"
        flush(stderr)
        data = joinpath(outdir, "data.jld2") |> load_object
        trainloader, valloader = makeloaders(data)
        result = train(;
            loss = makeloss(net_stuff, g),
            setup,
            trainloader,
            valloader,
            net_stuff,
            checkpointfile,
            resume = mode == :resume,
        )
        save_object(
            file,
            (;
                ps = result.ps |> cpu_device(),
                st = result.st,
                result.losses_train,
                result.losses_valid,
                result.timing,
            ),
        )
        isfile(checkpointfile) && rm(checkpointfile)
    end

    d = load_object(file)
    ps = d.ps |> adapt(setup.backend)
    chain = wrap(net_stuff, ps, d.st, g)
    return chain, (; d.losses_train, d.losses_valid, d.timing)
end

function create_tbnn(setup, mode = :resume)
    precision = setup.train_setup.precision
    return create_model(
        setup, mode;
        key = :tbnn,
        buildnet = function ()
            net = tbnn_net(setup, setup.tbnn_setup.layers)
            net |> display
            flush(stdout)
            f = precision === Float32 ? f32 : f64
            ps, st =
                Lux.setup(Xoshiro(setup.train_setup.seed), net) |> f |>
                adapt(setup.backend)
            return (; net, ps, st, project = identity)
        end,
        makeloss = (ns, g) -> create_loss_tbnn(g),
        makeloaders = data -> create_dataloader_tbnn(
            setup, data;
            setup.train_setup.nsample,
            setup.train_setup.batchsize,
            rng = Xoshiro(setup.train_setup.seed),
        ),
        wrap = (ns, ps, st, g) -> tbnn(ns.net, ps, st, setup.Δ, g, precision),
    )
end

function create_equi(setup, mode = :resume)
    return create_model(
        setup, mode;
        key = :equi,
        buildnet = () -> equivariant_net(setup, setup.equi_setup.layers),
        makeloss = (ns, g) -> create_loss(ns.project),
        makeloaders = data -> create_dataloader(
            setup, data;
            setup.train_setup.nsample,
            setup.train_setup.batchsize,
        ),
        wrap = (ns, ps, st, g) ->
        fullchain(setup, ns.net, ns.project, ps, st, setup.Δ),
    )
end

function create_conv(setup, mode = :resume)
    return create_model(
        setup, mode;
        key = :conv,
        buildnet = () ->
        cnn(setup, setup.conv_setup.layers; setup.conv_setup.same_as_equi),
        makeloss = (ns, g) -> create_loss(ns.project),
        makeloaders = data -> create_dataloader(
            setup, data;
            setup.train_setup.nsample,
            setup.train_setup.batchsize,
        ),
        wrap = (ns, ps, st, g) ->
        fullchain(setup, ns.net, ns.project, ps, st, setup.Δ),
    )
end
