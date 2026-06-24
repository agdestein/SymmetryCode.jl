# Generic training loop and coordinate-driven per-model orchestration.

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
    nb = nbasis(g)

    # Destructure invariants and basis. Split from the *end*: the basis is the
    # last nt·nb channels, so the invariant block absorbs the optional Re_Δ row.
    ninv = size(x, D + 1) - nt * nb
    i = selectdim(x, D + 1, 1:ninv)
    b = selectdim(x, D + 1, (ninv + 1):size(x, D + 1))

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

"""
Fixed-budget training: linear-warmup → cosine LR decay to the end, returning the
*final* parameters. The nets are tiny and train fast, so there is no early
stopping, patience, or file checkpointing; the validation loss is tracked only
for the convergence curve (`case.schedule.val_fraction`), never for model
selection.
"""
function train(; loss, case, trainloader, valloader, net_stuff)
    (; backend, schedule) = case
    (; nepoch, learning_rate, log_every, grad_clip, warmup_frac) = schedule
    (; net, ps, st) = net_stuff
    device = adapt(backend)
    opt = OptimiserChain(ClipNorm(grad_clip), AdamW(learning_rate))

    # Compile the train step on a throwaway state so the timed loop starts from
    # the initial parameters.
    let ts = Training.TrainState(net, deepcopy(ps), deepcopy(st), opt)
        x, y = first(trainloader) |> device
        Training.single_train_step!(AutoZygote(), loss, (x, y), ts)
    end

    train_state = Training.TrainState(net, deepcopy(ps), st, opt)
    b_valid = first(valloader) |> device
    losses_train = Float64[]
    losses_valid = Float64[]

    nbatch = length(trainloader)
    total = nepoch * nbatch
    nwarm = max(1, round(Int, warmup_frac * total))
    # Linear warmup then cosine decay to zero.
    schedfn(s) =
        s < nwarm ? learning_rate * s / nwarm :
        learning_rate * (1 + cos(π * (s - nwarm) / max(1, total - nwarm))) / 2

    step = 0
    timing = time()
    for iepoch in 1:nepoch
        for (ibatch, batch) in enumerate(trainloader)
            step += 1
            Optimisers.adjust!(train_state.optimizer_state, schedfn(step))
            x, y = batch |> device
            _, l_train, _, train_state =
                Training.single_train_step!(AutoZygote(), loss, (x, y), train_state)

            # Validation loss for the convergence curve (not for selection).
            l_valid =
                loss(net, train_state.parameters, train_state.states, b_valid) |> first
            push!(losses_train, l_train)
            push!(losses_valid, l_valid)

            if ibatch % log_every == 0
                @info join(
                    [
                        "epoch $(iepoch)/$(nepoch)",
                        "batch $(ibatch)/$(nbatch)",
                        "lr = $(round(schedfn(step); sigdigits = 3))",
                        "loss (valid) = $(round(l_valid; sigdigits = 4))",
                        "loss (train) = $(round(l_train; sigdigits = 4))",
                    ],
                    ",\t",
                )
                flush(stderr)
            end
        end
    end
    timing = time() - timing

    return (;
        ps = train_state.parameters, st = train_state.states,
        losses_train, losses_valid, timing,
    )
end

# --- coordinate-driven model orchestration ---

"""
Lightweight `setup`-shaped view carrying just what the `nets.jl` builders read —
the grid plus the network-init `seed` and `precision`. Lets those builders stay
unchanged while the *network* seed becomes a model coordinate (`netseed`),
distinct from the DNS seed.
"""
make_netsetup(case, netseed) = (;
    case.D, case.l, case.n_les, case.backend,
    train_setup = (; seed = netseed, precision = case.schedule.precision),
)

"""
Untrained net + initial parameters/states for a learned closure `arch`
(`:tbnn`/`:equi`/`:conv`) at hidden widths `layers`. `netsetup` carries the grid
+ init seed/precision (see [`make_netsetup`](@ref)). For `:equi` the `ps` is the
compact pre-synthesis basis and `project` is the synthesis operator; for
`:tbnn`/`:conv` `project = identity`.
"""
function build_net_stuff(netsetup, arch, layers; same_as_equi = false, use_redelta = false)
    if arch === :tbnn
        net = tbnn_net(netsetup, layers; use_redelta)
        net |> display
        flush(stdout)
        f = netsetup.train_setup.precision === Float32 ? f32 : f64
        ps, st =
            Lux.setup(Xoshiro(netsetup.train_setup.seed), net) |> f |> adapt(netsetup.backend)
        return (; net, ps, st, project = identity)
    elseif arch === :equi
        return equivariant_net(netsetup, layers; use_redelta)
    elseif arch === :conv
        return mlp(netsetup, layers; same_as_equi, use_redelta)
    else
        error("not a learned closure: $(arch)")
    end
end

"""
Inference closure `(u, ∇ū) -> τ` for a trained learned-model coordinate `m`,
wrapping `psfile(case, m)` for evaluation at `setup` (which supplies the filter
width Δ). (Re_Δ-augmented inference is wired in a later increment.)
"""
function build_model(case, m, setup)
    layers = case.tiers[m.tier][m.arch]
    ns = build_net_stuff(make_netsetup(case, m.netseed), m.arch, layers; m.use_redelta)
    d = load(psfile(case, m))
    ps = d["ps"] |> f64 |> adapt(case.backend)
    st = d["st"]
    redelta_norm = d["redelta_norm"]
    g = Grid{case.D}(; case.l, n = case.n_les, case.backend)
    return m.arch === :tbnn ?
        tbnn(ns.net, ps, st, setup.Δ, g; m.use_redelta, redelta_norm, setup.visc) :
        fullchain(setup, ns.net, ns.project, ps, st, setup.Δ; m.use_redelta, redelta_norm)
end

"""
Default training pool: every `(dns, Δ_factor)` from the training DNS runs crossed
with the training filter ratios. Pass an explicit list to subset it.
"""
build_trainpool(case; dns = dns_runs().train, filters = case.filters_train) =
    [(d, Δf) for d in dns for Δf in filters]

"Standardization `(; μ, σ)` of `log Re_Δ` over the trainpool — stored with `ps`."
function compute_redelta_norm(case, trainpool)
    logre = Float64[]
    for (dns, Δf) in trainpool
        append!(logre, log.(load(fieldsfile(case, dns, Δf), "redelta")))
    end
    return (; μ = mean(logre), σ = std(logre))
end

make_loaders(case, m, trainpool, redelta_norm) = let rng = Xoshiro(m.netseed)
    loader = m.arch === :tbnn ? create_dataloader_tbnn : create_dataloader
    loader(case, trainpool; case.schedule.batchsize, rng, m.use_redelta, redelta_norm)
end

"""
Train one learned-model coordinate `m = (; arch, tier, netseed, use_redelta)` on
`trainpool`, persisting `ps`, states, loss curves, timing, and the Re_Δ
standardization to `psfile(case, m)`. Fixed-budget (see [`train`](@ref)); skips
when the file already exists unless `force`.
"""
function train_model(case, m, trainpool; force = false)
    file = psfile(case, m)
    if !force && isfile(file)
        @info "skip (exists): $(modelkey(m))"
        flush(stderr)
        return
    end
    g = Grid{case.D}(; case.l, n = case.n_les, case.backend)
    layers = case.tiers[m.tier][m.arch]
    net_stuff = build_net_stuff(make_netsetup(case, m.netseed), m.arch, layers; m.use_redelta)
    redelta_norm = m.use_redelta ? compute_redelta_norm(case, trainpool) : nothing
    trainloader, valloader = make_loaders(case, m, trainpool, redelta_norm)
    loss = m.arch === :tbnn ? create_loss_tbnn(g) : create_loss(net_stuff.project)

    @info "Training $(modelkey(m)) on $(length(trainpool)) datasets"
    flush(stderr)
    result = train(; loss, case, trainloader, valloader, net_stuff)
    jldsave_atomic(
        file;
        ps = result.ps |> cpu_device(),
        st = result.st,
        result.losses_train,
        result.losses_valid,
        result.timing,
        redelta_norm,
    )
    return
end

"""
Train the Cartesian product of model coordinates (`archs × tiers × netseeds ×
use_redelta`) on `trainpool`, one at a time with `clean()` between them so each
model's GPU working set is reclaimed before the next.
"""
function train_models(
        case, trainpool = build_trainpool(case);
        archs = (:tbnn, :equi, :conv),
        tiers = (:saturated,),
        netseeds = 0:0,
        use_redelta = (false,),
        force = false,
    )
    clean()
    for arch in archs, tier in tiers, netseed in netseeds, ur in use_redelta
        train_model(case, (; arch, tier, netseed, use_redelta = ur), trainpool; force)
        clean()
    end
    return
end
