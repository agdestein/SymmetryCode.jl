function equivariant_net(setup, nchan)
    (; D, backend) = setup
    dev = adapt(backend)
    # dev = identity
    rng = Xoshiro(0)
    T, f = Float64, f64
    nten = D^2
    (; elements) = group_stuff(D)
    (; r_lift, r_sink, r_mid) = get_weight_projectors(D)
    nreg = length(elements)
    e_lift = eigen(r_lift / nreg; sortby = -).vectors[:, 1:nten]
    e_mid = eigen(r_mid / nreg; sortby = -).vectors[:, 1:nreg]
    e_sink = eigen(r_sink / nreg; sortby = -).vectors[:, 1:nten]
    proj_lift = Dense(nten => nten * nreg; use_bias = false)
    proj_sink = Dense(nten => nreg * nten; use_bias = false)
    proj_mid = Dense(nreg => nreg^2; use_bias = false)
    proj_lift_ps, _ = Lux.setup(rng, proj_lift) |> f |> dev
    copyto!(proj_lift_ps.weight, e_lift)
    proj_sink_ps, _ = Lux.setup(rng, proj_sink) |> f |> dev
    copyto!(proj_sink_ps.weight, e_sink)
    proj_mid_ps, _ = Lux.setup(rng, proj_mid) |> f |> dev
    copyto!(proj_mid_ps.weight, e_mid)
    kern = ntuple(Returns(1), D)
    function project_lift(ps)
        w, b = ps.weight, ps.bias
        _, c_out = size(w)
        w = proj_lift(w, proj_lift_ps, (;)) |> first
        w = reshape(w, nreg, nten, c_out)
        w = permutedims(w, (2, 1, 3))
        weight = reshape(w, kern..., nten, nreg * c_out)
        bias = reshape(repeat(reshape(b, 1, :), nreg), :)
        return (; weight, bias)
    end
    function project_mid(ps)
        w, b = ps.weight, ps.bias
        _, c_out, c_in = size(w)
        w = reshape(w, nreg, :)
        w = proj_mid(w, proj_mid_ps, (;)) |> first
        w = reshape(w, nreg, nreg, c_out, c_in)
        w = permutedims(w, (2, 4, 1, 3))
        weight = reshape(w, kern..., nreg * c_in, nreg * c_out)
        bias = reshape(repeat(reshape(b, 1, :), nreg), :)
        return (; weight, bias)
    end
    function project_sink(ps)
        w = ps.weight
        _, c_in = size(w)
        w = proj_sink(w, proj_sink_ps, (;)) |> first
        w = reshape(w, nten, nreg, c_in)
        w = permutedims(w, (2, 3, 1))
        weight = reshape(w, kern..., nreg * c_in, nten)
        return (; weight)
    end
    function project(ps)
        lift, mids..., sink, symm = ps
        return (;
            lift = project_lift(lift),
            map(project_mid, mids)...,
            sink = project_sink(sink),
            symm,
        )
    end
    net = Chain(;
        lift = Conv(kern, nten => nreg * nchan[1], gelu),
        map(
            i ->
            Symbol(:mid_, i) =>
                Conv(kern, nreg * nchan[i] => nreg * nchan[i + 1], gelu),
            1:(length(nchan) - 1),
        )...,
        sink = Conv(kern, nreg * nchan[end] => nten; use_bias = false),
        symm = WrappedFunction() do σ
            if D == 2
                xx = selectdim(σ, 3, 1:1)
                xy = (selectdim(σ, 3, 2:2) + selectdim(σ, 3, 3:3)) / 2
                yy = selectdim(σ, 3, 4:4)
                cat(xx, yy, xy; dims = 3)
            else
                xx = selectdim(σ, 4, 1:1)
                yy = selectdim(σ, 4, 5:5)
                zz = selectdim(σ, 4, 9:9)
                xy = (selectdim(σ, 4, 2:2) + selectdim(σ, 4, 4:4)) / 2
                yz = (selectdim(σ, 4, 6:6) + selectdim(σ, 4, 8:8)) / 2
                zx = (selectdim(σ, 4, 3:3) + selectdim(σ, 4, 7:7)) / 2
                cat(xx, yy, zz, xy, yz, zx; dims = 4)
            end
        end,
    )
    net |> display
    ps =
        (;
        lift = (;
            weight = kaiming_uniform(rng, T, nten, nchan[1]),
            bias = zeros(T, nchan[1]),
        ),
        map(
            i ->
            Symbol(:mid_, i) => (;
                weight = kaiming_uniform(rng, T, nreg, nchan[i + 1], nchan[i]),
                bias = zeros(T, nchan[i + 1]),
            ),
            1:(length(nchan) - 1),
        )...,
        sink = (; weight = kaiming_uniform(rng, T, nten, nchan[end])),
        symm = (;),
    ) |> dev
    st = map(Returns((;)), ps)
    return (; project, net, ps, st)
end

"Same as `equivariant_net` but without the weight projection."
function cnn(setup, nchan; same_as_equi)
    (; D, backend) = setup
    dev = adapt(backend)
    # dev = identity
    rng = Xoshiro(0)
    f = f64
    nt_nonsym = D^2
    nt = D == 2 ? 3 : 6
    (; elements) = group_stuff(D)
    nreg = if same_as_equi
        length(elements)
    else
        1
    end
    kern = ntuple(Returns(1), D)
    net = Chain(;
        lift = Conv(kern, nt_nonsym => nreg * nchan[1], gelu),
        map(
            i ->
            Symbol(:mid_, i) =>
                Conv(kern, nreg * nchan[i] => nreg * nchan[i + 1], gelu),
            1:(length(nchan) - 1),
        )...,
        sink = Conv(kern, nreg * nchan[end] => nt; use_bias = false),
        symm = WrappedFunction(identity),
    )
    net |> display
    ps, st = Lux.setup(rng, net) |> f |> dev
    project = identity # No projection
    return (; project, net, ps, st)
end
