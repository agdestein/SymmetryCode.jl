# Inference-closure construction for evaluation (the coordinate-driven case/setup
# layer lives in experiment.jl; classical closures are built here against the LES grid).

"""
Build the inference closures listed in `models` against eval `setup`, returned as
a NamedTuple keyed by `modelname(m)`. Learned entries are model coordinates
`(; arch, tier, netseed, use_redelta)`, loaded via [`build_model`](@ref) from
`psfile`; classical entries are symbols built against a fresh LES grid. (`:convsym`
and the seed sweep are handled by the eval driver, not here.)
"""
function build_models(case, setup, models)
    g = Grid{case.D}(; case.l, n = case.n_les, case.backend)
    classical = (;
        nomo = () -> (_, _) -> fill!(stack(spacetensorfield(g)), 0),
        dynsmag = () -> create_dynamic_smagorinsky(setup.Δ, g),
        clar = () -> create_clark(setup.Δ, g),
        smag = () -> create_smagorinsky(0.1, setup.Δ, g),
        vers = () -> create_verstappen(1.0, setup.Δ, g),
        bard = () -> create_bardina(2.0, setup.Δ, g),
    )
    return NamedTuple(
        (m isa NamedTuple ? modelname(m) => build_model(case, m, setup) : m => classical[m]())
            for m in models
    )
end
