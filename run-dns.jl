# Generate the forced-HIT DNS data for the Re_Δ experiment.
#
# Coordinate-driven: `case_snellius` holds everything shared across the sweep and
# `dns_runs()` lists the (ν, seed, role) realizations. For each run we warm up the
# DNS (`create_dns` -> dnsfile) and generate the (ūbar, τ) pairs at every filter
# ratio of the run's role (`create_data` -> dnsmetafile + per-Δ fieldsfile/lesmeta).
# `run-les.jl` consumes these.
#
# Phases (`julia run-dns.jl <phase>`, default `all` = serial end-to-end):
# `data` generates one DNS run per SLURM_ARRAY_TASK_ID; `reduce` writes the shared
# stat tables; `count` prints the `data` array size. `submit.sh` chains data →
# reduce with an `afterok` dependency.

@info "Loading packages"
flush(stderr)

using Adapt
using CairoMakie
using CUDA
using JLD2
using Statistics: mean

import SymmetryCode as S

get_case() = S.case_snellius()

get_config() = (;
    # Pipeline stages, in order. `:dns` writes dnsfile (warm-up); `:data` reads it
    # and writes dnsmetafile + per-Δ fieldsfile/lesmetafile. The plot stages read
    # whichever artifact already exists.
    experiments = [
        :dns,            # create_dns -> dnsfile (DNS warm-up)
        :evolution_dns,  # plot_evolution_dns -> warm-up time series
        :spectrum_dns,   # plot_spectrum_dns -> DNS vs filtered-DNS spectrum (per Δ)
        :data,           # create_data -> dnsmetafile + fieldsfile/lesmetafile per Δ
        :evolution_data, # plot_evolution_data -> data-window time series
        :spectrum_data,  # plot_spectrum_data -> time-averaged spectra (per Δ)
    ],

    # Stages whose cache is invalidated for this run. Only :dns/:data are cached;
    # the plots always regenerate. Add e.g. `push!(config.force, :data)` at the REPL.
    force = Set{Symbol}([]),
)

# Per-script table file (run-dns.jl and run-les.jl share case.plotdir).
tablefile() = "tables-data.txt"
reset_tables(case; kwargs...) = S.reset_tables(case; filename = tablefile(), kwargs...)
tabulate(args...; kwargs...) = S.tabulate(args...; filename = tablefile(), kwargs...)

"Time-mean of a vector of NamedTuple statistics."
timemean(stats) = let ks = keys(first(stats))
    NamedTuple{ks}(map(k -> mean(getindex.(stats, k)), ks))
end

print_count(case) = println(length(S.dns_runs().all))

"""
Phase `data` (array over `dns_runs().all`): warm up the DNS (`create_dns`) and
generate the (ūbar, τ) data (`create_data`) for one run, plus that run's plots.
`task_id === nothing` does every run; an integer does only that 1-based run.
"""
function run_data!(case, config, task_id)
    for (i, dns) in enumerate(S.dns_runs().all)
        isnothing(task_id) || i == task_id || continue
        @info "===== DNS run $i: visc=$(dns.visc), seed=$(dns.seed), role=$(dns.role) ====="
        flush(stderr)
        filters = dns.role === :train ? case.filters_train : case.filters_test

        :dns in config.experiments && S.create_dns(case, dns; force = :dns in config.force)
        :evolution_dns in config.experiments && S.plot_evolution_dns(case, dns)
        if :spectrum_dns in config.experiments
            for Δf in filters
                S.plot_spectrum_dns(case, dns, Δf)
            end
        end

        :data in config.experiments && S.create_data(case, dns; force = :data in config.force)
        :evolution_data in config.experiments && S.plot_evolution_data(case, dns)
        if :spectrum_data in config.experiments
            for Δf in filters
                S.plot_spectrum_data(case, dns, Δf)
            end
        end
    end
    return
end

"""
Phase `reduce` (serial): the per-run DNS-stat tables and the paper-ready
cross-dataset DNS-stats table, read back from what the `data` phase wrote (a run
whose artifact is missing is skipped). The shared text table is written only here,
so array `data` tasks never race it.
"""
function run_reduce!(case, config)
    reset_tables(case)
    for dns in S.dns_runs().all
        :dns in config.experiments && isfile(S.dnsfile(case, dns)) && tabulate(
            case, "DNS stats after warm-up (visc=$(dns.visc), seed=$(dns.seed))",
            load(S.dnsfile(case, dns), "statistics")[end],
        )
        :data in config.experiments && isfile(S.dnsmetafile(case, dns)) && tabulate(
            case, "Time-averaged DNS stats, data window (visc=$(dns.visc), seed=$(dns.seed))",
            timemean(load(S.dnsmetafile(case, dns), "statistics_dns")),
        )
    end
    S.write_dns_table(case, S.dns_runs().all)
    @info "Done (reduce)."
    flush(stderr)
    return
end

"""
Entry point. `julia run-dns.jl [phase]`, `phase ∈ all|data|reduce|count`
(default `all` = every run then the tables, serial). The cluster submits `data` as
a SLURM array (one unit per DNS run) then `reduce` serially, chained by `afterok`
(see `submit.sh`); `count` prints the `data` array size.
"""
function (@main)(args)
    case = get_case()
    config = get_config()
    phase = isempty(args) ? "all" : first(args)
    task_id = let s = get(ENV, "SLURM_ARRAY_TASK_ID", nothing)
        isnothing(s) ? nothing : parse(Int, s)
    end

    phase in ("all", "data", "reduce", "count") ||
        error("unknown phase '$(phase)'; expected all|data|reduce|count")

    if phase == "count"
        print_count(case)
    elseif phase == "all"
        run_data!(case, config, nothing)
        run_reduce!(case, config)
    elseif phase == "data"
        run_data!(case, config, task_id)
    elseif phase == "reduce"
        run_reduce!(case, config)
    end
    return 0
end
