# Generate the forced-HIT DNS data for the Re_Δ experiment.
#
# Coordinate-driven: `case_snellius` holds everything shared across the sweep and
# `dns_runs()` lists the (ν, seed, role) realizations. For each run we warm up the
# DNS (`create_dns` -> dnsfile) and generate the (ūbar, τ) pairs at every filter
# ratio of the run's role (`create_data` -> dnsmetafile + per-Δ fieldsfile/lesmeta).
# `run-les.jl` consumes these.

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

# Per-script table file (create-data.jl and run-les.jl share case.plotdir).
tablefile() = "tables-data.txt"
reset_tables(case; kwargs...) = S.reset_tables(case; filename = tablefile(), kwargs...)
tabulate(args...; kwargs...) = S.tabulate(args...; filename = tablefile(), kwargs...)

"Time-mean of a vector of NamedTuple statistics."
timemean(stats) = let ks = keys(first(stats))
    NamedTuple{ks}(map(k -> mean(getindex.(stats, k)), ks))
end

function main()
    case = get_case()
    config = get_config()
    reset_tables(case)

    slurm_id = get(ENV, "SLURM_ARRAY_TASK_ID", nothing)
    task_id = isnothing(slurm_id) ? nothing : parse(Int, slurm_id)

    for (i, dns) in enumerate(S.dns_runs().all)
        isnothing(task_id) || i == task_id || continue
        @info "===== DNS run: visc=$(dns.visc), seed=$(dns.seed), role=$(dns.role) ====="
        flush(stderr)
        filters = dns.role === :train ? case.filters_train : case.filters_test

        if :dns in config.experiments
            S.create_dns(case, dns; force = :dns in config.force)
            let stats = load(S.dnsfile(case, dns), "statistics")
                tabulate(case, "DNS stats after warm-up (visc=$(dns.visc), seed=$(dns.seed))", stats[end])
            end
        end

        :evolution_dns in config.experiments && S.plot_evolution_dns(case, dns)
        if :spectrum_dns in config.experiments
            for Δf in filters
                S.plot_spectrum_dns(case, dns, Δf)
            end
        end

        if :data in config.experiments
            S.create_data(case, dns; force = :data in config.force)
            let stats = load(S.dnsmetafile(case, dns), "statistics_dns")
                tabulate(
                    case,
                    "Time-averaged DNS stats, data window (visc=$(dns.visc), seed=$(dns.seed))",
                    timemean(stats),
                )
            end
        end

        :evolution_data in config.experiments && S.plot_evolution_data(case, dns)
        if :spectrum_data in config.experiments
            for Δf in filters
                S.plot_spectrum_data(case, dns, Δf)
            end
        end
    end

    @info "Done."
    flush(stderr)
    return
end

main()
