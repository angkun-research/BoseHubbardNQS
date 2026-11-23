include("dmrg.jl")
using Serialization

mutable struct CheckpointObserver <: AbstractObserver
    log::SweepLog
    completed::Int
    signature::NamedTuple
    path::String
end

function ITensorMPS.measure!(observer::CheckpointObserver; kwargs...)
    ITensorMPS.measure!(observer.log; kwargs...)
end

function ITensorMPS.checkdone!(observer::CheckpointObserver; psi, sweep, kwargs...)
    completed = observer.completed + sweep
    temporary = observer.path * ".tmp"
    serialize(temporary, (; observer.signature, completed, state=psi,
                           energies=observer.log.energies, truncations=observer.log.truncations))
    # Replace the previous checkpoint in a single rename on the same filesystem.
    Base.Filesystem.rename(temporary, observer.path)
    return false
end

function staged_main(args)
    L, nmax, offset, seed = parse.(Int, args[[1, 3, 4, 5]])
    U = parse(Float64, args[2])
    stages = [(parse(Int, dim), parse(Int, count))
              for (dim, count) in split.(split(args[6], ','), ':')]
    prefix = args[7]
    cap = length(args) == 8 ? parse(Int, args[8]) : last(stages)[1]
    @assert L >= 3 && nmax >= 2 && offset in -1:1 && seed >= 0
    @assert isfinite(U) && U >= 0 && all(dim >= 4 && count > 0 for (dim, count) in stages)
    @assert issorted(first.(stages)) && allunique(first.(stages))
    @assert cap in first.(stages)
    BLAS.set_num_threads(1)
    Random.seed!(seed)

    signature = (; L, U, nmax, offset, seed, stages)
    checkpoint = prefix * ".jls"
    if isfile(checkpoint)
        saved = deserialize(checkpoint)
        saved.signature == signature || error("Checkpoint settings differ from the requested run")
        (; completed, state, energies, truncations) = saved
        sites = siteinds(state)
    else
        completed, energies, truncations = 0, Float64[], Float64[]
        sites = siteinds("Boson", L^2; dim=nmax + 1, conserve_qns=true)
        occupations = fill("1", L^2)
        occupations[cld(L^2, 2)] = string(1 + offset)
        state = random_mps(sites, occupations; linkdims=4)
    end
    @assert flux(state) == QN("Number", L^2 + offset)
    mapping, bonds = square_torus(L)
    H = hamiltonian(sites, mapping, bonds, U)
    mode = offset == 0 ? "neutral" : (offset == 1 ? "particle" : "hole")
    planned = 0
    for (chi, count) in stages
        chi > cap && break
        old_planned = planned
        planned += count
        planned < completed && continue
        if planned > completed
            remaining_noise = [j == 1 ? (old_planned == 0 ? 1e-6 : 1e-7) : 0.0
                               for j in (completed - old_planned + 1):count]
            log = SweepLog(0.0, energies, truncations)
            observer = CheckpointObserver(log, completed, signature, checkpoint)
            println("offset=$offset chi=$chi starting sweep $(completed + 1) / $planned")
            _, state = dmrg(H, state; nsweeps=planned-completed, maxdim=[chi], cutoff=1e-12,
                            noise=remaining_noise, observer, outputlevel=1,
                            eigsolve_krylovdim=5, eigsolve_maxiter=3)
            completed = planned
        end
        # The latest checkpoint can regenerate the measurement if a job exits during output.
        chi < 384 && chi != last(stages)[1] && continue
        normalize!(state)
        @assert flux(state) == QN("Number", L^2 + offset)
        sector = Dict{String, Any}("energy" => real(inner(state', H, state)),
            "chi" => maxlinkdim(state), "sweep_energies" => energies,
            "sweep_truncations" => truncations, "stage_sweeps" => count)
        offset == 0 && merge!(sector, correlations(state, mapping, L; midpoint=true))
        data = Dict("L" => L, "U" => U, "nmax" => nmax, "boundary" => "periodic_xy",
                    "chi_requested" => chi, "seed" => seed, "mode" => mode,
                    "sweeps_requested" => planned, "status" => "complete",
                    "sectors" => Dict(string(offset) => sector),
                    "ITensors" => string(pkgversion(ITensors)),
                    "ITensorMPS" => string(pkgversion(ITensorMPS)))
        save_result("$prefix-chi$chi.toml", data)
    end
end

staged_main(ARGS)
