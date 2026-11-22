using ITensors, ITensorMPS
using LinearAlgebra, Random, Statistics, TOML

mutable struct SweepConvergence <: AbstractObserver
    energies::Vector{Float64}
    truncations::Vector{Float64}
    max_truncation::Float64
    min_sweeps::Int
end
SweepConvergence(min_sweeps) = SweepConvergence(Float64[], Float64[], 0.0, min_sweeps)

function ITensorMPS.measure!(observer::SweepConvergence; spec, kwargs...)
    observer.max_truncation = max(observer.max_truncation, ITensorMPS.truncerror(spec))
    return nothing
end

function ITensorMPS.checkdone!(observer::SweepConvergence; energy, sweep, kwargs...)
    push!(observer.energies, energy)
    push!(observer.truncations, observer.max_truncation)
    observer.max_truncation = 0.0
    sweep >= observer.min_sweeps || return false
    return maximum(abs, diff(observer.energies[end-2:end])) < 1e-10
end

function hamiltonian(sites, coupling, boundary)
    length = Base.length(sites)
    terms = OpSum()
    n_bonds = boundary == "periodic" ? length : length - 1
    for i in 1:n_bonds
        j = mod1(i + 1, length)
        terms += -1.0, "Adag", i, "A", j
        terms += -1.0, "A", i, "Adag", j
    end
    for i in 1:length
        terms += coupling / 2, "N * N", i
        terms += -coupling / 2, "N", i
    end
    return MPO(terms, sites)
end

function measure_correlations(state, length, boundary)
    density = expect(state, "N")
    covariance = correlation_matrix(state, "N", "N"; ishermitian=true) - density * density'
    rho = correlation_matrix(state, "Adag", "A"; ishermitian=true)
    g1 = Float64[]
    block_variance = Float64[]
    for r in 1:(length ÷ 2)
        if boundary == "periodic"
            push!(g1, mean(rho[i, mod1(i + r, length)] for i in 1:length))
            push!(block_variance, mean(1:length) do i
                indices = mod1.(i:i+r-1, length)
                sum(@view covariance[indices, indices])
            end)
        else
            lefts = unique(((length - r + 1) ÷ 2, (length - r + 2) ÷ 2))
            push!(g1, mean(rho[i, i+r] for i in lefts))
            starts = unique(((length - r + 2) ÷ 2, (length - r + 3) ÷ 2))
            push!(block_variance, mean(starts) do start
                indices = start:start+r-1
                sum(@view covariance[indices, indices])
            end)
        end
    end
    return Dict("particle_count" => sum(density), "density_cov_sum" => sum(covariance),
                "g1" => g1, "block_variance" => block_variance)
end

function charged_initial_state(sites, neutral_state, offset, maximum_bond)
    operator = offset == 1 ? "Adag" : "A"
    terms = OpSum()
    for i in eachindex(sites)
        terms += sinpi(i / (length(sites) + 1)), operator, i
    end
    guess = apply(MPO(terms, sites), neutral_state; cutoff=1e-12, maxdim=maximum_bond)
    normalize!(guess)
    return guess
end

function main(args)
    @assert Base.length(args) == 7 || (Base.length(args) == 8 && args[8] == "--warm-gap")
    length, nmax, chi = parse.(Int, args[[1, 3, 5]])
    coupling = parse(Float64, args[2])
    boundary, mode, output = args[[4, 6, 7]]
    @assert length >= 4 && nmax >= 2 && chi >= 2 && isfinite(coupling) && coupling > 0
    @assert boundary in ("open", "periodic") && mode in ("gap", "correlations")
    Random.seed!(123)
    BLAS.set_num_threads(1)
    sites = siteinds("Boson", length; dim=nmax + 1, conserve_qns=true)
    H = hamiltonian(sites, coupling, boundary)
    sectors = Dict{String, Any}()
    neutral = nothing
    warm_gap = Base.length(args) == 8
    @assert !warm_gap || (mode == "gap" && boundary == "open")
    for offset in (mode == "gap" ? (0, -1, 1) : (0,))
        use_neutral = warm_gap && offset != 0
        if use_neutral
            initial = charged_initial_state(sites, neutral, offset, chi)
        else
            occupation = fill("1", length)
            occupation[length ÷ 2] = string(1 + offset)
            initial = random_mps(sites, occupation; linkdims=8)
        end
        dimensions = use_neutral ? [chi] : min.(chi, [32, 64, 96, 128, 192, chi])
        noise = use_neutral ? [0.0] : [1e-6, 1e-7, 1e-8, 0.0]
        observer = SweepConvergence(use_neutral ? 4 : 10)
        energy, state = dmrg(H, initial; nsweeps=28,
            maxdim=dimensions, cutoff=1e-12, noise, observer, outputlevel=1,
            eigsolve_krylovdim=5, eigsolve_maxiter=2)
        warm_gap && offset == 0 && (neutral = state)
        measured_energy = real(inner(state', H, state))
        result = Dict{String, Any}(
            "energy" => measured_energy, "dmrg_energy" => energy,
            "chi" => maxlinkdim(state), "sweeps" => Base.length(observer.energies),
            "last_delta_E" => observer.energies[end] - observer.energies[end-1],
            "last_truncation" => observer.truncations[end],
            "converged_sweeps" => maximum(abs, diff(observer.energies[end-2:end])) < 1e-10,
        )
        println("N=$(length + offset): $result")
        flush(stdout)
        if mode == "correlations"
            merge!(result, measure_correlations(state, length, boundary))
        end
        sectors[string(offset)] = result
    end
    data = Dict("L" => length, "U_phys" => coupling, "nmax" => nmax,
                "boundary" => boundary, "mode" => mode, "chi_requested" => chi, "warm_gap" => warm_gap,
                "sectors" => sectors, "ITensors" => string(pkgversion(ITensors)),
                "ITensorMPS" => string(pkgversion(ITensorMPS)))
    tmp = "$output.tmp-$(getpid())"
    open(io -> TOML.print(io, data; sorted=true), tmp, "w")
    mv(tmp, output; force=true)
end

main(ARGS)
