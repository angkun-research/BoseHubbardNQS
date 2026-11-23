using ITensors, ITensorMPS
using LinearAlgebra, Random, Statistics, TOML

mutable struct SweepLog <: AbstractObserver
    sweep_truncation::Float64
    energies::Vector{Float64}
    truncations::Vector{Float64}
end

function ITensorMPS.measure!(log::SweepLog; spec, energy, sweep_is_done, kwargs...)
    log.sweep_truncation = max(log.sweep_truncation, ITensorMPS.truncerror(spec))
    if sweep_is_done
        push!(log.energies, real(energy))
        push!(log.truncations, log.sweep_truncation)
        log.sweep_truncation = 0.0
    end
end

function square_torus(L)
    # Physical indexing agrees with pythonTest/2D/lattice.py. The MPS snakes
    # through columns, reversing direction on alternating columns.
    site(x, y) = 1 + mod(x, L) + L * mod(y, L)
    mapping = [1 + x * L + (iseven(x) ? y : L - 1 - y)
               for y in 0:L-1 for x in 0:L-1]
    bonds = [(site(x, y), site(x + dx, y + dy))
             for y in 0:L-1 for x in 0:L-1 for (dx, dy) in ((1, 0), (0, 1))]
    @assert length(Set(minmax(i, j) for (i, j) in bonds)) == 2L^2
    return mapping, bonds
end

function hamiltonian(sites, mapping, bonds, U)
    terms = OpSum()
    for (i, j) in bonds
        a, b = mapping[i], mapping[j]
        terms += -1.0, "Adag", a, "A", b
        terms += -1.0, "Adag", b, "A", a
    end
    for i in eachindex(sites)
        terms += U / 2, "N * N", i
        terms += -U / 2, "N", i
    end
    return MPO(terms, sites)
end

function correlations(state, mapping, L; midpoint=false)
    rho = correlation_matrix(state, "Adag", "A"; ishermitian=true)[mapping, mapping]
    momentum_x = [cis(2π * mod(i-1, L) / L) for i in 1:L^2]
    momentum_y = [cis(2π * ((i-1) ÷ L) / L) for i in 1:L^2]
    n0 = sum(rho) / L^2
    nkx = real(dot(momentum_x, rho * momentum_x)) / L^2
    nky = real(dot(momentum_y, rho * momentum_y)) / L^2
    result = Dict("n0" => n0, "nkx" => nkx, "nky" => nky,
                  "ratio" => 1 - (nkx + nky) / (2n0),
                  "density_range" => maximum(real.(diag(rho))) - minimum(real.(diag(rho))))
    if midpoint
        r = L ÷ 2
        result["G_mid_x"] = mean(real(rho[1+x+L*y, 1+mod(x+r, L)+L*y])
                                 for y in 0:L-1 for x in 0:L-1)
        result["G_mid_y"] = mean(real(rho[1+x+L*y, 1+x+L*mod(y+r, L)])
                                 for y in 0:L-1 for x in 0:L-1)
        result["G_mid_diag"] = mean(real(rho[1+x+L*y, 1+mod(x+r, L)+L*mod(y+s, L)])
                                    for s in (-r, r) for y in 0:L-1 for x in 0:L-1)
    end
    return result
end

function save_result(path, data)
    temporary_path = path * ".tmp"
    open(io -> TOML.print(io, data; sorted=true), temporary_path, "w")
    Base.Filesystem.rename(temporary_path, path)
end

function main(args)
    L, nmax, chi, sweeps, seed = parse.(Int, args[[1, 3, 4, 5, 6]])
    U = parse(Float64, args[2])
    mode, output = args[7:8]
    @assert L >= 3 && nmax >= 2 && chi >= 4 && sweeps >= 4 && seed >= 0
    offsets = Dict("neutral" => (0,), "hole" => (-1,), "particle" => (1,), "gap" => (0, -1, 1))
    @assert isfinite(U) && U >= 0 && haskey(offsets, mode)
    BLAS.set_num_threads(1)
    Random.seed!(seed)
    mapping, bonds = square_torus(L)
    sites = siteinds("Boson", L^2; dim=nmax + 1, conserve_qns=true)
    H = hamiltonian(sites, mapping, bonds, U)
    dimensions = [min.(chi, [32, 64, 128]); fill(chi, sweeps - 3)]
    sectors = Dict{String, Any}()
    data = Dict("L" => L, "U" => U, "nmax" => nmax, "boundary" => "periodic_xy",
                "chi_requested" => chi, "seed" => seed, "mode" => mode,
                "sweeps_requested" => sweeps, "status" => "running", "sectors" => sectors,
                "ITensors" => string(pkgversion(ITensors)),
                "ITensorMPS" => string(pkgversion(ITensorMPS)))
    for offset in offsets[mode]
        occupations = fill("1", L^2)
        occupations[mapping[1]] = string(1 + offset)
        initial = random_mps(sites, occupations; linkdims=4)
        @assert flux(initial) == QN("Number", L^2 + offset)
        log = SweepLog(0.0, Float64[], Float64[])
        _, state = dmrg(H, initial; nsweeps=sweeps, maxdim=dimensions, cutoff=1e-12,
                        noise=[1e-6, 1e-7, 1e-8, 0.0], observer=log, outputlevel=1,
                        eigsolve_krylovdim=5, eigsolve_maxiter=3)
        normalize!(state)
        @assert flux(state) == QN("Number", L^2 + offset)
        result = Dict{String, Any}("energy" => real(inner(state', H, state)),
            "chi" => maxlinkdim(state), "sweep_energies" => log.energies,
            "sweep_truncations" => log.truncations)
        offset == 0 && merge!(result, correlations(state, mapping, L))
        sectors[string(offset)] = result
        save_result(output, data)
    end
    data["status"] = "complete"
    save_result(output, data)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main(ARGS)
end
