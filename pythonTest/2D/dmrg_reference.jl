#!/usr/bin/env julia
# Frozen CLI (run through ./julia-local from the repository root):
#   ./julia-local --threads=1 pythonTest/2D/dmrg_reference.jl [OPTIONS]
#   --Lx INT (4), --Ly INT (Lx), --Ntot INT (Lx*Ly), --Nmax INT (4)
#   --t FLOAT (1), --U FLOAT (16.74), --mu FLOAT (0), --seed INT (123)
#   --pbc | --no-pbc (both periodic by default)
#   --pbc-x | --no-pbc-x, --pbc-y | --no-pbc-y (override global flags)
#   --chi INT (128), --sweeps INT (6), --cutoff FLOAT (1e-9)
#   --spot-chi INT (0: disabled), --spot-sweeps INT (2)
#   --eigsolve-krylovdim INT (5), --eigsolve-maxiter INT (3)
#   --variance (optional H-squared contraction; omitted by default)
#   --output PATH (new directory relative to pythonTest/2D/), --help
# Cylinder: --no-pbc-x --pbc-y. Physical i=y*Lx+x is zero based; MPS indices
# are one based and snake through short y within each long-x column.
# Outputs: metadata.toml; sites.csv; bonds.csv; sweeps.csv; energy.csv;
# runtime.csv; chi<N>-density.csv, chi<N>-onsite_factorial.csv, chi<N>-rho.csv.
# CSV rho is the FULL physical-index <b_i^dag b_j>, with real/imag columns.
# Finite-chi reference, not exact or automatically converged. Optional spot run
# continues the primary MPS with zero noise; it is a sensitivity check, not a
# rigorous error bound. Guards: <=32 sites, Nmax<=9, chi<=256, <=12+4 sweeps.
# Uses existing ITensors/ITensorMPS and Julia stdlibs only. No package actions.

const SCRIPT_START = time_ns()
using ITensors, ITensorMPS
using LinearAlgebra, Random, Dates, SHA, TOML

const ROOT_2D = realpath(@__DIR__)
elapsed(start) = Float64(time_ns() - start) / 1e9

function output_path(value)
    path = abspath(isabspath(value) ? value : joinpath(ROOT_2D, value))
    ancestor = path
    while !ispath(ancestor) && !islink(ancestor)
        ancestor = dirname(ancestor)
    end
    path = normpath(joinpath(realpath(ancestor), relpath(path, ancestor)))
    relative = relpath(path, ROOT_2D)
    if relative == "." || first(splitpath(relative)) == ".." || ispath(path) || islink(path)
        error("output must be a new directory under pythonTest/2D/")
    end
    return path
end

function parse_cli(arguments)
    defaults = Dict{String, Any}(
        "Lx" => 4, "Ly" => nothing, "Ntot" => nothing, "Nmax" => 4,
        "t" => 1.0, "U" => 16.74, "mu" => 0.0, "seed" => 123,
        "pbc" => true, "pbc_x" => nothing, "pbc_y" => nothing,
        "chi" => 128, "sweeps" => 6, "cutoff" => 1e-9,
        "spot-chi" => 0, "spot-sweeps" => 2,
        "eigsolve-krylovdim" => 5, "eigsolve-maxiter" => 3, "variance" => false,
        "output" => "outputs/dmrg-" * Dates.format(now(UTC), "yyyymmddTHHMMSSsss"),
    )
    integers = Set(["Lx", "Ly", "Ntot", "Nmax", "seed", "chi", "sweeps",
                    "spot-chi", "spot-sweeps", "eigsolve-krylovdim", "eigsolve-maxiter"])
    floats = Set(["t", "U", "mu", "cutoff"])
    flags = Dict("--pbc" => ("pbc", true), "--no-pbc" => ("pbc", false),
                 "--pbc-x" => ("pbc_x", true), "--no-pbc-x" => ("pbc_x", false),
                 "--pbc-y" => ("pbc_y", true), "--no-pbc-y" => ("pbc_y", false),
                 "--variance" => ("variance", true))
    k = 1
    while k <= length(arguments)
        flag = arguments[k]
        if flag == "--help"
            for line in readlines(@__FILE__)[2:end]
                startswith(line, "#") || break
                println(chopprefix(line, "# "))
            end
            return nothing
        elseif haskey(flags, flag)
            key, value = flags[flag]
            defaults[key] = value
            k += 1
            continue
        end
        startswith(flag, "--") || error("expected a named --option, got $flag")
        key = flag[3:end]
        haskey(defaults, key) && (key in integers || key in floats || key == "output") || error("unknown option $flag")
        k < length(arguments) || error("missing value for $flag")
        value = arguments[k + 1]
        defaults[key] = key in integers ? parse(Int, value) : key in floats ? parse(Float64, value) : value
        k += 2
    end
    o = defaults
    o["Ly"] = isnothing(o["Ly"]) ? o["Lx"] : o["Ly"]
    n = o["Lx"] * o["Ly"]
    o["Ntot"] = isnothing(o["Ntot"]) ? n : o["Ntot"]
    1 <= o["Lx"] <= 32 && 1 <= o["Ly"] <= 32 && n <= 32 || error("geometry guard: require 1..32 sites")
    0 <= o["Nmax"] <= 9 && 0 <= o["Ntot"] <= n * o["Nmax"] || error("invalid fixed-number sector or local cutoff")
    all(isfinite(o[key]) for key in ("t", "U", "mu", "cutoff")) || error("physical scalars and cutoff must be finite")
    o["t"] >= 0 && o["U"] >= 0 && 0 < o["cutoff"] < 1 || error("require t,U>=0 and 0<cutoff<1")
    1 <= o["chi"] <= 256 && 1 <= o["sweeps"] <= 12 || error("budget guard: chi<=256, sweeps<=12")
    (o["spot-chi"] == 0 || o["chi"] < o["spot-chi"] <= 256) && 1 <= o["spot-sweeps"] <= 4 || error("spot chi must exceed primary chi and be <=256; spot sweeps<=4")
    2 <= o["eigsolve-krylovdim"] <= 20 && 1 <= o["eigsolve-maxiter"] <= 10 || error("invalid bounded eigensolver settings")
    0 <= o["seed"] || error("seed must be nonnegative")
    for axis in ("pbc_x", "pbc_y")
        o[axis] = isnothing(o[axis]) ? o["pbc"] : o[axis]
    end
    o["output"] = output_path(o["output"])
    return o
end

function geometry(Lx, Ly, px, py)
    bonds = Set{Tuple{Int, Int}}()
    mapping = zeros(Int, Lx * Ly) # physical i+1 -> one-based MPS site
    for x in 0:(Lx - 1), y in 0:(Ly - 1)
        i = y * Lx + x
        mapping[i + 1] = x * Ly + (iseven(x) ? y : Ly - 1 - y) + 1
        for (nx, ny) in ((x + 1, y), (x, y + 1))
            ((nx >= Lx && !px) || (ny >= Ly && !py)) && continue
            j = mod(ny, Ly) * Lx + mod(nx, Lx)
            i != j && push!(bonds, minmax(i, j))
        end
    end
    @assert sort(mapping) == collect(1:(Lx * Ly))
    return sort!(collect(bonds)), mapping
end

function hamiltonian(sites, bonds, mapping, o)
    terms = OpSum()
    for (i, j) in bonds
        a, b = mapping[i + 1], mapping[j + 1]
        terms += -o["t"], "Adag", a, "A", b
        terms += -o["t"], "Adag", b, "A", a
    end
    for j in eachindex(sites)
        terms += o["U"] / 2, "N * N", j
        terms += -o["U"] / 2 - o["mu"], "N", j
    end
    return MPO(terms, sites)
end

mutable struct SweepLog <: ITensorMPS.AbstractObserver
    phase::String
    started::UInt64
    maxerr::Float64
    rows::Vector{NamedTuple}
end

function ITensorMPS.measure!(o::SweepLog; spec, psi, sweep, sweep_is_done, energy, kwargs...)
    o.maxerr = max(o.maxerr, Float64(ITensorMPS.truncerror(spec)))
    if sweep_is_done
        push!(o.rows, (phase=o.phase, sweep=sweep, projected_energy=real(energy),
                       maxlinkdim=maxlinkdim(psi), max_truncation_both_halves=o.maxerr,
                       elapsed_seconds=elapsed(o.started)))
        o.maxerr = 0.0
    end
    return nothing
end

function optimize(H, psi, o, chi, sweeps; warm=false)
    label = "chi$chi"
    observer = SweepLog(label, time_ns(), 0.0, NamedTuple[])
    dimensions = warm ? fill(chi, sweeps) : [min(chi, 32 * 2^min(k - 1, 3)) for k in 1:sweeps]
    noise = [!warm && k <= 2 ? (k == 1 ? 1e-6 : 1e-7) : 0.0 for k in 1:sweeps]
    start = time_ns()
    _, state = dmrg(H, psi; nsweeps=sweeps, maxdim=dimensions, cutoff=o["cutoff"], noise=noise,
                    eigsolve_krylovdim=o["eigsolve-krylovdim"], eigsolve_maxiter=o["eigsolve-maxiter"],
                    observer=observer, outputlevel=1)
    seconds = elapsed(start)
    normalize!(state)
    return state, observer.rows, seconds, dimensions, noise
end

function measure_state(psi, H, mapping, bonds, o)
    norm2 = real(inner(psi, psi))
    energy = real(inner(prime(psi), H, psi)) / norm2
    variance = o["variance"] ? real(inner(H, psi, H, psi)) / norm2 - energy^2 : nothing
    density = real.(expect(psi, "N"))[mapping]
    factorial = (real.(expect(psi, "N * N")) .- real.(expect(psi, "N")))[mapping]
    # Force both triangles to be independently contracted for orientation checks.
    chain_rho = correlation_matrix(psi, "Adag", "A"; ishermitian=false)
    rho = chain_rho[mapping, mapping]
    qn = flux(psi)
    qn == QN("Number", o["Ntot"]) || error("MPS left the required Number QN sector: $qn")
    return summarize(energy, variance, norm2, density, factorial, rho, bonds, o; qn=string(qn), actual_chi=maxlinkdim(psi))
end

function summarize(energy, variance, norm2, density, factorial, rho, bonds, o; qn, actual_chi)
    rebuilt = o["U"] / 2 * sum(factorial) - o["mu"] * sum(density)
    rebuilt -= o["t"] * sum(real(rho[i + 1, j + 1] + rho[j + 1, i + 1]) for (i, j) in bonds; init=0.0)
    hermiticity = maximum(abs.(rho - rho'))
    diagonal = maximum(abs.(diag(rho) - density))
    number_error = abs(sum(density) - o["Ntot"])
    all(isfinite, density) && all(isfinite, factorial) && all(isfinite, rho) || error("nonfinite observables")
    all(isfinite, (energy, norm2, rebuilt)) || error("nonfinite energy or norm")
    number_error < 1e-8 && hermiticity < 1e-8 && diagonal < 1e-8 || error("density/rho invariants failed")
    abs(rebuilt - energy) < 1e-8 * max(1, abs(energy)) || error("observable energy reconstruction failed")
    isnothing(variance) || (isfinite(variance) && variance >= -1e-8) || error("invalid energy variance")
    minimum(eigvals(Hermitian(rho))) >= -1e-8 || error("one-body matrix is not positive semidefinite")
    metrics = Dict{String, Any}("energy"=>energy, "energy_per_site"=>energy / length(density),
        "energy_from_observables"=>rebuilt, "norm_squared"=>norm2,
        "energy_variance_status"=>isnothing(variance) ? "omitted; --variance not requested" : "computed",
        "density_number_error"=>number_error, "rho_hermiticity_max_abs"=>hermiticity,
        "rho_diagonal_density_max_abs"=>diagonal, "number_qn"=>qn, "actual_maxlinkdim"=>actual_chi,
        "density"=>density, "factorial"=>factorial, "rho"=>rho)
    !isnothing(variance) && (metrics["energy_variance"] = variance)
    return metrics
end

function analytic_state(occupations, bonds, o)
    density = Float64.(occupations)
    factorial = density .* (density .- 1)
    energy = o["U"] / 2 * sum(factorial) - o["mu"] * sum(density)
    return summarize(energy, o["variance"] ? 0.0 : nothing, 1.0, density, factorial, Matrix(Diagonal(density)), bonds, o;
                     qn="Number=$(o["Ntot"]) (analytic Fock state)", actual_chi=1)
end

function write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join(string.(Tuple(row)), ','))
        end
    end
end

function write_observables(output, label, metrics, Lx, Ly)
    density, factorial, rho = metrics["density"], metrics["factorial"], metrics["rho"]
    write_csv(joinpath(output, "$label-density.csv"), ["i", "x", "y", "density"],
              [(i, i % Lx, i ÷ Lx, density[i + 1]) for i in 0:(Lx * Ly - 1)])
    write_csv(joinpath(output, "$label-onsite_factorial.csv"), ["i", "x", "y", "n_nminus1"],
              [(i, i % Lx, i ÷ Lx, factorial[i + 1]) for i in 0:(Lx * Ly - 1)])
    write_csv(joinpath(output, "$label-rho.csv"), ["i", "j", "x_i", "y_i", "x_j", "y_j", "real", "imag"],
              [(i, j, i % Lx, i ÷ Lx, j % Lx, j ÷ Lx, real(rho[i + 1, j + 1]), imag(rho[i + 1, j + 1]))
               for i in 0:(Lx * Ly - 1) for j in 0:(Lx * Ly - 1)])
end

write_metadata(output, data) = open(io -> TOML.print(io, data; sorted=true), joinpath(output, "metadata.toml"), "w")

function main(arguments=ARGS)
    o = parse_cli(arguments)
    isnothing(o) && return nothing
    BLAS.set_num_threads(1)
    Random.seed!(o["seed"])
    output = o["output"]
    mkpath(output)
    meta = Dict{String, Any}("status"=>"running", "kind"=>"finite_chi_reference_not_exact",
        "created_at_utc"=>string(now(UTC))*"Z", "settings"=>o,
        "conventions"=>Dict("Hamiltonian"=>"-t sum_bonds(b_i^dag b_j+h.c.)+U/2 sum n(n-1)-mu sum n",
            "flattening"=>"i=y*Lx+x; physical zero-based", "snake"=>"short y inner within x columns, y reversed on odd x; MPS one-based",
            "bonds"=>"unique unordered nearest neighbors, no self bonds, one edge on length-two axes",
            "rho"=>"rho[i,j]=<b_i^dag b_j>, physical order; independent triangles"),
        "environment"=>Dict("julia"=>string(VERSION), "ITensors"=>string(pkgversion(ITensors)),
            "ITensorMPS"=>string(pkgversion(ITensorMPS)), "julia_threads"=>Threads.nthreads(), "blas_threads"=>BLAS.get_num_threads()),
        "source_sha256"=>Dict("dmrg_reference.jl"=>bytes2hex(sha256(read(@__FILE__))),
            "root_Project.toml"=>bytes2hex(sha256(read(joinpath(ROOT_2D, "..", "..", "Project.toml")))),
            "root_Manifest.toml"=>bytes2hex(sha256(read(joinpath(ROOT_2D, "..", "..", "Manifest.toml"))))),
        "reference_limitations"=>"Finite chi and sweep budget; spot difference is sensitivity, not rigorous error bar. No thermodynamic or criticality claim.",
        "phases"=>Dict{String, Any}[])
    write_metadata(output, meta)
    try
        Lx, Ly = o["Lx"], o["Ly"]
        n = Lx * Ly
        bonds, mapping = geometry(Lx, Ly, o["pbc_x"], o["pbc_y"])
        meta["bond_count"] = length(bonds)
        write_csv(joinpath(output, "sites.csv"), ["i", "x", "y", "mps_site_one_based"],
                  [(i, i % Lx, i ÷ Lx, mapping[i + 1]) for i in 0:(n - 1)])
        write_csv(joinpath(output, "bonds.csv"), ["i", "j", "mps_i_one_based", "mps_j_one_based"],
                  [(i, j, mapping[i + 1], mapping[j + 1]) for (i, j) in bonds])
        base, extra = divrem(o["Ntot"], n)
        occupations = [base + (i <= extra ? 1 : 0) for i in 1:n]
        trivial = n == 1 || o["t"] == 0 || o["Ntot"] == 0 || o["Ntot"] == n * o["Nmax"]
        H = psi = nothing
        if !trivial
            sites = siteinds("Boson", n; dim=o["Nmax"] + 1, conserve_qns=true)
            chain_occupations = occupations[sortperm(mapping)]
            psi = random_mps(sites, string.(chain_occupations); linkdims=min(4, o["chi"]))
            flux(psi) == QN("Number", o["Ntot"]) || error("initializer QN mismatch")
            H = hamiltonian(sites, bonds, mapping, o)
        end
        timings = [("imports_and_setup", elapsed(SCRIPT_START))]
        sweep_rows = NamedTuple[]
        energy_rows = []
        phases = [(o["chi"], o["sweeps"], false)]
        o["spot-chi"] > 0 && push!(phases, (o["spot-chi"], o["spot-sweeps"], true))
        for (chi, sweeps, warm) in phases
            label = "chi$chi"
            dimensions, noise = Int[], Float64[]
            if !trivial
                psi, rows, seconds, dimensions, noise = optimize(H, psi, o, chi, sweeps; warm=warm)
                append!(sweep_rows, rows)
                push!(timings, ("$label-dmrg", seconds))
            end
            start = time_ns()
            metrics = trivial ? analytic_state(occupations, bonds, o) : measure_state(psi, H, mapping, bonds, o)
            push!(timings, ("$label-final_observables", elapsed(start)))
            write_observables(output, label, metrics, Lx, Ly)
            push!(energy_rows, (label, chi, metrics["actual_maxlinkdim"], metrics["energy"], metrics["energy_per_site"],
                               metrics["energy_from_observables"], get(metrics, "energy_variance", ""), metrics["energy_variance_status"], metrics["norm_squared"]))
            phase = Dict(k=>v for (k, v) in metrics if !(k in ("density", "factorial", "rho")))
            merge!(phase, Dict("label"=>label, "requested_chi"=>chi, "requested_sweeps"=>sweeps,
                              "maxdim_schedule"=>dimensions, "noise_schedule"=>noise,
                              "method"=>trivial ? "analytic_Fock_sector" : "two_site_QN_DMRG"))
            push!(meta["phases"], phase)
            write_metadata(output, meta)
            println("$label final energy=$(metrics["energy"]) variance=$(get(metrics, "energy_variance", "omitted"))")
        end
        write_csv(joinpath(output, "energy.csv"), ["phase", "requested_chi", "actual_chi", "energy", "energy_per_site", "energy_from_observables", "energy_variance", "energy_variance_status", "norm_squared"], energy_rows)
        write_csv(joinpath(output, "sweeps.csv"), ["phase", "sweep", "projected_energy", "maxlinkdim", "max_truncation_both_halves", "elapsed_seconds"], sweep_rows)
        push!(timings, ("total_through_final_observables_and_CSV", elapsed(SCRIPT_START)))
        write_csv(joinpath(output, "runtime.csv"), ["component", "seconds"], timings)
        meta["runtime_scope"] = "Seconds from script entry (includes imports/JIT/setup) through final observables/CSV; excludes Julia executable startup and final metadata write. Stage times include JIT on first use."
        meta["status"] = "completed"
        write_metadata(output, meta)
        println("reference output: $output")
    catch err
        meta["status"] = "failed"
        meta["error"] = sprint(showerror, err)
        write_metadata(output, meta)
        rethrow()
    end
    return output
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
