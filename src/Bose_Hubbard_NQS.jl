using PythonCall

const _bhnet = Ref{Py}()

function bhnet()
    if !isassigned(_bhnet)
        sys = pyimport("sys")
        repo_root = normpath(joinpath(@__DIR__, ".."))      # BoseHubbardNQS/
        python_src = joinpath(@__DIR__, "python")           # BoseHubbardNQS/src/python
        sys.path.insert(0, repo_root)
        sys.path.insert(0, python_src)
        _bhnet[] = pyimport("Bose_Hubbard_NQS")
    end
    return _bhnet[]
end

function load_nn(path::String="data/bose_hubbard_nqs.pt";
                 arch=nothing, n_sites=nothing, nmax=nothing, hidden_dim=nothing,
                 map_location::String="cpu")
    return bhnet().load_model(
        path;
        arch=arch,
        n_sites=n_sites,
        nmax=nmax,
        hidden_dim=hidden_dim,
        map_location=map_location,
    )
end

function nn_value(model, occ_config::AbstractVector{<:Integer}; nmax=nothing)
    cfg = collect(Int.(occ_config))
    y = bhnet().predict_scalar(model, cfg; nmax=nmax)
    return pyconvert(Float64, y)
end

function ansatz_log_amplitude(model, sigma::AbstractVector{<:Integer}; nmax=nothing)
    return nn_value(model, sigma; nmax=nmax)
end