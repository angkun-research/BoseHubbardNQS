"""A 2D Torch model and validated inference-checkpoint metadata.

Flatten single configurations with x fastest: i=y*Lx+x (Python indices).
Raw inference permits all particle sectors; fixed-N sampling belongs to workflows.
"""
struct NQS2D
    model::Py
    metadata::Dict{String,Any}
end

const _nqs2d = Ref{Py}()
function nqs2d()
    if !isassigned(_nqs2d)
        bhnet()  # install repository and bridge import roots, never the legacy 2D directory
        _nqs2d[] = pyimport("bosehubbard_nqs_2d_bridge")
    end
    return _nqs2d[]
end

# PythonCall's Any conversion keeps nested containers as PyDict/PyList wrappers.
# Materialize checkpoint metadata and child JSON reports as native Julia data.
function _julia_data(value::Py)
    if pyisinstance(value, pybuiltins.dict)
        return Dict{String,Any}(pyconvert(String, k) => _julia_data(value[k]) for k in value)
    elseif pyisinstance(value, pybuiltins.list) || pyisinstance(value, pybuiltins.tuple)
        return Any[_julia_data(v) for v in value]
    end
    return pyconvert(Any, value)
end

"""Load an eval-mode 2D checkpoint, preserving saved float32/64. Defaults to CPU."""
function load_nn_2d(path::AbstractString; device=nothing)
    result = nqs2d().load_model(abspath(path); device=device)
    return NQS2D(result[0], _julia_data(result[1]))
end

function nn_value(model::NQS2D, config::AbstractVector{<:Integer}; nmax=nothing)
    return pyconvert(Float64, nqs2d().predict_scalar(model.model, collect(Int, config); nmax=nmax))
end

function nn_values(model::NQS2D, configs::AbstractMatrix{<:Integer}; nmax=nothing, chunk_size=1024)
    result = nqs2d().predict_values(model.model, Py(Matrix{Int}(configs)); nmax=nmax, chunk_size=chunk_size)
    return pyconvert(Vector{Float64}, result)
end

_workflow_option(v::Union{AbstractVector,Tuple}) = pylist(_workflow_option(x) for x in v)
_workflow_option(v) = v

function _workflow_2d(workflow; kwargs...)
    options = Dict{String,Any}(string(k) => v for (k,v) in kwargs)
    # Keep caller-relative input paths meaningful when the child runs at repo root.
    for key in ("checkpoint", "warm_start", "output")
        if haskey(options, key) && options[key] !== nothing
            options[key] = abspath(options[key])
        end
    end
    result = nqs2d().run_workflow(workflow, pydict(k => _workflow_option(v) for (k,v) in options))
    return _julia_data(result)
end

"""Run the existing 2D VMC CLI in a child Python process; return report, paths and logs.

Keywords map to CLI flags (e.g. update_method, sr_solver, pbc_x); CLI guards apply.
Checkpoints contain inference weights, not exact optimizer/walker resume state.
"""
train_vmc_2d(; kwargs...) = _workflow_2d("train"; kwargs...)
"""Fit a tiny or budgeted fixed-N ED sector via the existing isolated Python CLI."""
train_ed_2d(; kwargs...) = _workflow_2d("train_ed"; kwargs...)
"""Measure a frozen checkpoint via the existing isolated Python evaluation CLI."""
evaluate_2d(; kwargs...) = _workflow_2d("evaluate"; kwargs...)
