# Julia API

```julia
using BoseHubbardNQS
```

## 1D loading and evaluation

```julia
load_nn(path::String="data/bose_hubbard_nqs.pt";
        arch=nothing, n_sites=nothing, nmax=nothing, hidden_dim=nothing,
        map_location="cpu", device=nothing)
nn_value(model, sigma::AbstractVector{<:Integer}; nmax=nothing)
ansatz_log_amplitude(model, sigma::AbstractVector{<:Integer}; nmax=nothing)
nn_values(model, configs::AbstractMatrix{<:Integer}; nmax=nothing, chunk_size=1024)
```

`load_nn` returns the existing Python model object. Legacy loading keywords,
CPU defaults and scalar `Float64` results remain available. A supplied execution
`device` controls model placement; `map_location` retains its deserialization
meaning. The batch method returns a `Vector{Float64}` with one result per
configuration column. All outputs are raw log amplitudes. The 1D path argument is a `String`; use
`String(path)` for a `SubString` or another string subtype. The 2D loader
accepts `AbstractString`.

## 2D loading and evaluation

```julia
load_nn_2d(path; device=nothing)
nn_value(model2d, sigma::AbstractVector{<:Integer}; nmax=nothing)
ansatz_log_amplitude(model2d, sigma::AbstractVector{<:Integer}; nmax=nothing)
nn_values(model2d, configs::AbstractMatrix{<:Integer}; nmax=nothing, chunk_size=1024)
```

The loader returns `NQS2D`, a handle containing the Python model and checkpoint
metadata. It preserves `Lx`, `Ly`, occupation cutoff, particle-sector metadata,
both boundary flags, architecture, floating dtype and Hamiltonian conventions.
The familiar scalar and batch functions dispatch on this handle without
changing the existing 1D model type.

The occupation vector has `Lx*Ly` sites, ordered `i=y*Lx+x`. Inference does not
project into the saved particle-number sector. A scalar result is `Float64`;
a batch result is `Vector{Float64}`. Empty batches return an empty vector.

## 2D workflows

```julia
train_vmc_2d(; kwargs...)
train_ed_2d(; kwargs...)
evaluate_2d(; kwargs...)
```

The workflow wrappers launch the established Python CLIs using the PythonCall
interpreter and return their reports. They isolate backend global state from
Julia. See [the workflow examples](../python.md) for the options, physical
settings, output locations and measurement conventions of the underlying tools.
Workflows inherit CLI defaults, including `device="auto"`. Inference loaders
default to CPU. Pass `device="cpu"` explicitly for CPU-only workflow execution.

Keyword names follow CLI names, replacing hyphens with underscores. Preserve
capitalization for `Lx`, `Ly`, `Ntot`, `Nmax` and `U`. For example:

```julia
result = train_ed_2d(arch="FCNet", Lx=2, Ly=2, Ntot=4, Nmax=2,
                     width=4, epochs=2, device="cpu",
                     output="pythonTest/2D/outputs/julia-example/best.pt")
report = result["report"]
paths = result["paths"]
```

Reports are Julia dictionaries. Results also contain captured `stdout` and
`stderr` and output paths. Julia input/output paths are resolved from the
caller's working directory before entering the subprocess. Python CLI defaults
still resolve under the 2D directory when a path is omitted.

All 2D training and evaluation output paths remain guarded under
`pythonTest/2D/`. Checkpoints contain inference weights and metadata, not exact
training-resume state. A failed subprocess raises an error with its diagnostics.

## Validation and errors

Invalid site counts, out-of-range or non-integer occupations, unsupported
checkpoint schemas and unavailable explicit CUDA requests are rejected.
Use [the conventions](../physics.md) when comparing dimensions and
[the batch guide](../batching.md) for layout and precision details.
