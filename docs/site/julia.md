# Julia interface

Julia uses PythonCall to evaluate the existing PyTorch networks. The current
1D scalar API remains available; batching and 2D entry points are additive.

The [1D model guide](model-1d.md#use-the-trained-state-in-julia) and
[2D model guide](model-2d.md#use-the-trained-state-in-julia) provide complete
Python-training-to-Julia examples and tables of loading/evaluation keywords.

## Existing 1D calls

```julia
using BoseHubbardNQS

model = load_nn("data/example-1d.pt"; map_location="cpu")
sigma = [1, 1, 1, 1]
logpsi = nn_value(model, sigma)
same_logpsi = ansatz_log_amplitude(model, sigma)
```

`load_nn` returns the Python model as before. A single evaluation returns a
Julia `Float64` containing the network's raw log amplitude. Existing calls
continue to use a CPU model unless an execution device is requested explicitly.
The legacy `map_location` keyword controls checkpoint deserialization.

## Batches and CUDA

```julia
model = load_nn("data/example-1d.pt"; device="auto")
configs = hcat([1, 1, 1, 1], [0, 2, 1, 1])
logpsi_batch = nn_values(model, configs)
```

Each column is one configuration. The output vector has one entry per column,
in the same order. An empty batch has shape `(n_sites, 0)` and returns an empty
vector. See [batching and CUDA](batching.md) for chunking and device policy.

## 2D models

```julia
model2d = load_nn_2d("pythonTest/2D/outputs/docs-example/ed-2x3.pt";
                    device="auto")
sigma2d = [1, 1, 1, 1, 1, 1]
logpsi2d = nn_value(model2d, sigma2d)
batch2d = nn_values(model2d, hcat(sigma2d, [0, 2, 1, 1, 1, 1]))
```

The explicit loader returns a 2D model handle with checkpoint metadata. It
preserves the saved architecture, dtype, geometry, boundaries and physical
conventions. Single configurations use physical x-fast site order. For a Julia
lattice image shaped `(Lx, Ly)`, use `vec(image)` explicitly.

Julia checkpoint paths are resolved from the Julia working directory. In the
Python 2D loaders, relative paths instead start at `pythonTest/2D/`; pass absolute
paths when combining the two interfaces directly.

## Training and measurements

The Julia workflow entry points launch the established Python training and
evaluation programs in a subprocess. This isolates backend RNG state, flags and
GPU resources from the calling Julia process. Reports and saved model paths
return to Julia; scalar and batched inference remain in-process.

The [Julia API reference](api/julia.md) documents the workflow functions and
their options. Training is still performed by the same Python engine. Output
paths retain the 2D engine's guards, and saved checkpoints remain inference-only.
