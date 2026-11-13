# Batching and CUDA

## Layout and return values

| Boundary | Shape |
| --- | --- |
| Julia single configuration | `(sites,)` integer vector |
| Julia batch | `(sites, walkers)` integer matrix |
| PyTorch occupations | `(walkers, sites)` |
| PyTorch one-hot network input | `(walkers, sites, Nmax+1)` |
| Julia batch result | `(walkers,)` vector of `Float64` log amplitudes |

Occupation values are counts `0:Nmax`, not Julia indices. All columns must
match the loaded model's site count. Passing a matrix to `nn_values` always
means a batch; flatten a single 2D image explicitly before `nn_value`.

```julia
configs = hcat([1, 1, 1, 1], [0, 2, 1, 1])
values = nn_values(model, configs; chunk_size=512)
```

Chunking bounds each inference forward pass while preserving column order. It
does not bound the complete returned vector or the memory of an entire VMC run.
The batch path avoids a separate Python call and `.item()` synchronization for
every walker.

## Device policy

| Request | Behavior |
| --- | --- |
| Existing 1D call without `device` | Preserves legacy CPU behavior |
| `load_nn_2d` without `device` | CPU inference |
| `device="cpu"` | CPU model and inputs |
| `device="auto"` | CUDA when PyTorch reports availability, otherwise CPU |
| `device="cuda"` or `"cuda:0"` | Explicit CUDA execution; unavailable devices raise an error |

The model moves once at loading. Inference inputs follow the model's parameter
device. CUDA availability is determined by the Python interpreter used by
PythonCall. Installing Julia CUDA packages alone does not configure PyTorch.
The training and measurement wrappers retain the Python CLIs' `device="auto"`
default; specify CPU explicitly when that is the intended workflow device.

The initial 1D bridge preserves its float32 inference contract. Returning a
Julia `Float64` does not add precision to a float32 forward pass. 2D inference
preserves the checkpoint's float32 or float64 dtype.

## Performance and numerical agreement

Batching amortizes Python-call and device-transfer overhead. GPU inference is
most useful when there is enough work per call; a small single-configuration
network may run faster on CPU. Measure the walker counts and architectures you
actually use, including warm-up and transfers back to Julia.

CPU/GPU and batched/scalar results can differ slightly because kernels and
reduction orders differ. Compatibility tests use dtype-appropriate tolerances.
The existing CPU scalar path is separately checked against its legacy values.

For custom Julia MC sampling, cache current log amplitudes and evaluate walker
proposals together. Full energy-based VMC continues to use the existing Python
sampler and SR implementation. Its internal device tensors are distinct from
the host arrays returned by the inference interface.
