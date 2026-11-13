# Python API

## 1D entry points

`pythonTest/TrainVMCstates.py` is the checkpoint-producing 1D VMC CLI.
`pythonTest/NeuralNetworkClass.py` defines the four supported VMC model families
and checkpoint metadata. The Julia bridge uses these same classes.

Use `--help` on training entry points for current argument names. The
[workflow guide](../python.md) provides complete runnable commands.

## 2D package

The importable namespace is `pythonTest.bosehubbard2d`. The existing
`pythonTest/2D/` scripts remain available. The namespaced route isolates common
module names such as `models`, `lattice` and `vmc` from other Python packages.

```python
from pythonTest.bosehubbard2d.models import build_model, load_model
from pythonTest.bosehubbard2d.exact_diagonalization import exact_ground_state
```

Do not mix the legacy bare-module and namespaced import routes in one process.
They are guarded against duplicate class identities; CLI examples run in their
own processes.

| Function | Contract |
| --- | --- |
| `build_model(arch, Lx, Ly=None, Nmax=4, pbc=True, **settings)` | Construct an unnormalized real log-amplitude model; explicit axis-boundary overrides are supported |
| `load_model(path, device="cpu")` | Return `(model, metadata)`, preserving saved dtype and validating checkpoint conventions |
| `save_model(path, model, Ntot=None, t=1, U=16.74, mu=0)` | Save inference weights and metadata under the 2D directory |
| `parameter_count(model)` | Number of trainable parameters |
| `spatial_symmetries(model)` | Guaranteed spatial invariances |
| `exact_ground_state(Lx, Ly=None, ...)` | Return `(energy, state, basis, sparse_H)` with pre-allocation resource guards |

All seven model families accept one-hot `(batch, sites, Nmax+1)` and return
`(batch,)`. Move inputs to the parameter device. For generic families, check
architecture settings such as transformer head divisibility. GCNN requires
periodic boundaries in both axes and reports D4 on squares or D2 on rectangles.

## Shared VMC operations

The namespaced `vmc` module exposes `generate_initial_states`, `sample`,
`local_energy`, `native_vmc_update`, `compute_log_psi_jacobian`, `sr_gradient`
and `sr_update`. It is the same implementation used by the CLI workflows.

Sampling preserves a chosen particle total. `sample` returns retained samples,
updated walkers and diagnostics; `local_energy` returns one local energy per
configuration. Ridge SR uses centered, sample-normalized Jacobians and requires
native SGD. Full Jacobians and dense solves have explicit resource checks.

For detailed signatures and shapes, consult the
[source API inventory](https://github.com/angkun-research/BoseHubbardNQS/blob/main/pythonTest/2D/docs/API.md).
These reference pages describe the library boundary; benchmark launchers are
reproduction tools rather than a second public VMC engine.
