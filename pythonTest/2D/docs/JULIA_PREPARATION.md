# Julia integration contract

The Julia/PyTorch bridge now exposes 2D inference and the existing Python
workflows. The Python algorithms and 1D/2D physical conventions remain separate.
The former preparation plan is retained in Git and in the pre-integration
snapshot under `outputs/integration-baseline-2026-11-13/`.

## Supported boundary

- `load_nn` retains the 1D Python-model return type, CPU default, loading keywords
  and scalar `Float64` API. Explicit `device` selection is additive.
- `load_nn_2d` returns `NQS2D`, containing the PyTorch model and validated metadata.
  It preserves saved float32/64 dtype, geometry, axis boundaries and physics.
- `nn_value`/`ansatz_log_amplitude` take one integer configuration vector.
  `nn_values` takes a `(sites,walkers)` integer matrix and returns one log
  amplitude per column. No scalar normalization or particle projection is added.
- Julia paths are resolved from the caller's working directory. The Python 2D
  routines retain their original `pythonTest/2D/` relative path convention.
- `train_ed_2d`, `train_vmc_2d` and `evaluate_2d` run the existing Python entry
  points in a child interpreter selected by PythonCall. This isolates RNG state,
  backend flags and workflow failures. Reports, paths and captured logs return
  to Julia. Output guards remain under `pythonTest/2D/`.

The maintained [Julia guide](../../../docs/site/julia.md),
[API reference](../../../docs/site/api/julia.md) and
[batch/device contract](../../../docs/site/batching.md) provide examples.

## Namespace and numerical preservation

`pythonTest.bosehubbard2d` loads the original modules from the existing `2D/`
location through an isolated namespace. Relative sibling imports prevent generic
`models`, `vmc` and `lattice` names from leaking into an embedded application.
Legacy CLI scripts retain their imports and paths in separate interpreters.
Mixing legacy and namespaced copies of the model module in one process is
rejected to protect class identity and checkpoint validation.

The namespace changes import wiring, not the Hamiltonian, sampler, local
energies, loss, SR equations or optimizer. Exact source snapshots remain with
historical benchmark evidence; recorded hashes identify those earlier versions.

## Physical and array conventions

2D uses `U*n*(n-1)/2`; 1D uses `U*n*(n-1)`. Equal interactions require
`U_1D=U_2D/2`. Periodic length-two bonds also differ. Neither is silently changed
by the adapter. Cross-dimension physics tests use an open chain of length four.

2D physical zero-based order is `i=y*Lx+x`. A Julia `(Lx,Ly)` image has the
required order under `vec`; a batch is sites by walkers. The Python one-hot
shape is `(batch,sites,Nmax+1)`. Frozen Python chains retain time and walker
axes; inference batch transport does not flatten these axes implicitly.

The selected PythonCall interpreter must contain the required PyTorch and
numerical dependencies. CUDA availability is checked in that interpreter.
Checkpoints remain inference-only; no exact optimizer/walker/RNG resume or
Julia CUDA zero-copy transport is promised.
