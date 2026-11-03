# Julia integration preparation

This is a handoff contract and future merge plan, not a merged implementation.
The 2D Python algorithms and the existing 1D Julia adapter remain unchanged.

## Existing integration

The root Julia package uses `PythonCall`. Its `src/Bose_Hubbard_NQS.jl` loads
`src/python/Bose_Hubbard_NQS.py`, which imports the existing 1D networks and
exports the behavior behind `load_nn`, `nn_value`, and `ansatz_log_amplitude`.
That loader does not understand the 2D checkpoint schema. Do not pass a 2D
checkpoint to it or silently reinterpret its `n_sites` as a lattice shape.

The Python environment selected by Julia must provide the same PyTorch/NumPy/
SciPy dependencies used here. A working repository `.venv` alone does not prove
that Julia's selected interpreter has those dependencies. No interpreter,
Julia dependency, root `Project.toml`, or `Manifest.toml` is changed by this work.

## Current 2D interface to preserve

See [API.md](API.md) for exact signatures and return values.

| Boundary | Current contract | Future adapter responsibility |
| --- | --- | --- |
| Model loading | `models.load_model(path, device)` returns `(model, metadata)` | Unpack both; preserve saved dtype, architecture and physical metadata |
| Model evaluation | One-hot `(batch, Lx*Ly, Nmax+1)` → real `(batch,)` log amplitudes | Construct tensors on the model's device and dtype; return log amplitude, not `exp(log amplitude)` |
| Occupations | Integer values `0..Nmax`; fixed-sector VMC inputs | Validate counts and total particle number without adding one to occupation values |
| Geometry | Physical zero-based site `i=y*Lx+x` | Translate Julia site identifiers explicitly; require `Lx`, `Ly`, and both boundary flags |
| VMC | One PyTorch CPU/GPU implementation in `vmc.py` | Keep sampler/SR state in Python; batch calls instead of crossing languages per site or sample |
| Observables | Physical ordering, time/walker axes retained | Preserve axis meanings and directed pair ordering when converting arrays |
| Serialization | Metadata plus `state_dict`, inference checkpoint | Do not claim exact training resume: optimizer, walkers and RNG state are absent |

`load_model` accepts an absolute read path; a relative path is resolved from
`pythonTest/2D/`. Save/training/evaluation paths are guarded to stay under 2D.
Future exporting outside this directory requires an explicit API change.

## Indexing, shapes and numerical meaning

- A Julia occupation vector already ordered by `i=y*Lx+x` can be passed as a
  sequence of occupation values. Values are counts, not one-based site indices.
- For a Julia matrix `occupation[x+1,y+1]` of shape `(Lx,Ly)`, `vec(occupation)`
  has the required x-fast site order. A matrix shaped `(Ly,Lx)` needs an explicit
  reorder. Never infer physical order from an array's memory layout alone.
- Network input is `(batch, site, occupation_channel)`. VMC samples are flat
  `(retained_time*walkers, site)`, ordered time first, then walker. Frozen chains
  retain `(retained_time, walker, site)`.
- `G[i,j]=<b_i† b_j>` is directed; for the real states here its exact expectation
  is symmetric. Raw MC estimates need not be. Hermitian bond composites and
  spatial averages must be formed before time blocking to retain covariance.
- Extract a Python scalar with `.item()`. To expose a tensor as a host array,
  make the detach/device conversion explicit, e.g. `.detach().cpu().numpy()`.
  This synchronizes/copies CUDA data and must not be hidden in a per-sample loop.
- The model returns an unnormalized real log amplitude. Normalization is within
  the chosen sector; a raw model evaluated outside it is not number projected.
- SR builds/solves in float64 on the tensor device. A float32 model still has
  float32 forward/derivative arithmetic before conversion; it is not a fully
  float64 computation. CPU and GPU share the same implementation.

## Compatibility differences to resolve before merging

| Topic | Existing 1D reference | 2D contract |
| --- | --- | --- |
| Interaction | `U_1D*n*(n-1)` in `pythonTest/BoseHubbardED.py` | `U_2D*n*(n-1)/2`; equal interactions require `U_1D=U_2D/2` |
| Short periodic axes | 1D length two includes parallel hopping bonds | Unique unordered edges; length two has one edge and length one has no self edge |
| Geometry | A chain length | Explicit `Lx,Ly`, `pbc_x,pbc_y`; mixed boundaries have metadata `pbc=null` |
| Checkpoint | Existing adapter expects 1D top-level fields | 2D `metadata` plus `state_dict`, format1/2 validation |
| SR convention | Historical uncentered/unnormalized CPU routine | Centered, sample-normalized full Jacobian; ridge primal/dual solve; native SGD |
| Python imports | Existing 1D adapter imports its own module | 2D modules currently use sibling imports from the 2D directory |

The length-two distinction is a graph convention, not an implementation bug to
hide in an adapter. Initial parity checks should use open chains or periodic
chains of length at least three, with the interaction conversion above.

## Proposed merge sequence — not executed

1. Give dimension-specific Python modules a collision-free package namespace.
   The current directory name `2D` is not an ordinary dotted-import identifier;
   temporarily inserting it into `sys.path` exposes generic names such as
   `models` and `vmc`. This is documented current behavior, not a merged namespace.
2. Add an isolated 2D inference adapter with explicit metadata dispatch. Preserve
   the existing 1D call signatures; reject ambiguous or mismatched checkpoints.
3. Establish inference parity against direct Python: six architectures, both
   rectangular and square geometries, open/periodic/mixed boundaries, float32/64,
   and save/load behavior. Test array ordering with non-square configurations.
4. Expose batched observables and then VMC operations using the same Python core.
   Match seeds only within a controlled device/environment; do not promise
   bitwise CPU/GPU equivalence. Preserve SR regularization and optimizer semantics.
5. Check tiny-sector energies and full observables against ED, plus canonical
   particle conservation, resource guards and rejected invalid inputs. Keep the
   independent Jacobian reference and dense-solve parity checks.
6. Merge only after review and the user's instruction. No root integration,
   adapter, package migration, 1D edits, or Julia round-trip test is claimed here.
