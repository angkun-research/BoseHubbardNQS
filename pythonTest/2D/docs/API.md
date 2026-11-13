# 2D Python API reference

This documents the current Python implementation, also exposed through the
Julia bridge. Embedded Python uses the `pythonTest.bosehubbard2d` namespace;
standalone scripts retain their existing paths. Exact public signatures and source
hashes are also available in [api-signatures.json](api-signatures.json).
Names beginning with `_` are implementation details. Benchmark recipes are
not an alternative public VMC API.

## Shared conventions

- `Lx,Ly` are positive integers; omitted `Ly` means `Lx`. Geometry-aware ED
  defaults to unit filling when `Ntot=None`. Site index is zero-based
  `i=y*Lx+x`. Occupation values are integers `0..Nmax` inclusive.
- `H=-t*sum_bonds(b_i†b_j+h.c.)+U/2*sum_i n_i(n_i-1)-mu*sum_i n_i`.
  Defaults are `t=1,U=16.74,mu=0,Nmax=4`, periodic in both axes.
  Axis overrides independently take precedence over `pbc`.
- Bonds are unique unordered neighbors: no self bonds; a periodic length-two
  axis contributes one edge. This differs from the existing1D short-ring code.
- Network input: one-hot floating tensor `(batch,sites,Nmax+1)`. Output:
  `(batch,)` real **unnormalized log amplitude**. Occupations passed to VMC are
  integer tensors `(samples,sites)`, not one-hot arrays.
- VMC checks a common particle total across the supplied batch. It does not
  compare that total to checkpoint metadata: the caller chooses the sector.
  Models have no fixed-N mask. Raw off-sector predictions remain unconstrained.
- Callers must put one-hot inputs on the parameter device. A model forward
  casts input dtype but does not move an existing tensor to that device; VMC
  occupation wrappers perform explicit device conversion. `load_model`
  preserves the saved floating dtype. Explicit `.double()` changes in-memory
  arithmetic, not checkpoint bytes. Julia scalar/batch inference explicitly
  returns host values, with one bulk copy for a batch. Training workflows keep
  their device computations inside the Python subprocess; see
  [JULIA_PREPARATION.md](JULIA_PREPARATION.md).
- Relative checkpoint/report paths resolve from `pythonTest/2D/`, not shell cwd.
  Guarded writes stay in2D. Inference checkpoints do not contain optimizer,
  walker or RNG state and cannot exactly resume training.

## Module contracts

### Geometry — `lattice.py`

`lattice_shape` returns `(Lx,Ly)`. `resolve_boundaries` returns `(pbc_x,pbc_y)`;
flags must be bool/None, not arbitrary truthy objects. `bonds_2d` returns a
sorted Python list of unique `(i,j)` physical indices with `i<j`.
`add_boundary_arguments` registers CLI flags and `resolve_cli_boundaries`
mutates an argparse namespace; explicit axis flags win independent of order.
Invalid integers, dimensions and boundary flags raise `ValueError`.

### Models and checkpoints — `models.py`

All seven constructors return `torch.nn.Module` instances with geometry,
`architecture_settings`, `channels`, `Nmax`, and `n_sites` attributes. Factory
`build_model` returns a model; call `.to(device=...,dtype=...).eval()` explicitly.
The factory does not normalize amplitudes or enforce total N.

| Family | Defaults and structural constraints |
| --- | --- |
| FCNet | Width64, two tanh hidden layers; `depth` aliases `hidden_layers`; conflicting nondefault values fail |
| JastrowNet | Basic displacement-shared onsite/pair tables; no hidden width; `reflection_orbit` is an optional cylinder-only research variant |
| Conv2DNet | Width64; separate circular/zero padding per periodic/open axis, then dense head |
| RBMNet | `n_hidden=64`; softplus binary-hidden expression used as log amplitude |
| TransformerNet | Width32; one head, learned x/y positions, mean pool; deterministic zero dropout |
| Conv2DTransformerNet | Width32 divisible by4; conv/SiLU → one four-head block (FF=2h) → flattened dense/SiLU head; no positional embeddings, zero dropout |
| GroupEquivariantConv2DNet | Default width16; fully periodic geometry; D4-tied lifting/SiLU → group convolution/SiLU → mean over sites and orientations → linear scalar head. D4 lattice symmetry on squares, D2 on rectangles |

Recent width32 benchmarks do not change constructor defaults. `parameter_count`
returns an integer including trainable Jastrow redundancies. `spatial_symmetries`
returns guaranteed symmetry names. `lift_jastrow_reflection_orbit` returns a
function-preserving lifted cylinder model; it is not the primary baseline.
`save_model` writes metadata + `state_dict` and returns the resolved `Path`;
it can replace an existing file. `load_model` returns `(eval_model,metadata)`
with validated Hamiltonian/bond/output conventions and format1/2 compatibility.
It reads relative2D paths or absolute paths, preserves dtype, and loads weights
strictly. Invalid architecture, settings, metadata, shape or convention fails
rather than inferring a different lattice/model. File/PyTorch errors propagate.
`BASELINE_ARCHITECTURES` keeps the original five default sweep;
`GENERAL_ARCHITECTURES` adds the hybrid for geometry-generic checks.
`ARCH_BUILDERS` also registers the periodic group CNN. It guarantees translations
and D4 invariance on squares, and translations and D2 invariance on rectangles.
The same D4-tied local filters are used on both; a quarter turn maps a rectangle
to its transposed shape, not to a state on the original lattice. It does not
number-project its raw log amplitude. `lift_features(x)` and
`forward_features(x)` return `(batch,8,hidden_dim,Ly,Lx)` regular-representation
fields; `group_features(features)` accepts the same shape and mixes all eight
relative orientations. These expose the layers for independent verification,
while `forward(x)` retains the standard `(batch,)` output contract.

### Square group — `square_group.py`

The group CNN uses eight point-group indices `r+4*m`, representing `R**r M**m`
with `R(x,y)=(-y,x)` and `M(x,y)=(x,-y)`. Composition applies the right operand
first. Coordinates on the lattice wrap modulo L; kernel offsets are centered
integer coordinates. `site_pullback` returns indices for transforming an input,
not the old-site-to-new-site permutation. `transform_features` additionally
pulls back the orientation index. See [SQUARE_SYMMETRY.md](SQUARE_SYMMETRY.md)
and the dedicated teaching slides for the layer equations and symmetry checks.

### Canonical ED — `exact_diagonalization.py`

| Function | Result |
| --- | --- |
| `count_basis_states` | Integer dimension of the bounded fixed-N sector |
| `generate_basis` | Lexicographically ordered Python list of occupation tuples |
| `basis_to_one_hot` | NumPy float32 `(dimension,sites,nmax+1)`; empty input is supported |
| `estimate_sparse_bytes` | Planning estimate in bytes, not process peak memory |
| `build_bose_hubbard_hamiltonian` | `(SciPy CSR H,basis)` with float64 matrix entries |
| `exact_ground_state` | `(float energy,normalized real NumPy state,basis,CSR H)` |

Default guards:20,000 basis states,512MiB estimated sparse memory, counting
work limit1,000,000. Guarded invalid/infeasible counts, nonfinite physical
parameters, negative t/U, excessive dimension/memory/work raise `ValueError`.
Atomic/diagonal limits select a normalized ground vector deterministically;
ground states can be degenerate there. Solver exceptions are not hidden.
Do not increase guards to turn inaccessible larger-system ED into a default.

### Shared PyTorch VMC — `vmc.py`

There is one CPU/CUDA sampler/local-energy/update implementation. It does not
have a separate NumPy CPU or duplicated GPU VMC module.

| Function | Result and effects |
| --- | --- |
| `physical_parameters` | Validated `(t,U,mu)` scalar tuple |
| `occupancy_tensor_to_onehot` | One-hot tensor on the occupation tensor's device; requested float dtype |
| `generate_initial_states` | Integer `(walkers,sites)` feasible configurations; not equilibrium samples |
| `propose_hops` | `(proposed_states,valid_mask)`; one random bond/direction per walker; invalid moves remain self proposals |
| `sample` | `(flat_samples,final_walkers,diagnostics)`; flat order is time then walker; no retained autograd graphs |
| `local_energy` | `(samples,)` local energies in the model floating dtype, using model forward ratios |
| `native_vmc_update` | Mutates parameters with supplied optimizer; returns a Python float gradient-surrogate loss, not the energy |
| `check_sr_budget` | Dict of parameter/sample/solve dimensions and estimated allocation bytes |
| `compute_log_psi_jacobian` | Full float64 `(M,P)` derivative matrix on the tensor device; parameter order follows trainable model parameters |
| `sr_gradient` | Float64 direction `(P,)`, or `(direction,diagnostics)`; no parameter update |
| `sr_update` | Assigns direction as optimizer gradients and takes one SGD step; returns direction or `(direction,diagnostics)` |

`n_steps` is retained time per walker; total M=`n_steps*n_walkers`. Burn-in and
`proposal_steps_per_sample` count local proposals per walker, not lattice sweeps.
Burn-in is excluded from diagnostics; accepted includes accepted self proposals,
whereas moved counts occupation changes. Each call rebuilds its log cache,
including after a parameter update.

The Jacobian default is `vjp`, chunk1 (scalar per-sample reference). Batched VJP
supports chunks≤64. The optimized `vmap` path supports chunks≤128; request
`jacobian_method='vmap',jacobian_batch_size=128` explicitly. All paths retain the
full Jacobian, with numerical parity checked by the regression suite. Chunking
bounds derivative workspace; it does not eliminate M×P storage.

Ridge is default: Jc=J−mean(J), Ec=E−mean(E), S=Jc.T@Jc/M,
f=Jc.T@Ec/M; `(S+tau*I)direction=f`. `auto` chooses parameter space when P≤M,
otherwise the mathematically equivalent dense sample-space solve. SGD subtracts
`lr*direction` with no extra2. All Gram/solve arithmetic is float64 on the J
device; float32 forwards/derivatives are not retroactively full precision.
Optional `regularizer='minsr'` uses the power6 soft spectral pseudoinverse;
`rpinv/apinv` set its cutoff and `tau` is unused. It is not a second VMC core.

SR requires `torch.optim.SGD`; the update checks its class, so callers must also
keep momentum/weight-decay at their native zero defaults to match the documented
plain-SGD protocol. Native energy-gradient updates have their own factor2 and
can use the supplied optimizer; do not confuse that gradient with the SR force.
Default SR guards:1,000,000 parameters and512MiB estimated workspace. Invalid
states/options, mismatched lengths and guard failures raise `ValueError`;
nonfinite log amplitudes, local energies, derivatives or directions raise
`RuntimeError`. Dense solver/device errors propagate. Guards exclude forward
activations and vendor workspace and are not a hard process memory ceiling.

### Frozen observables — `observables.py`

`sample_frozen` returns `(chains,final_walkers,diagnostics)`, reshaping retained
states to `(T,C,sites)`. It does not update parameters. `one_body_estimators`
returns raw float64 `(T,C,sites,sites)` for `pairs=None`, or `(T,C,K)` for K
explicit directed pairs in supplied physical order. Diagonals equal occupations;
off-diagonals use row-action ratios sqrt(n_i*(n_j+1))*exp(f(n−e_i+e_j)−f(n)),
zero unless n_i>0 and n_j<Nmax. Neither triangle is silently symmetrized.

`energy_from_observables` reconstructs energy with arbitrary leading sample
axes; selected pairs must include both directions of every Hamiltonian edge.
It raises rather than filling missing directions by assumed Hermiticity.
`check_observable_budget` returns allocation estimates (default512MiB limit).

`measure_observables` returns a dict with raw `density_samples`,
`onsite_factorial_moment_samples`, `one_body_samples`, ordered
`one_body_pair_samples`/`one_body_pairs`, `local_energy_samples`, optional
`reconstructed_energy_samples`, arithmetic means, diagnostics, budget, and
optional `statistics`. `one_body_matrix` is a mean matrix only for `pairs=None`.
The site/pair means are **unweighted arithmetic means of supplied chains**;
feeding enumerated configurations does not automatically weight by |psi|².

`block_mean_statistics` takes `(time,walker,...)` and returns tensor means,
block/walker SEMs, block means, used/discarded times and assumptions. It never
flattens chains before blocking; fewer than two complete blocks per walker
provides no block SEM. The report analysis uses max(blockSEM,between-chainSEM).
Form Hermitian bonds/spatial composites within each sample before blocking;
adding independently estimated errors discards their covariance. D=<n(n−1)>
is a factorial moment, not Var(n). Invalid shapes/pairs/budgets fail explicitly.

### Supervised ED — `train_ed.py`

`make_ed_dataset` returns `x,target_log,raw_ed_state,H,basis,energy,metadata` in
a dict; x is a Torch one-hot tensor, target_log a Torch log-target vector,
raw_ed_state a normalized NumPy vector, H SciPy CSR. Only the log target uses
`eps` flooring. Exact zeros and mixed-sign targets are rejected, rather than
hiding them by abs/clamping. A common global negative sign is permitted.
This positive-amplitude training requires t>0; it is stricter than ED's t≥0.

`normalized_log_mse` and `normalized_infidelity` return differentiable scalar
Torch losses. `compare_predictions` returns both losses, labeled gradient
norms, raw-ED absolute/squared overlap, energy, variance and gate fields;
it does not train. A single metrics call can temporarily build derivative
graphs. The larger-reference overlap gate applies to sites≥9; strict tiny
joint-gate fields remain present and separate.

`train_model_on_ed` uses Adam and returns `(model,report,best_state_dict)`.
The model retains final-epoch weights; the returned best state is selected by
strict full-sector selected-loss minimum including initialization, with earliest
ties. Do not assume it has already been loaded into the returned model. The CLI
loads/saves that selected state. Primary loss is log-MSE; optional infidelity
is separately labeled. No threshold/plateau early stopping is performed.

### Raw sector extension — `leakage_analysis.py`

`state_batches` returns `(batch_iterator,{N:dimension})` with NumPy integer
occupations and a default2,000,000-state enumeration guard. `evaluate_logs`
returns NumPy float64 log values using the model's current forward precision.
`summarize_sector_weights` returns window probabilities/log weights and sector
dimensions; it does not on its own assert global normalization.
`analyze_sectors` streams exact enumeration, returns that summary plus timing/
precision, and adds global mass only when all truncated particle sectors were
covered. It performs no ED, Hamiltonian construction or training.

`projected_sector_probabilities` returns a delta dict at N0. `apply_number_gauge`
returns a reweighted result without mutating model parameters. `full_overlap`
returns sqrt(Pfull(N0))*F_N0 for absolute unsquared overlap with a zero-embedded
canonical target. Window-conditional probabilities cannot replace global mass;
raw model output is not itself a projected physical state. Guard/invalid input
errors are explicit; finite arithmetic can underflow tiny probabilities to0,
so retain log weights as well.

## Entry points and side effects

| Entry point | Main behavior |
| --- | --- |
| `train_ed.py` | ED supervision with Adam; saves selected inference checkpoint and JSON report |
| `train.py` | Native or SR VMC training; native SGD required for SR; saves fixed final weights/sidecar |
| `evaluate.py` | Frozen evaluation; writes JSON plus `.json.samples.npz`, refuses existing destinations |
| `dmrg_reference.jl` | Optional ITensor DMRG reference with explicit physical/MPS ordering; bounded geometry/bond/sweep controls |

The Python `parse_args(argv)` helpers validate CLI namespaces; `main(argv)`
functions execute file-writing workflows and are not pure inference APIs. `train_ed.py`
rejects existing output paths. Cold-start `train.py` can replace an existing
checkpoint/sidecar; warm-start runs require fresh output paths. Use
`--help` for parser defaults; [BENCHMARKING.md](../BENCHMARKING.md) gives current
commands. Checkpoint and report replacements are not one atomic cross-file
transaction. The Julia integration remains future work.

## Read-only CPU example

Run this in a fresh Python process from the repository root. It enumerates the
existing tiny hybrid checkpoint's sector and **probability-weights** the raw
operator estimators. It neither samples nor trains, and writes no files.

```python
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path("pythonTest/2D").resolve()))
from models import load_model
from exact_diagonalization import generate_basis
from vmc import occupancy_tensor_to_onehot, local_energy
from observables import one_body_estimators

torch.set_num_threads(1)
model, meta = load_model(
    "outputs/2026-11-01-backup-extension/Hybrid32-2x3-sr500-seed123.pt",
    device="cpu",
)
model = model.double().eval()
states = torch.tensor(generate_basis(model.n_sites, meta["Ntot"], model.Nmax))
with torch.no_grad():
    logs = model(occupancy_tensor_to_onehot(states, model.Nmax, torch.float64))
    weights = torch.softmax(2 * logs, dim=0)
    density = weights @ states.double()
    # Here the time/walker axes merely batch all basis rows, not an MC sample.
    raw_G = one_body_estimators(model, states[:, None, :])[:, 0]
    G = torch.einsum("s,sij->ij", weights, raw_G)
    E = weights @ local_energy(model, states, meta["t"], meta["U"], meta["mu"])
print(float(E), density.tolist(), G.shape)
```

Expected energy is approximately−2.580816327645t, with sum(density)=6 and a6×6
one-body matrix whose diagonal equals density. “Exact” here means full-sector
floating-point expectation of that saved NQS, not equality with the ED ground
state. Monte Carlo measurement instead supplies samples from |psi_N|² and uses
their arithmetic averages and blocked errors.

## Complete public signatures

The following list is extracted from the current source AST. Constructor
signatures omit `self` for caller readability; the JSON preserves literal AST
signatures. Symbolic defaults resolve to the module guard values stated above.
Private helpers, benchmark scripts and Julia internals are not promoted to APIs.

### lattice.py

Source: [implementation](../lattice.py).

```python
lattice_shape(Lx, Ly=None)
```

Validate dimensions; an omitted Ly gives a square lattice.

```python
resolve_boundaries(pbc=True, pbc_x=None, pbc_y=None)
```

Resolve each optional axis override against the legacy both-axis flag.

```python
add_boundary_arguments(parser)
```

Shared CLI flags; overrides win regardless of global-flag ordering.

```python
resolve_cli_boundaries(args)
```

Resolve flags without changing the historical fully periodic default.

```python
bonds_2d(Lx, Ly=None, pbc=True, *, pbc_x=None, pbc_y=None)
```

Return sorted unique site pairs for open or periodic boundaries.

### models.py

Source: [implementation](../models.py).

```python
FCNet(Lx, Ly=None, Nmax=4, pbc=True, hidden_dim=64, *, hidden_layers=2, depth=None, pbc_x=None, pbc_y=None)
```

Tanh MLP with positive hidden_layers (default two); depth is an alias.

Default depth preserves the legacy net.0/net.2/net.4 state_dict keys.
depth overrides the default hidden_layers=2; conflicting nondefault values
are rejected. Only FCNet changes capacity through this setting.

```python
JastrowNet(Lx, Ly=None, Nmax=4, pbc=True, *, pbc_x=None, pbc_y=None, pair_sharing='displacement')
```

Onsite table plus all unordered pairs, shared by absolute (dx, dy).

Periodic distances use the minimum along each axis; open distances do not wrap.
Default displacement tables are averaged with their transpose. Optional
reflection_orbit is restricted to open-x/periodic-y cylinders: onsite tables
use x-reflection orbits, pairs use reflected unordered x pairs and minimal
periodic |dy|. Reflected canonicalization swaps occupation roles. Only same-x
or self-reflection pair classes are transpose-symmetric; edge/bulk tables
retain ordered occupation roles. Pair buffers avoid Python forward loops;
gather storage scales as batch * number_of_site_pairs.

```python
lift_jastrow_reflection_orbit(model)
```

Lift displacement cylinder weights exactly; no optimizer or jitter.

Old symmetric pair tables are copied by (dx, minimal periodic |dy|).
Non-self-reflection orbits subsequently allow ordered occupation tables.

```python
Conv2DNet(Lx, Ly=None, Nmax=4, pbc=True, hidden_dim=64, *, pbc_x=None, pbc_y=None)
```

One 3x3 convolution and a dense head; no translation invariance.

Each axis is padded separately, circular when periodic and zero otherwise,
including length-one axes. Zero feature padding is not an occupation-zero
one-hot outside site. Legacy convolution/head state_dict keys are unchanged.

```python
RBMNet(Lx, Ly=None, Nmax=4, pbc=True, n_hidden=64, *, pbc_x=None, pbc_y=None)
```

Binary-hidden RBM expression interpreted directly as log amplitude.

```python
TransformerNet(Lx, Ly=None, Nmax=4, pbc=True, hidden_dim=32, feedforward_dim=None, *, pbc_x=None, pbc_y=None)
```

One non-autoregressive, one-head attention block; learned x/y positions.

```python
Conv2DTransformerNet(Lx, Ly=None, Nmax=4, pbc=True, hidden_dim=32, *, pbc_x=None, pbc_y=None)
```

2D generalization of the 1D ConvTransformer, for optional backup studies.

A boundary-aware 3x3 convolution and SiLU produce physical-order tokens
i=y*Lx+x. One four-head encoder uses FF width 2h and ReLU, followed by
the original flattened dense/SiLU/scalar head. There are no positional
embeddings or mean pooling. The flattened head imposes no translation
invariance. Unlike the 1D module's framework-default dropout=0.1, dropout
is fixed to zero for deterministic VMC forwards in train and eval mode.

```python
GroupEquivariantConv2DNet(Lx, Ly=None, Nmax=4, pbc=True, hidden_dim=16, *, pbc_x=None, pbc_y=None)
```

Periodic D4-tied regular-representation CNN with invariant scalar readout.
Both square and rectangular fully periodic geometries are accepted, with D4
and D2 lattice symmetry respectively. Feature methods are
documented above; no particle-number mask is applied by the model.

```python
build_model(arch, Lx, Ly=None, Nmax=4, pbc=True, *, pbc_x=None, pbc_y=None, **settings)
```

```python
parameter_count(model)
```

Raw trainable parameter count, including Jastrow redundancies.

```python
spatial_symmetries(model)
```

Guaranteed symmetries; additional accidental ones depend on parameters.

```python
save_model(path, model, Ntot=None, t=1.0, U=16.74, mu=0.0)
```

Save an inference checkpoint under 2D/ (relative paths start there).

Stores no optimizer, random-generator, or walker state; not an exact training
resume checkpoint. Path resolution rejects symlink escapes before writing.

```python
load_model(path, device='cpu')
```

Return (eval-mode model, metadata), preserving saved floating dtype.

### square_group.py

```python
d4_product(left, right)
d4_inverse(element)
d4_coordinate(element, x, y)
kernel_pullbacks()
site_pullback(L, element, tx=0, ty=0, device="cpu")
space_group_pullbacks(L)
transform_features(features, element, tx=0, ty=0)
```

`d4_product` and `d4_inverse` return integer group indices, while
`d4_coordinate` returns an integer coordinate pair. Group-index inputs to these
three low-level algebra helpers are assumed to be in `0..7`.
`kernel_pullbacks` returns a CPU long tensor `(8,9)` containing centered-stencil
pullback indices. `site_pullback` validates L and the group index and returns
an `(L*L,)` long tensor on the requested device. `space_group_pullbacks` stacks
all `8*L*L` such rows in `(orientation,ty,tx)` order; the action need not be
faithful for very small L. `transform_features` maps a square
`(batch,8,channels,L,L)` tensor to a tensor of the same shape and device, with
site and orientation indices both pulled back. It preserves autograd.

### exact_diagonalization.py

Source: [implementation](../exact_diagonalization.py).

```python
count_basis_states(n_sites, Ntot, Nmax=4)
```

Exact bounded-sector count, using integer DP before enumerating states.

Particle-hole symmetry reduces work near full filling. No basis is allocated.
At most MAX_COUNTING_WORK DP cells may be processed; larger requests raise
ValueError before allocating the particle-count-sized workspace.

```python
generate_basis(n_sites, Ntot, Nmax=4, max_dimension=DEFAULT_MAX_DIMENSION)
```

Enumerate bounded fixed-number occupation tuples in lexicographic order.

```python
basis_to_one_hot(basis, n_sites, nmax)
```

Return float32 one-hot features shaped (dimension, n_sites, nmax+1).

```python
estimate_sparse_bytes(dimension, n_sites, n_bonds)
```

Conservative planning estimate, not a hard process-memory bound.

Allow one diagonal and two directed hops per bond per column, even when
occupation constraints forbid them. Budget 64 bytes per possible entry for
COO arrays, CSR conversion and temporaries; also budget Python basis/index
storage and 32 float64 eigensolver vectors. No dense matrix is included.

```python
build_bose_hubbard_hamiltonian(Lx, Ly=None, Ntot=None, Nmax=4, t=1.0, U=16.74, mu=0.0, pbc=True, max_dimension=DEFAULT_MAX_DIMENSION, max_sparse_bytes=DEFAULT_MAX_SPARSE_BYTES, *, pbc_x=None, pbc_y=None)
```

Return (CSR Hamiltonian, basis); guard allocations before enumeration.

Defaults are a square periodic lattice at unit filling. Both dimension and
estimated sparse-memory limits must pass; increase either explicitly if needed.
Counting also has a fixed MAX_COUNTING_WORK budget before workspace allocation.

```python
exact_ground_state(Lx, Ly=None, Ntot=None, Nmax=4, t=1.0, U=16.74, mu=0.0, pbc=True, max_dimension=DEFAULT_MAX_DIMENSION, max_sparse_bytes=DEFAULT_MAX_SPARSE_BYTES, *, pbc_x=None, pbc_y=None)
```

Return (energy, normalized real state, basis, CSR Hamiltonian).

### vmc.py

Source: [implementation](../vmc.py).

```python
physical_parameters(t=1.0, U=16.74, mu=0.0)
```

```python
occupancy_tensor_to_onehot(states, Nmax, dtype=torch.float32)
```

```python
generate_initial_states(n_sites, Ntot, Nmax=4, n_walkers=64, device='cpu')
```

Generate feasible integer walkers; initialization is not a target sample.

```python
propose_hops(states, bonds, Nmax)
```

Return (proposals, valid-hop mask); choose all bonds/directions uniformly.

States and bonds are device-resident long tensors, with unique non-self bonds.
Invalid hops and the no-bond case return self proposals, never resampled moves.

```python
sample(model, initial_states, n_steps, burnin_steps=0, evaluation_batch_size=256, *, proposal_steps_per_sample=1)
```

Return (samples, final walkers, diagnostics), flattened step then walker.

Burn-in proposals are excluded from diagnostics and samples. Accepted includes
accepted self proposals; moved counts only accepted occupation changes. Caches
are local to this call and cannot survive parameter updates.
n_steps counts RETAINED samples per walker. After burnin_steps proposals,
advance proposal_steps_per_sample proposals between retained configurations.
Diagnostics count every post-burn proposal, including discarded intervals.
The default stride one preserves the previous seeded trajectory exactly.

```python
local_energy(model, states, t=1.0, U=16.74, mu=0.0, evaluation_batch_size=256)
```

Detached H psi / psi with U/2, bounded connected-state allocations.

Traverse directed bonds sequentially inside each sample block. At most one
evaluation batch of connected occupations exists at a time, rather than a
batch times coordination-number expansion. Returns one energy per state.

```python
native_vmc_update(model, states, energies, optimizer, evaluation_batch_size=256)
```

Accumulate 2*mean((E-Emean).detach()*logpsi) gradients per chunk.

```python
check_sr_budget(model, n_samples, max_sr_parameters=DEFAULT_MAX_SR_PARAMETERS, max_sr_bytes=DEFAULT_MAX_SR_BYTES, *, solver='auto', jacobian_batch_size=1, jacobian_method='vjp', regularizer='ridge')
```

Guard before allocating J or float64 dense solve arrays.

Budget three MxP float64 Jacobians (input/conversion/centered), eight DxD
float64 arrays for covariance, regularization, solver copies/workspace, and
vectors, D=P or M according to solver. Auto selects min(P,M). The explicit
byte budget guards M*P even for sample-space solves. The generous parameter
cap is a separate fail-safe, not a substitute for this allocation guard.
Optional MinSR budgets eight additional DxD arrays plus spectral vectors
before any Jacobian/eigh allocation. This conservative estimate does not
bound model activations, all vendor-library memory, or total process memory.

```python
compute_log_psi_jacobian(model, states, max_sr_parameters=DEFAULT_MAX_SR_PARAMETERS, max_sr_bytes=DEFAULT_MAX_SR_BYTES, *, solver='auto', jacobian_batch_size=1, jacobian_method='vjp', regularizer='ridge')
```

Guarded log-amplitude Jacobian; default one-sample reference is unchanged.

B>1 evaluates B inputs once and uses is_grads_batched=True with an identity
cotangent matrix, returning B parameter VJPs. Attention explicitly uses the
math SDPA backend because fused attention backward lacks batching rules.
Guards include returned VJP/cotangent arrays, not model activation memory;
intermediate backward work can scale quadratically in B. Cap B at 64.
Opt-in vmap(grad(functional_call)) differentiates one configuration at a time,
vectorized in chunks up to 128. Guards include returned gradient/input blocks;
model activations and total process memory are not bounded by this estimate.

```python
sr_gradient(jacobian, energies, tau=0.001, max_sr_parameters=DEFAULT_MAX_SR_PARAMETERS, max_sr_bytes=DEFAULT_MAX_SR_BYTES, *, solver='auto', return_diagnostics=False, regularizer='ridge', rpinv=1e-12, apinv=0.0)
```

Centered, sample-normalized imaginary-time direction with no extra 2.

Call check_sr_budget before constructing the input Jacobian. The training API
does this automatically; the solve also checks before allocating workspace.
All covariance and solve arithmetic uses float64.
Sample space: delta=Jc.T @ solve(Jc@Jc.T + M*tau*I, E-Emean).
It is exactly equivalent to the parameter-space ridge direction, with no
extra factor two. Auto solves in the smaller space; it still stores J.
Optional real MinSR uses the Methods20-23 soft spectral pseudoinverse from
Chen/Heyl (2024), power6, rather than a ridge. Gram formation, eigh and
backprojection are float64; tau is unused. The equivalent parameter-space
form applies the same filter to A.T@A and then A.T@e. No extra truncation.

```python
sr_update(model, states, energies, optimizer, tau=0.001, max_sr_parameters=DEFAULT_MAX_SR_PARAMETERS, max_sr_bytes=DEFAULT_MAX_SR_BYTES, *, solver='auto', return_diagnostics=False, jacobian_batch_size=1, jacobian_method='vjp', regularizer='ridge', rpinv=1e-12, apinv=0.0)
```

Assign the guarded imaginary-time direction to optimizer gradients.

### observables.py

Source: [implementation](../observables.py).

```python
check_observable_budget(n_samples, n_sites, n_pairs, max_bytes=DEFAULT_MAX_OBSERVABLE_BYTES)
```

Conservative float64 storage estimate, excluding model/forward workspace.

```python
sample_frozen(model, initial_states, n_steps, burnin_steps=0, evaluation_batch_size=256, *, proposal_steps_per_sample=1)
```

Wrap the stable sampler, returning occupations as (time, walker, site).

```python
one_body_estimators(model, chains, pairs=None, evaluation_batch_size=256, max_observable_bytes=DEFAULT_MAX_OBSERVABLE_BYTES)
```

Return raw row-action estimators, without averaging or symmetrization.

pairs=None explicitly requests full (T,C,N,N) output, intended for tiny
references. Selected physical pairs give (T,C,K), in the supplied order.
At most one evaluation batch of connected configurations is constructed.

```python
energy_from_observables(density, onsite_factorial_moment, one_body, Lx, Ly=None, t=1.0, U=16.74, mu=0.0, pbc=True, *, pbc_x=None, pbc_y=None, pairs=None)
```

Linear reconstruction from raw rho and <n(n-1)>; accepts sample axes.

A selected-pair reconstruction requires both directions of every bond.
Missing entries raise; they are never filled by assuming Hermiticity.

```python
block_mean_statistics(values, block_length)
```

Nonoverlapping time blocks within each walker; never flatten before blocking.

The block SEM assumes stationary, sufficiently decorrelated blocks and
independent walkers. Between-chain SEM is also reported as a diagnostic.
Neither checks mixing or estimates ESS. Tail steps are explicitly reported;
the blocked mean/error use complete blocks, with an all-sample mean separate.
Fewer than two complete blocks per chain yields no block error estimate.

```python
measure_observables(model, chains, t=1.0, U=16.74, mu=0.0, evaluation_batch_size=256, *, pairs=None, block_length=None, max_observable_bytes=DEFAULT_MAX_OBSERVABLE_BYTES)
```

Frozen density, onsite moment, selected/full rho and energy with raw checks.

### train_ed.py

Source: [implementation](../train_ed.py).

```python
normalized_log_mse(pred_log, target_log)
```

Uniform MSE in normalized log amplitudes over the supplied full sector.

```python
normalized_infidelity(pred_log, raw_ed_state)
```

Infidelity to the raw ED vector; no target flooring or minibatch norms.

```python
make_ed_dataset(Lx, Ly=None, Ntot=None, Nmax=4, t=1.0, U=16.74, mu=0.0, pbc=True, eps=1e-12, dtype=torch.float64, max_dimension=DEFAULT_MAX_DIMENSION, max_sparse_bytes=DEFAULT_MAX_SPARSE_BYTES, *, pbc_x=None, pbc_y=None)
```

Return one-hot x, target_log, raw_ed_state, sparse H, basis and metadata.

The return value is a dictionary. Flooring affects only target_log. Reject
exact zeros or mixed signs rather than hiding them behind abs/clamp. A
globally negative ED vector is an equivalent physical global phase.

```python
compare_predictions(model, dataset, loss='log_mse')
```

Report both objectives, labeled gradients and physical raw-ED metrics.

Legacy gradient_norm remains the log-MSE gradient norm for either objective.
Working larger-reference assessment applies to n_sites>=9. This threshold
is a reporting convention, not a physics criterion.

```python
train_model_on_ed(model, dataset, lr=0.001, epochs=2000, device='cpu', eval_every=100, loss='log_mse')
```

Return (final model, JSON-ready report, best-by-loss CPU state_dict).

Adam only, full batch. Select the lowest loss among initialization and every
post-update state, not only reporting epochs. Fixed epoch budget; no early
stopping, including when the physical fit gate passes or loss plateaus.

```python
parse_args(argv=None)
```

```python
main(argv=None)
```

### leakage_analysis.py

Source: [implementation](../leakage_analysis.py).

```python
state_batches(n_sites, Nmax, sectors=None, batch_size=2048, max_states=DEFAULT_MAX_STATES)
```

Return (iterator, exact sector dimensions), validated before iteration.

Full basis uses vectorized base counting, last physical site fastest. Nearby
sectors use bounded lexicographic DFS. At most one state/one-hot batch lives
in memory, apart from model inference workspace. No full basis is retained.

```python
evaluate_logs(model, occupations)
```

Finite real logs on a supplied integer batch, without sector restriction.

```python
summarize_sector_weights(logZ, dimensions, N0, window)
```

Separate window-conditional probabilities from genuine full-cutoff mass.

```python
analyze_sectors(model, N0, sectors=None, *, window=None, batch_size=2048, max_states=DEFAULT_MAX_STATES)
```

Exact enumeration in floating-point arithmetic, no H/ED/optimizer calls.

```python
projected_sector_probabilities(sectors, N0)
```

Exact delta projection, not a finite negative sentinel in log space.

```python
apply_number_gauge(result, a)
```

Analytic f->f+a(N-N0), with no model mutation or new forward pass.

```python
full_overlap(F_N0, P_full_N0)
```

Absolute unsquared overlap against the zero-embedded canonical target.

### train.py

Source: [implementation](../train.py).

```python
parse_args(argv=None)
```

```python
main(argv=None)
```

### evaluate.py

Source: [implementation](../evaluate.py).

```python
main(argv=None)
```
