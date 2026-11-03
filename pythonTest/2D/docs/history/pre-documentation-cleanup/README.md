# Two-dimensional Bose–Hubbard NQS

This directory is self-contained. Source, tests, checkpoints, logs, and benchmark
records stay here; existing 1D code is not imported by the implementation.
The 1D parity test reads the existing ED reference without changing it.

Current results and known qualification gaps are summarized in [STATUS.md](STATUS.md).
For the runnable, bounded cylinder workflow, see [BENCHMARKING.md](BENCHMARKING.md).

Research workflow: qualify each architecture by **ED-supervised training with
Adam first**, then use **SR with native SGD** for scaling through VMC. The
earlier VMC/energy-minimization results below are exploratory and do not establish
supervised expressiveness. The initial supervised gate requires absolute overlap
at least `0.9999` and energy error per site at most `1e-3*t`; see [PLAN.md](PLAN.md).
The confirmed working larger-reference assessment (`n_sites>=9`, including
rectangles) instead uses **absolute unsquared overlap >=0.99**; energy and
variance are reported separately. Historical strict joint-gate fields and raw
records remain unchanged.
SR plus Adam is rejected by the CLI. A failed finite-budget supervised fit must
be checked for optimization and width dependence before claiming a capacity limit.

## Physical conventions

```text
H = -t sum_<i,j> (b_i† b_j + b_j† b_i)
    + (U/2) sum_i n_i(n_i-1) - mu sum_i n_i
i = y*Lx + x;  0 <= n_i <= Nmax;  sum_i n_i = Ntot
```

Defaults are `t=1`, `U=16.74`, inclusive `Nmax=4`, `mu=0`, unit filling,
and periodic boundaries on both axes. The core APIs and training CLI use
`Ly=Lx` when omitted. Legacy `pbc` applies to both axes; independent `pbc_x` and
`pbc_y` overrides are supported throughout geometry, ED, all models and VMC.
All existing trainers and general benchmark launchers retain fully periodic
defaults. `t` and `U` must
be finite and nonnegative; `mu` must be finite. Within a fixed number sector,
changing `mu` only shifts energy by `-mu*Ntot`.

Boundary precedence is explicit and independent of command-line ordering:
`--pbc`/`--no-pbc` first set both axes; `--pbc-x`/`--no-pbc-x` and
`--pbc-y`/`--no-pbc-y` then override their respective axes. Thus `--no-pbc
--pbc-y` always means open x/periodic y, even with the flags reversed. Conflicting
positive/negative flags for the same axis are rejected. No single boolean
implies a cylinder. New cylinder studies pass `--no-pbc-x --pbc-y` explicitly
(x long/open, y short/periodic); tori pass `--pbc` or omit all boundary flags.
In CLI settings `pbc` is the legacy fallback, with resolved `pbc_x/pbc_y`
authoritative. Model/checkpoint physical metadata uses `pbc=null` for mixed
boundaries and always records both explicit axis flags in format 2.

FCNet accepts `hidden_layers` (default 2) or the `depth` alias. Trainers and
launchers expose `--hidden-layers`/`--depth`; widths retain their existing
options. Counts must be positive integers. Other four architectures do not
change depth. Format-2 checkpoints record FC hidden-layer count; format-1
checkpoints load their boolean `pbc` onto both axes and default missing FC depth
to two, preserving `net.0/net.2/net.4` weights and uniform-boundary predictions.
Axes are padded separately in Conv2DNet; Jastrow distances wrap only along
periodic axes and symmetry metadata lists the supported translations per axis.

The proof-of-principle cylinder sequence is 4x4, then 6x4, at most 8x4 initially;
stop growth when energy/site and matched bulk correlations change little within
reasonable uncertainty. The first DMRG reference uses fixed chi 128 with one chi
256 spot-check on the smallest cylinder and modest sweeps. Larger sizes and
width-six studies need evidence rather than an automatic convergence campaign.
Milestone A covered boundary/depth compatibility only. Subsequent sections
document guarded SR and frozen observables added in milestones B/C; the DMRG
reference is maintained as a separate implementation/review task.

Each unordered nearest-neighbor bond is included once, with both hopping
directions. Length-one axes have no self-bonds. A periodic length-two axis has
one bond, whereas the existing 1D code counts two parallel bonds. Consequently,
2x2 and 2x3 periodic grids have reduced coordination relative to a larger square
torus. This convention matters for tiny reference calculations. The 1D interaction
coefficient also differs: compare using `U_1D=U_2D/2`, on open chains or periodic
chains of length at least three.

Periodic wrapping on 3x3 creates odd cycles, so that finite graph is not
bipartite, although the real nonnegative-hopping Hamiltonian remains suitable
for positive ground-state amplitudes. Even-size sequences starting at 4x4 are
the main route for larger-system physics; tiny and odd-size grids serve as
finite-system checks rather than interchangeable square-lattice scaling points.

`U/t=16.74` is a selected study point. Small finite lattices with an occupation
cutoff do not establish a thermodynamic critical coupling. Establish energy and
cutoff convergence before interpreting observables or critical behavior.

## Validation of the ED reference

ED was initially validated by complementary construction and solve checks:

- Atomic limits check the onsite diagonal `U*n*(n-1)/2 - mu*n` directly.
- For one particle, the unique-edge periodic 3x2 graph has ground energy
  `-3*t-mu`; the open 3x2 graph gives `-t*(1+sqrt(2))-mu`.
- On one bond with two bosons and `Nmax>=2`, the ground energy is
  `(U-sqrt(U**2+16*t**2))/2 - 2*mu`, checking both the interaction factor and
  bosonic hopping coefficients.
- An independent tensor-product ladder-operator Hamiltonian, projected into a
  fixed-number sector, checks the complete tiny-system matrix. Small dense
  eigensolves independently check sparse ground energies and eigenvectors.
- Hermiticity, particle-sector conservation, normalization, chemical-potential
  shifts and eigenvector residuals provide numerical consistency checks.

The initial 3x3 residual norm was approximately `5.2e-14`. A residual establishes
that a vector solves the assembled matrix; the independent operator and analytic
checks establish the matrix conventions. Occupation-cutoff convergence is a
separate question: the measured 3x3 ED energy change from `Nmax=4` to the full
nine-particle cutoff is approximately `3.25e-6`. These checks do not constitute
a thermodynamic benchmark.

The subsequent matched-cylinder DMRG implementation is
[dmrg_reference.jl](dmrg_reference.jl). On the 3x3 open-x/periodic-y cylinder,
its chi-128 energy differs from independently assembled ED by `4.27e-10*t`,
and its full one-body matrix differs by at most `3.08e-9`. The ED cylinder
energy is `-4.385696750796594*t`. On 4x4, DMRG gives
`-7.671673531665546*t` at chi 128 and `-7.671699717843744*t` at chi 256;
the energy/site change is `1.64e-6*t` and maximum bulk one-body change is
`1.67e-5`. This is a useful modest reference, with finite-chi sensitivity
reported separately from NQS sampling errors. See the versioned reports
`benchmarks/results/dmrg-3x3-cylinder-validation-physical-geometry.json` and
`benchmarks/results/dmrg-4x4-cylinder-chi-sensitivity.json`.

For the supervised reference, a separate wavefunction cutoff check embeds the
3x3 `Nmax=4` ED state into the complete nine-particle basis. Its absolute overlap
with the full ED state is `0.99999999015`, and the full state's probability
inside the cutoff-four subspace is `0.99999998037`.
[Raw cutoff records](benchmarks/results/ed-cutoff-wavefunction-check.json) include
the corresponding 2x3 check and explicitly increased full-solve dimension guard.

## Models

All models accept one-hot `(batch, Lx*Ly, Nmax+1)` inputs and return real,
unnormalized log amplitudes `(batch,)`. Positive amplitudes suit the unfrustrated
ground-state problem. Models do not enforce number conservation outside the
sector used by sampling/evaluation. Normalize only within that sector.

| Architecture | Design and guaranteed spatial symmetries |
| --- | --- |
| `FCNet` | Flattened features, configurable tanh hidden layers (default two), scalar head; no imposed spatial invariance. |
| `JastrowNet` | Shared onsite table and displacement-shared occupation-pair tables, symmetrized under occupation exchange. Independent x/y reflections; translations along each periodic axis. Axes remain distinct, even on a square. No imposed axis-exchange symmetry. |
| `Conv2DNet` | One 3x3 convolution, SiLU, flatten and dense head; separate circular/zero feature padding on each periodic/open axis. The dense head does not enforce translation invariance. |
| `RBMNet` | Flattened one-hot visible units, softplus binary-hidden expression interpreted as log amplitude; no imposed spatial invariance. |
| `TransformerNet` | Occupation projection, learned x/y positions, one encoder block with one attention head, mean pooling and scalar head. Dropout zero; deterministic in train and eval modes; no autoregression or imposed spatial invariance. |
| `Conv2DTransformerNet` | 2D extension of the existing 1D ConvTransformer: boundary-aware 3×3 convolution and SiLU, one four-head encoder (FF width twice hidden width, ReLU, dropout zero), flattened dense/SiLU scalar head. No positional embeddings, mean pooling or imposed translation invariance. Default width32; width must be divisible by four. |

The original five architectures remain the default benchmark sweep. The hybrid
is an explicitly selected backup study; use `--arch Conv2DTransformerNet` in
either trainer. Its deterministic dropout setting is intentional for VMC.

Default Jastrow distances are minimal absolute x/y displacements for periodic boundaries
and ordinary absolute displacements for open boundaries. Only occurring
displacements are parameterized. Sharing reduces capacity versus independent
site-pair tables and excludes higher-body terms in log amplitude. Raw parameter
counts include redundant antisymmetric table directions. Pair evaluation is
vectorized, with storage proportional to batch size times site-pair count.

Optional `--pair-sharing reflection_orbit` in both trainers is restricted to
open-x/periodic-y cylinders. Onsite tables use `min(x,Lx-1-x)`; pair tables use
unordered x-coordinate pairs modulo global x reflection and minimal periodic
`|dy|`. Canonical reflection swaps occupation roles. Only same-x and
self-reflection classes are transpose-symmetrized; other edge/bulk classes keep
ordered occupation tables. This relaxes excess edge/bulk coefficient tying while
preserving physical x/y reflections and periodic-y translations. It adds no
higher-body terms. `models.lift_jastrow_reflection_orbit` exactly lifts old
displacement-cylinder weights without optimizer state or perturbation. New
checkpoints record the mode explicitly; old missing-mode checkpoints load as
displacement with unchanged state keys/predictions. Independent group-action,
lift, Jacobian and legacy checks are retained in
`tests/test_jastrow_orbits.py` and the
[checker qualification review](benchmarks/results/2026-10-31-all-model-refinement/checker-jastrow-orbit-qualification-review.json).

## ED-supervised fitting: first gate

Use `train_ed.py` to test supervised fitting before launching new VMC campaigns.
Its CLI defaults to 2x2 and resolves omitted `--Ly` to `--Lx`, matching the API.
Request the 2x3 diagnostic explicitly with `--Lx 2 --Ly 3`; the benchmark launcher
retains its explicit 2x3 default. Existing campaigns pass both dimensions, so
this default correction does not change their inputs. Per-run source hashes
identify the trainer version used.
Training uses Adam only and one full fixed-number sector. The primary and
default objective is `--loss log_mse`, with the loss

```text
pred   = logpsi - 0.5*logsumexp(2*logpsi)
rawlog = log(abs(psi_ED).clamp_min(eps))
target = rawlog - 0.5*logsumexp(2*rawlog)
loss   = mean((pred-target)^2) over the ENTIRE sector
```

The default is float64 with `eps=1e-12`; reports count clamped targets explicitly.
There is no independent minibatch normalization. Runs require positive `t` and
reject zero or mixed-sign ED targets. Physical energies, variances and absolute
**unsquared** overlaps use the raw ED state, not the floored loss target.
The joint gate requires **both** overlap `>=0.9999` and energy error/site
`<=0.001*t`. Failure within a finite budget does not establish a capacity limit.
Earlier VMC tables and energy-minimization runs are exploratory evidence only.

Experimental `--loss infidelity` is available in both `train_ed.py` and the
launcher. It minimizes `1-|dot(normalized_model_amplitudes, raw_normalized_ED)|^2`,
using global logsumexp normalization and no target flooring. Adam plus log-MSE
remains primary. Both modes report log-MSE, squared and unsquared overlap,
energy, variance, and separately labeled log-MSE/selected-objective gradient
norms. Legacy `gradient_norm` continues to mean the log-MSE gradient norm.
For `n_sites>=9`, `larger_system_fidelity_converged` assesses absolute unsquared
overlap `>=0.99`, independently of the retained strict `fit_gate_passed`; it is
null outside that working assessment scope. These are numerical acceptance
criteria. Existing raw records are not rewritten under the newer assessment.

`benchmarks/run_ed_fits.py` launches architectures/seeds sequentially. Defaults
are 2x3, all five models, seed 123, width 32, 10,000 epochs, Adam at `1e-3`, and
float64; width is unused for JastrowNet. Geometry, counts, width, seeds, device,
learning rate, dtype and reporting frequency are configurable.

```bash
# Minimal one-epoch launcher smoke, not a fit-accuracy result.
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python pythonTest/2D/benchmarks/run_ed_fits.py --Lx 2 --Ly 1 \
  --Ntot 2 --Nmax 2 --arches FCNet --epochs 1 --width 4 --seeds 123 \
  --eval-every 1 --device cpu

# Full supervised campaign command; launch only when assigned by the manager.
.venv/bin/python pythonTest/2D/benchmarks/run_ed_fits.py --Lx 2 --Ly 3 \
  --epochs 10000 --width 32 --seeds 123 --device cuda:0 --eval-every 100 \
  --run-name ed-fit-2x3-width32-seed123
```

Existing run directories are refused. Checkpoints go to `outputs/<run-name>/`.
Logs, per-run records, incremental summaries, and **byte-for-byte copies of the
full trainer JSON reports** stay in `benchmarks/results/<run-name>/`, preserving
history and trainer source hashes even if ignored checkpoints are removed.
Execution status and completed/failed/expected run counts describe whether the
process and report retention succeeded. A false fit gate remains a completed
scientific observation and does not stop the remaining architecture/seed runs.

The saved checkpoint minimizes the **selected** full-sector objective over
initialization and every post-update state; ties retain the earliest epoch.
`best_training_loss` and `loss_name` identify that selection. For default log-MSE
runs, `best_normalized_log_mse` retains its original minimum-loss meaning; for
infidelity runs it is null, and `normalized_log_mse_at_selected_checkpoint`
reports log-MSE at the infidelity-selected weights without claiming it is a
log-MSE optimum. `best_checkpoint_metrics` are
recomputed from those saved weights, while `final_metrics` describe the last
update and may differ. Training uses a fixed epoch budget without automatic
stopping on the gate or descriptive plateau flags. Snapshots report the actual
loss-gradient norm and loss trends. Checkpoints remain inference-only. Subsequent
VMC scaling uses SR with native SGD; the SR/Adam combination is rejected.

## Training and inference

Run commands from the repository root with the existing environment. Thread
limits also keep small CPU checks inexpensive; disable bytecode writes so caches
do not appear alongside source files.

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/bhnqs-2d-mpl

# Short smoke; these settings do not demonstrate convergence.
.venv/bin/python pythonTest/2D/train.py --Lx 2 --Ly 3 --arch FCNet \
  --epochs 2 --walkers 4 --steps 2 --burnin 2 --device cpu \
  --evaluation-batch-size 4 --output outputs/example.pt

# Tiny optional centered SR and guarded ED reference.
.venv/bin/python pythonTest/2D/train.py --Lx 2 --Ly 1 --Nmax 2 \
  --arch TransformerNet --hidden-dim 2 --update-method sr --epochs 1 \
  --walkers 2 --steps 2 --burnin 1 --device cpu --ed-reference \
  --output outputs/sr-smoke.pt
```

Training defaults: 4x4, width 32, 100 epochs, 64 walkers, 32 sample steps and
32 burn-in steps per measurement, native update, Adam, learning rate 0.001.
Use `--no-pbc`, `--seed`, `--dtype float64`, and `--device cpu`, `cuda[:index]`,
or `auto` as needed. `--ed-reference` is opt-in; the default never runs 4x4 ED.
Use `--help` for all options.

`--disable-tf32` is opt-in: for CUDA training it sets both matmul and cuDNN
TF32 flags false before the first forward. By default the backend flags are
left unchanged. The report records the actual flags, resolved device and dtype.
The bounded accelerator validation used disabled TF32; final physical metrics
still require independent CPU float64 evaluation.

For SR+SGD, `--warm-start <checkpoint>` loads inference weights after checking
architecture settings, geometry, each boundary axis, sector, cutoff and all
Hamiltonian couplings. It creates a fresh SGD optimizer and fresh walkers; it
does not resume optimizer, walker or RNG state. The destination must be new.
`--initial-burnin N` performs exactly one equilibration of those fresh walkers
before any measurement or update, in addition to each measurement's `--burnin`.
N counts proposals per walker, independently of thinning; no samples are retained.
Its default zero skips the sampler call and preserves the previous seeded path.
The JSON records completed initial proposal counts and units. An appropriate
burn-in still requires checking the frozen model's sampling behavior.

Optional `--lr-end` and `--tau-end` give smooth geometric schedules from the
starting values at epoch 1 to the endpoints at the last epoch; a single epoch
uses the starting values. Omitted endpoints keep the previous constant values.
Each history row records the values used for its update. Source hashes and the
command are captured at startup. These controls do not establish convergence.

`generate_initial_states` creates feasible integer walkers. `sample` returns
samples, final walkers and diagnostics. Its graph-free Metropolis cache is
recomputed at each call, including after parameter updates. Uniform bond and
direction proposals retain invalid hops as self proposals. Accepted counts include
self proposals; moved counts only accepted occupation changes. `local_energy`
uses the `U/2` convention and bounds connected-state batches.

`--thin K` in `train.py`, `evaluate.py` and the VMC baseline launcher controls
proposals between retained configurations. It defaults to **1**, preserving the
previous seeded behavior. The API spelling is keyword-only
`proposal_steps_per_sample=K` in `vmc.sample` and `sample_frozen`. Burn-in remains
a count of proposals per walker; then `steps*K` post-burn proposals are made,
retaining after every Kth proposal. `--steps` always counts **retained** time
points per walker, so SR's nominal M and stored sample count do not multiply by
K. Diagnostics count attempts/accepts/moves over **all** post-burn proposals,
including unretained intervals. JSON records these units explicitly.
Thinning may reduce adjacent duplicates at additional computation cost; it does
not establish mixing, independent gradients, ESS, or convergence.

`native_vmc_update` accumulates backward per evaluation chunk for
`2*mean((E-Emean).detach()*logpsi)`. Optional dense SR uses
`Jc=J-mean(J)`, `S=Jc.T@Jc/M`, `f=Jc.T@(E-Emean)/M`, and solves
`(S+tau*I)delta=f`. This imaginary-time direction has **no extra factor of two**;
SR requires native SGD. `--sr-solver auto` selects the smaller of parameter
count P and sample count M; explicit `parameter` and `sample` overrides are
available. The equivalent centered sample-space direction is
`Jc.T @ solve(Jc@Jc.T + M*tau*I, E-Emean)`, with the same SGD update and no extra
factor of two. Startup logs and JSON report solver, solve dimension, M, P,
Jacobian bytes and estimated allocation budget; update diagnostics include the
direction norm.

The default explicit budget is 512 MiB for Jacobian/conversion/centering arrays,
solve copies/workspace and vectors. A separate generous cap of 1,000,000
parameters is a fail-safe, replacing the obsolete 2,000-parameter restriction.
Both guards precede allocation. Sample-space SR still stores M*P derivatives:
the smaller solve does not remove that memory cost. Per-sample autograd supports
attention but remains a throughput limitation. The conservative estimate is not
a hard model-activation or process-memory bound. Raise limits only explicitly
for a measured workload.

`--jacobian-batch-size` defaults to **1**, the reference per-sample autograd path.
Opt-in chunks 2–64 use batched VJPs with an explicit allocation guard; Transformer
uses the math attention kernel for this path. The centered SR/SGD equations are
unchanged. Bounded CPU float64 Conv32 4x4 timings found chunk 8 about 2.0–2.1
times faster for Jacobian construction at M=512/1024, with maximum differences
below `3.4e-16`; these single timings exclude the solve and training and are not
GPU or convergence claims. See
[timing evidence](benchmarks/results/conv32-batched-vjp-phase-readiness.json) and
[readiness and initial-burn-in checks](benchmarks/results/conv-phase-readiness.json).

`--jacobian-method vmap` opts into functional per-configuration gradients using
`vmap(grad(functional_call))`, with chunk cap128; `vjp` remains the default and
keeps cap64. Both return the same float64 Jacobian for the same model arithmetic
and retain the centered SR/SGD equations. Occupation/sector and finite-forward
checks are retained outside the transform; Transformer uses math SDPA. The vmap
guard adds returned gradient blocks and one-hot inputs to the existing SR
workspace estimate before allocation, conservatively using float64 sizes. It
does not bound activations or total process memory. Method, chunk and budget are
recorded at startup and per update. See the
[standalone prototype checks and timings](benchmarks/results/vmap-jacobian-prototype.md)
and [integration evidence](benchmarks/results/vmap-integration-readiness.json).

`benchmarks/widen_conv.py` prepares an optional Conv32-to64 inference checkpoint
by duplicating convolution channels and dense hidden units, dividing incoming
dense and outgoing scalar weights by their replication counts. Optional seeded
zero-sum jitter can break gradient symmetry while preserving the initial
function. It checks feasible and frozen-sampled configurations in CPU float64
against a `1e-11` tolerance, preserves Hamiltonian/geometry metadata and the
source file, and refuses existing destinations. `--self-test` checks forward
agreement, finite gradients and replica splitting after one toy SGD update.
Prepared checkpoints have no continuation epochs; increased width also increases
the guarded SR allocation. See the
[prepared-option evidence and memory estimates](benchmarks/results/conv64-prepared-readiness.json).

`--evaluation-batch-size` bounds model evaluations and native backward graphs,
but not retained sample storage. Increasing walkers and steps still increases
sample memory. Checkpoints contain inference weights, dtype, complete configurable
architecture settings, geometry, sector, Hamiltonian and bond conventions,
parameter counts and implemented symmetries. They contain **no optimizer, walker,
or RNG state** and do not support exact training resumption. The JSON sidecar
`<checkpoint>.json` records settings/history and a fresh final measurement after
the final update. History measurements are taken before their corresponding update.

The CLI stages the checkpoint, reloads and measures those staged weights, then
serializes the sidecar before replacing either destination. Measurement or
serialization failure preserves the previous pair. The two file replacements
are individually atomic, **not a two-file transaction**: interruption or I/O
failure between replacements can leave a checkpoint and sidecar from different
runs. Check their metadata and provenance before combining them.

For inference from a Python process, add `pythonTest/2D` to `sys.path`, then:

```python
import torch
from models import load_model
from vmc import occupancy_tensor_to_onehot

model, metadata = load_model("outputs/example.pt", device="cpu")
occupations = torch.ones((1, metadata["Lx"] * metadata["Ly"]), dtype=torch.long)
dtype = next(model.parameters()).dtype
with torch.no_grad():
    log_amplitudes = model(occupancy_tensor_to_onehot(occupations, metadata["Nmax"], dtype))
```

## Frozen density and one-body measurements

`evaluate.py` loads an inference checkpoint without training it. By default it
measures a central physical-site correlation row and its reverse, all diagonals,
and both directions of Hamiltonian bonds. It does **not** automatically evaluate
all site pairs for every large Transformer sample. `--reference-sites i ...`
selects different rows; repeated `--pair i j` alone requests exactly those pairs.
Every selected pair is saved explicitly using `i=y*Lx+x`. `--full-rho` requests
the entire matrix for tiny checks. Missing directed bond entries leave energy
reconstruction unavailable rather than assuming Hermiticity to fill them.

```bash
# Frozen evaluation only; supply an existing checkpoint. Raw NPZ stays ignored.
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python pythonTest/2D/evaluate.py --checkpoint outputs/example.pt \
  --walkers 32 --steps 1024 --thin 1 --burnin 2048 --block-length 128 \
  --batch-size 256 --seed 321 --device cpu --dtype float64 \
  --output outputs/frozen-example/report.json
```

The checkpoint supplies boundaries, cutoff, sector and couplings. Evaluation
defaults to `--dtype checkpoint`, preserving its floating dtype. Explicit
`--dtype float32` or `--dtype float64` casts the loaded model in memory before
sampling, without rewriting weights or changing checkpoint metadata/checksum.
The report records `evaluation_dtype` separately from the saved metadata dtype.
Use `--device cpu --dtype float64` for the promised final double measurements of
float32-trained candidates. Density, factorial moment `n(n-1)`, local energy,
and raw one-body estimators retain `(time, walker, ...)` axes in
`<report>.samples.npz`. Model evaluations/connected states and retained observable
storage have separate batch/byte guards. Frozen samples cost memory even when
network evaluation is chunked.
The time axis counts retained configurations, not individual proposals. Block
length is in retained steps; its proposal span and retained proposal indices
are also saved so that thinned records can be interpreted without guessing.

For `rho[i,j]=<b_i† b_j>`, the row-action estimator at bra n is
`sqrt(n_i*(n_j+1))*psi(n-e_i+e_j)/psi(n)` for `i!=j`, with validity
`n_i>0, n_j<Nmax`; the diagonal is exactly `n_i`. This is distinct pointwise
from the transposed ket-action estimator. Raw Hermiticity is reported before any
symmetrization, along with diagonal-rho/density and trace-N checks. Energy is
reconstructed from raw directed rho entries and `U/2*<n(n-1)>-mu*<n>` and checked
against the existing local energy.

Nonoverlapping time blocks are formed **within each walker**, before pooling
block means. Complete-block means, all-sample means, discarded tails, block SEM
and between-chain SEM are explicit. Fewer than two complete blocks per chain
gives no block error estimate. Errors assume equilibrated independent walkers
and sufficiently decorrelated blocks; block-length stability and mixing must be
checked. They are not confidence intervals or ESS estimates. Low actual movement,
identical adjacent retained states, and noisy SR pilots do not establish
convergence or model capacity.

## Tests and bounded device verification

```bash
.venv/bin/python -m unittest discover -s pythonTest/2D/tests -v
.venv/bin/python pythonTest/2D/benchmarks/verify_gpu.py --cpu-only
.venv/bin/python pythonTest/2D/benchmarks/verify_gpu.py --cli-smoke
```

The tests cover lattice edges, exact basis counts and allocation guards, atomic
energies, independent hopping operators, 1D parity, model geometry and gradients,
proposal balance, the noninteracting factorial ground state, exact `Hpsi/psi`,
native/SR math, checkpoint round trips, and CLI smokes.

Device verification uses a fixed 2x2, two-particle, cutoff-two problem and width
two for all five models. It compares float64 local energies to ED, chunked native
steps, and tiny guarded SR. When CUDA is available it repeats with identical
weights, inputs, and supplied energies, comparing forward outputs and updates.
Default tolerances are `rtol=1e-8`, `atol=1e-9`, recorded with errors in JSON.
Unavailable CUDA is recorded as skipped, not passed. Independent stochastic
trajectories are not expected to match across CPU and GPU. `--cli-smoke` adds
one-epoch native/SR Transformer CLI checks. This script checks correctness, not
accuracy convergence or steady-state GPU performance.

## Reproducible baseline runs

The baseline runner defaults to **2x3** (unlike the square training default),
width 16, 100 epochs, 64 walkers, 16 sample steps, 32 burn-in steps, seeds
123/456/789, and all five architectures. Set both dimensions explicitly for
another shape. Baselines are native Adam with learning rate 0.001.

```bash
# Full default small baseline; run only when ready for sustained training.
.venv/bin/python pythonTest/2D/benchmarks/run_baseline.py --device cpu

# Configurable shape and seeds; exact evaluation is attempted under the guards.
.venv/bin/python pythonTest/2D/benchmarks/run_baseline.py --Lx 3 --Ly 3 \
  --seeds 123 456 --epochs 100 --width 16 --device cuda:0

# Larger case without any ED or sector enumeration.
.venv/bin/python pythonTest/2D/benchmarks/run_baseline.py --Lx 4 --Ly 4 \
  --skip-ed --device cuda:0

# Minimal runner smoke, not an accuracy baseline.
.venv/bin/python pythonTest/2D/benchmarks/run_baseline.py --Lx 2 --Ly 1 \
  --Nmax 2 --arches FCNet --seeds 123 --epochs 1 --width 2 \
  --walkers 2 --steps 1 --burnin 0 --device cpu
```

Each architecture/seed gets a separate training subprocess and checkpoint.
The default run name is a UTC timestamp; `--run-name` supplies a reproducible
label. Existing run directories are rejected to preserve earlier evidence.
The runner writes:

- `outputs/<run-name>/`: ignored binary inference checkpoints and CLI sidecars.
- `benchmarks/results/<run-name>/`: retained JSON summary, per-run JSON records,
  and subprocess logs. These include settings, seed, environment, parameter count,
  checkpoint paths and sampled final measurements.

New summaries record overall `status` (`running`, `completed`, or `failed`) and
`expected_run_count`. They are updated before each subprocess and after each
completed evaluation. A run is completed only after artifact loading and any
requested exact evaluation succeed; caught training, loading, or evaluation
exceptions record a failed status, stage, exception type, and message. Known ED
resource refusals remain explicit skips. A process killed before it can report
failure may leave `running`; that field alone does not establish that it is alive.

At run start, new summaries record SHA-256 hashes of the five core source modules
and benchmark Python scripts, with a UTC recording time. These are start-time
fingerprints, not archived source snapshots or proof that files stayed unchanged
through later subprocess launches. Older runs have no retroactive source hashes,
status fields, or source snapshots inferred by this update. Existing raw records
remain unchanged while older runners are active.

For feasible ED sectors, the runner also reports full-sector normalized variational
energy, energy/site, energy difference from ED, absolute amplitude overlap and
its square, and exact energy variance `||(H-E)psi||^2`. The energy difference is
a variational error, **not an excitation gap**. Overlap uses one ED ground vector;
degenerate ground spaces need a subspace comparison. Inference uses saved rounded
weights evaluated on CPU in float64. Sector-wide metrics are exact within that
finite lattice/cutoff/model, rather than Monte Carlo estimates.
New exact records name the variational energy difference `energy_error_to_ed`.
Legacy raw records may use `gap_to_ed_energy` for the same quantity; this legacy
alias also denotes a variational error, not an excitation gap.

ED counts the bounded sector before enumerating it, assembles sparse matrices
directly, and defaults to dimension 20,000 and a separate 512 MiB estimated-memory
guard. A counting-work guard also prevents oversized DP workspace. Resource
rejection is recorded explicitly and training continues without exact metrics.
`--skip-ed` skips enumeration and exact evaluation entirely. Never assume that
all cutoffs on 3x3 or larger geometries are ED-feasible.

Subprocess wall time includes Python startup/imports and artifact writing. CLI
internal training/final-measurement timing is stored separately. Neither measures
steady-state throughput; startup-separated throughput is not measured here.
Per-run records copy `cuda_peak_allocated_bytes` and `cuda_peak_allocated_scope`
from the CLI report. The peak is for the selected training device's PyTorch
allocator, reset after walker initialization and observed through final
measurement. It includes the resident allocation baseline, training updates,
and staged checkpoint reload; it excludes setup/ED startup. It measures allocated
bytes, not reserved allocator bytes or total device/process memory, and excludes
the runner's later CPU exact evaluation. CPU runs and older sidecars without
these fields report null rather than an inferred peak. Compare memory only with
matching scopes. Sampled local-energy variance is not an error bar
on mean energy. Autocorrelation and effective sample size are not estimated.
Multiple seeds are separate observations, without confidence claims or rankings.
Cutoff convergence, longer optimization, and larger-size convergence remain
necessary before physical accuracy claims.

## Initial measurements: October 30, 2026

The first validation campaign completed 30 short training runs: five models,
three seeds (123/456/789), and two geometries (2x3 and 3x3). Each used 100 native
Adam updates, width 16 where applicable, 64 walkers, 16 retained proposal steps,
32 burn-in proposals per walker per update, and evaluation batches of 1,024.
One step is one bond/direction proposal per walker, not a lattice sweep.
Settings were `t=1`, `U=16.74`, `mu=0`, `Nmax=4`, unit filling, periodic unique
bonds. Separate GPU queues ran the two geometries. Full records and logs:

- [2x3, 100 epochs, three seeds](benchmarks/results/2026-10-30-2x3-native100/summary.json)
- [3x3, 100 epochs, three seeds](benchmarks/results/2026-10-30-3x3-native100/summary.json)

A subsequent 3x3 run used the same settings with 500 updates and seed 123 on
GPU 0. Its cutoff-four ED reference is **-6.104592578442**. The following are
full-sector metrics of the final saved weights, evaluated in float64 on CPU:

| Architecture | Exact model energy | Energy error per site | Absolute overlap | Total energy variance |
| --- | ---: | ---: | ---: | ---: |
| FCNet | -4.862295 | 0.138033 | 0.973842 | 40.907804 |
| JastrowNet | -5.747738 | 0.039651 | 0.992614 | 13.587926 |
| Conv2DNet | -5.521676 | 0.064768 | 0.985708 | 14.265040 |
| RBMNet | -5.152907 | 0.105743 | 0.975313 | 23.275439 |
| TransformerNet | -5.199049 | 0.100616 | 0.979278 | 22.842581 |

These are **unconverged single-seed observations**, not an architecture ranking.
Every model improved its seed-123 exact energy relative to the 100-update run,
but residual errors and variances remain substantial. High overlap alone does
not establish an accurate eigenstate. The reference-cutoff error was checked
separately: increasing 3x3 from `Nmax=4` to the full nine-particle cutoff changes
ED energy by about `3.25e-6`; see [the manager's checks](PLAN.md).

[500-update records](benchmarks/results/2026-10-30-3x3-native500/summary.json)
include commands, source hashes, scoped memory peaks, and exact measurements.
End-to-end subprocess times were approximately 29–49 seconds; selected-device
allocator peaks were approximately 2–66 MiB under the documented CLI scope.
These workstation jobs had no exclusive hardware reservation or timing repeats.
They do not establish steady-state throughput.

All five models also completed three-update 4x4 CUDA execution checks with ED
disabled. [Those smoke records](benchmarks/results/2026-10-30-4x4-smoke/summary.json)
establish execution only; their energies are far from a converged ground state.
[GPU verification](benchmarks/results/verify_gpu0.json) records five-architecture
CPU/CUDA/ED checks and native/SR CLI smokes. The CPU regression suite passed
43 tests after the corrective reviews.

### Paired exact-gradient diagnostic

`benchmarks/check_training.py` compares sampled native training against direct
differentiation of the normalized full-sector Rayleigh energy. It is restricted
to a tiny 2x3 FCNet problem, CPU float64, at most 500 updates and dimension 1,000.
The models begin with identical weights and separate Adam states. It is a
diagnostic, not a scalable exact-training mode or an architecture benchmark.

```bash
.venv/bin/python pythonTest/2D/benchmarks/check_training.py \
  --updates 500 --tag paired-training-new --save-checkpoints
```

The [recorded seed-123 pair](benchmarks/results/paired-training-500-20261030/summary.json)
used width 16 and learning rate 0.001. At update 500, exact model energies were
**-2.062783** with sampled updates and **-2.074754** with exact updates, compared
with ED **-2.581090**. Both improved strongly, and both retained a sizable error.
Initial direct Rayleigh and probability-weighted covariance gradients agreed
within `2.7e-15`. In this one diagnostic, sampling alone does not explain the
remaining error. This does not establish a representational limit or exclude
sampling problems on other geometries. Its float64 initialization differs from
the float32 baseline; compare the paired branches to each other.

These earlier variational diagnostics do not replace the supervised gate now
specified above. Even-size convergence, optimizer/model capacity, and sampling
diagnostics still need to be settled before critical-region conclusions.

## Supervised fit results: retained-report snapshot

These observations use Adam, the normalized full-sector log-amplitude MSE,
`t=1`, `U=16.74`, `Nmax=4`, unit filling, and periodic unique bonds. Metrics below
come from the **best-by-loss saved checkpoint** in completed retained reports,
not necessarily the final update. Overlaps are absolute and unsquared. Both
overlap `>=0.9999` and energy error/site `<=0.001` must pass. All completed
reports included here have zero clamped targets. These are finite-budget fits,
not architecture rankings or capacity claims; earlier VMC tables remain
exploratory energy-minimization results.

### Fixed 10k-epoch 2x3 comparison

All five seed-123 float64 runs completed, with width 32 where applicable.
JastrowNet has no width parameter.

| Model | Normalized log MSE | Overlap | Energy error/site | Joint gate |
| --- | ---: | ---: | ---: | --- |
| FCNet | 8.14794e-6 | 0.999998424 | 1.20447e-5 | Pass |
| JastrowNet | 0.023511369 | 0.997297954 | 0.016957214 | Fail |
| Conv2DNet | 1.89900e-5 | 0.999997787 | 1.68752e-5 | Pass |
| RBMNet | 0.007757938 | 0.998617758 | 0.008604880 | Fail |
| TransformerNet | 0.000154214 | 0.999995272 | 4.29985e-5 | Pass |

[Fixed-budget reports](benchmarks/results/2026-10-30-ed-2x3-w32-e10000/summary.json).
Jastrow's selected loss-gradient norm is `9.23e-10`, with a persistent late
plateau; this describes stationarity for the present loss and parameterization.
No infidelity-objective experiment or Jastrow change is included here.

### Separate 20k-epoch RBM32 follow-up on 2x3

Longer float64 RBM32 runs pass both gates for both tested seeds. Keep these
separate from the fixed 10k table; training budget changed.

| Seed | Normalized log MSE | Overlap | Energy error/site | Joint gate |
| --- | ---: | ---: | ---: | --- |
| 123 | 0.000409434 | 0.999947282 | 0.000421183 | Pass |
| 456 | 0.000171709 | 0.999971984 | 0.000203653 | Pass |

[RBM32 follow-up reports](benchmarks/results/2026-10-30-ed-2x3-RBM32-e20000/summary.json).
Two seeds are separate observations, without confidence or generalization claims.

### Completed initial 3x3 trials

The original width-32, seed-123, float64 10k campaign retains four completed
models: FCNet, JastrowNet, Conv2DNet and RBMNet all fail the joint gate. The last
TransformerNet run was intentionally stopped with SIGINT for runtime reasons;
this is not an ansatz-failure observation. The separate float32 Transformer
replacement has completed; independent CPU float64 evaluation confirms that it
misses the joint gate (overlap `0.997979978`, energy error/site `0.009525741`).
Do not treat the original campaign as five completed fits.
[Current campaign records](benchmarks/results/2026-10-30-ed-3x3-w32-e10000/summary.json).
[Transformer replacement](benchmarks/results/2026-10-30-ed-3x3-Transformer32-f32-e10000/summary.json).

A separate float64 FC128, seed-123, 20k run improves on its 10k counterpart but
still fails: overlap `0.999382937`, energy error/site `0.002829422`, normalized
log MSE `0.000823582`, and nonzero loss-gradient norm `0.000474921`.
[FC128 20k report](benchmarks/results/2026-10-30-ed-3x3-FC128-e20000/FCNet-seed123-trainer.json).
The wider Conv128 float32 run has also completed. Independent CPU float64
evaluation of its selected weights gives overlap `0.999154918`, energy error/site
`0.003501212`, and normalized log MSE `0.001243754`, so it also misses the gate.
[Independent float64 rechecks](benchmarks/results/ed-float64-rechecks.json) retain
checkpoint hashes and precision comparisons for both float32-trained models.

The 3x3 failures above refer to the **historical strict joint gate**. For the
primary working assessment of absolute unsquared overlap `>=0.99`, prefer the
canonical width-32/10k candidates wherever available: the four retained original
float64 models and the independently float64-evaluated width-32 Transformer
replacement. All five satisfy that fidelity assessment, with energy and variance
reported separately. Wider FC/Conv and longer-budget trials remain separate
follow-ups; they are not needed to make this width-32 fidelity statement.
This working threshold is not a physics criterion or an architecture ranking.

Adam with log-MSE remains the primary supervised method. The completed optional
Adam/infidelity comparison is described below. Loss trends and nonzero gradients
do not establish an architecture impossibility claim.
The qualifications above refer to the earlier fully periodic reference. October 31 cylinder
VMC pilots follow those architecture checks; their execution and frozen
measurements are recorded under
`benchmarks/results/2026-10-31-cylinder-pilots/`. A separate matched 3x3-cylinder
supervised check is recorded in
`benchmarks/results/2026-10-31-3x3-cylinder-supervised/`. Qualification on one
boundary geometry is not silently reported as qualification on another. In
particular, the cylinder Jastrow log-MSE plateau has overlap about `0.9882`,
below the working `0.99` criterion; its loss/representation diagnostic is kept
separate. No 4x4 VMC ground-state fidelity is inferred from energies.

### Experimental Jastrow loss comparison

Both objectives use Adam at `lr=1e-3`, 10,000 full-sector updates, seed 123,
float64 and the same zero-initialized 80-parameter Jastrow ansatz. All physical
metrics use raw ED states and selected saved checkpoints. The 2x3 pair was run
fresh on CPU. The 3x3 comparison reuses the earlier stationary log-MSE run and
a new GPU-1 infidelity run with matched scientific settings. Trainer versions
differ, while frozen model/ED hashes and unchanged default log-MSE mathematics
were independently verified.

| Geometry | Adam objective | Absolute overlap | Energy error/site | Normalized log-MSE | Energy variance |
| --- | --- | ---: | ---: | ---: | ---: |
| 2x3 | Log-MSE (primary) | 0.997297954 | 0.016957214 | 0.023511369 | 2.043093 |
| 2x3 | Infidelity (experimental) | 0.999746299 | 0.003022773 | 0.386931591 | 0.672997 |
| 3x3 | Log-MSE (primary) | 0.992349382 | 0.028694622 | 0.023387399 | 5.043727 |
| 3x3 | Infidelity (experimental) | 0.998520270 | 0.010327029 | 0.309533940 | 3.126790 |

Infidelity fitting improves overlap, energy error and variance in these cases,
while increasing log-MSE. Uniform log-MSE penalizes relative amplitude errors
across all configurations, including very small amplitudes; overlap emphasizes
their contribution to the normalized wavefunction. This comparison does not
replace the chosen primary method or establish a universal optimizer ranking.
The infidelity-selected gradient norms remain nonzero, approximately `7.1e-6`
and `9.6e-5`, so these are finite-budget observations.

Both 3x3 fits satisfy the user's absolute-overlap threshold of 0.99. All four
miss the historical strict joint gate, whose definition remains in the reports.
[Independent checkpoint comparison](benchmarks/results/jastrow-loss-comparison.json)
includes hashes, source versions, both overlap definitions, gradients, and raw
report references. [Canonical 3x3 assessment](benchmarks/results/jastrow-paired-loss-and-canonical099.json)
confirms that all five width-32 candidates, where applicable, already meet the
0.99 absolute-overlap criterion using primary log-MSE training.
