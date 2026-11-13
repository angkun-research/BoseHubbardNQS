# BoseHubbardNQS

[![Documentation](https://img.shields.io/badge/documentation-online-blue)](https://angkun-research.github.io/BoseHubbardNQS/)

Neural quantum states for one- and two-dimensional Bose–Hubbard models, with
PyTorch training, CPU/CUDA variational Monte Carlo, and a Julia interface.

**[Documentation](https://angkun-research.github.io/BoseHubbardNQS/)**
— guides, API reference, examples, and benchmark results.

The Python numerical engines provide ED references, supervised fitting,
fixed-number sampling, energy gradients, stochastic reconfiguration and
observables. Julia evaluates the same models one configuration or one batch of
walkers at a time, and launches the existing 2D workflows.

- [Getting started](docs/site/getting-started.md)
- [1D model and examples](docs/site/model-1d.md)
- [2D model and examples](docs/site/model-2d.md)
- [Python workflows](docs/site/python.md) and [API](docs/site/api/python.md)
- [Julia interface](docs/site/julia.md) and [API](docs/site/api/julia.md)
- [Batching and CUDA](docs/site/batching.md)
- [Physics and conventions](docs/site/physics.md)
- [Benchmarks and limitations](docs/site/benchmarks.md)
- [Documentation website build and GitHub Pages](docs/site/development.md)

## Start small

Run from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python pythonTest/TrainVMCstates.py \
  --arch FCNet --L 4 --Ntot 4 --Nmax 4 --epochs 20 \
  --device cpu --ed-reference --output data/example-1d.pt
./julia-local -e 'using Pkg; Pkg.instantiate()'
```

The short run demonstrates the workflow; it does not reproduce a converged
benchmark. CUDA requires a compatible PyTorch build and driver. The
`julia-local` launcher selects this checkout's Python environment for PythonCall.

```julia
using BoseHubbardNQS

# Existing scalar calls keep their CPU default and Float64 return value.
model = load_nn("data/example-1d.pt")
logpsi = nn_value(model, [1, 1, 1, 1])

# Optional device selection and one configuration per column.
model_gpu = load_nn("data/example-1d.pt"; device="auto")
configs = hcat([1, 1, 1, 1], [0, 2, 1, 1])
logpsi_batch = nn_values(model_gpu, configs)
```

`device="auto"` selects CUDA when available to PyTorch, otherwise CPU. An
explicit unavailable CUDA request raises an error. Outputs are **unnormalized
real log amplitudes**, not probabilities. The batch matrix has shape
`(sites, walkers)` and results preserve column order.

## Two-dimensional models

Train a small checkpoint with the Python CLI:

```bash
.venv/bin/python -B pythonTest/2D/train_ed.py \
  --arch GroupEquivariantConv2DNet --Lx 2 --Ly 3 --width 8 \
  --epochs 20 --device cpu --dtype float64 \
  --output outputs/docs-example/ed-2x3.pt
```

Then load it through Julia:

```julia
model2d = load_nn_2d("pythonTest/2D/outputs/docs-example/ed-2x3.pt")
sigma = [1, 1, 1, 1, 1, 1]
logpsi2d = nn_value(model2d, sigma)
values2d = nn_values(model2d, hcat(sigma, [0, 2, 1, 1, 1, 1]))
```

`NQS2D` retains the saved geometry, dtype, architecture and physical metadata.
`train_ed_2d`, `train_vmc_2d` and `evaluate_2d` launch the established Python
programs in isolated subprocesses and return reports and paths. They do not
reimplement ED or VMC in Julia. See [the API](docs/site/api/julia.md) for
keyword and path conventions and [the runnable example](example/integration_2d.jl).

The 2D families are FCNet, JastrowNet, Conv2DNet, RBMNet, TransformerNet,
Conv2DTransformerNet and GroupEquivariantConv2DNet. The GCNN uses D4-tied local
filters: squares have D4 symmetry and rectangles have D2 lattice symmetry.
Both axes must be periodic for GCNN. The existing four 1D VMC families remain
FCNet, JastrowNet, Conv1DNet and RBMNet.

## Important conventions

- 1D uses `U*n*(n-1)`; 2D uses `U*n*(n-1)/2`. Match interactions with
  `U_1D = U_2D/2`. Device selection and loading do not convert this convention.
- The occupation cutoff is inclusive. Sampling preserves `Ntot`; raw inference
  has no number projection. Normalize over a specified sector, not per batch.
- 2D physical order is `i=y*Lx+x`; flatten Julia `(Lx,Ly)` images with `vec`.
- Short periodic axes have different historical 1D/2D bond conventions; use
  open chains or periodic lengths at least three for parity checks.
- Checkpoints contain inference weights and metadata, not complete optimizer,
  walker and RNG state for exact training resume.

## Validation and retained evidence

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m unittest discover -s pythonTest/2D/tests
.venv/bin/python -B -m unittest discover -s tests
./julia-local test/runtests.jl
```

Set `BHNQS_TEST_CUDA=1` for optional actual GPU bridge checks on a CUDA-capable
machine. GitHub CI runs CPU tests with generated small checkpoints; the website
build has its own lightweight Python environment and performs no training.

The [1D benchmark report](docs/VMC_BENCHMARKS.md) and
[completed GCNN study](pythonTest/2D/docs/GCNN_STUDY.md) distinguish exact
fidelity from sampled energies. The 3x4 GCNN fit misses the recorded working
criterion; 4x4/5x5 exact fidelity and ground-state convergence remain unverified.
Large checkpoints, raw chains and source snapshots remain local and ignored by
Git; compact results, source, tests and documentation are versioned.

The original ITensor/Julia reference examples remain under `example/`; their
hardcoded historical checkpoint dimensions may differ from current training
examples. Use the metadata-based loaders for new work.
