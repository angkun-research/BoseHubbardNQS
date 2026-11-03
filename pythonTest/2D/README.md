# Two-dimensional Bose–Hubbard NQS

A self-contained Python implementation for rectangular square lattices, with
ED, supervised fitting, PyTorch VMC, frozen observables and sector diagnostics.
The 1D code and Julia integration are separate and unchanged. **No merge has
been performed.**

## Documentation

| Document | Purpose |
| --- | --- |
| [API reference](docs/API.md) | Function signatures, tensors, returns, validation and examples |
| [Benchmark workflows](BENCHMARKING.md) | ED → supervised → VMC → frozen measurement; reproduction |
| [Julia preparation](docs/JULIA_PREPARATION.md) | Existing PythonCall boundary, data contract and future merge sequence |
| [Current results](STATUS.md) | Accepted evidence and known limitations |
| [Next steps](PLAN.md) | Preparation and review gates before any merge |
| [Cleanup record](docs/CLEANUP.md) | Removed disposable files, archived history and protected evidence |
| [Storage and Git policy](docs/STORAGE.md) | Pushable source versus local-only checkpoints and benchmark data |
| [Slide sources](slides/README.md) | Reproduce the accepted 55-page presentation |

## Physical and array conventions

```text
H = -t Σ_<i,j>(b_i† b_j + b_j† b_i)
    + (U/2) Σ_i n_i(n_i-1) - μ Σ_i n_i
i = y*Lx+x;  0 <= n_i <= Nmax;  Σ_i n_i = Ntot
```

Defaults: `t=1`, `U=16.74`, `mu=0`, inclusive cutoff `Nmax=4`, unit filling,
`Ly=Lx`, periodic x and y. Independent `pbc_x/pbc_y` override the common `pbc`
flag. Each unordered hopping edge occurs once; periodic length-two axes have
one edge, length-one axes have no self edge. The existing 1D ED uses a different
interaction coefficient and short-ring bond convention; see the Julia notes.

Models take one-hot `(batch, Lx*Ly, Nmax+1)` tensors and return `(batch,)` real
**unnormalized log amplitudes**. VMC accepts integer occupation tensors and
samples a fixed-number sector. Raw model outputs outside that sector are not
number projected. Absolute overlap means unsquared amplitude overlap throughout.

## One VMC implementation

`vmc.py` is the single batched PyTorch implementation for CPU and CUDA. It
contains sampling, local energies, the native energy gradient and SR. Jacobian
`vjp` is the compatibility/reference default (chunk1); optimized `vmap` is opt-in
(chunk up to128). These are interchangeable derivative backends, not separate
CPU/GPU VMC codes. Recent benchmarks explicitly used `vmap` with chunk128.

Ridge SR is the default, with mathematically equivalent parameter/sample-space
dense solves; `auto` chooses the smaller dimension. Full Jacobians are retained.
Optional MinSR uses a spectral soft filter. SR requires native `torch.optim.SGD`;
supervised ED fitting uses Adam + normalized log-MSE. No optimizer or numerical
default is changed by documentation or cleanup.

## Layout and execution

Core: `lattice.py`, `exact_diagonalization.py`, `models.py`, `vmc.py`,
`observables.py`, `leakage_analysis.py`. Entry points: `train_ed.py`, `train.py`,
`evaluate.py`; `dmrg_reference.jl` provides the optional ITensor reference.
`tests/` checks correctness; `benchmarks/` contains reusable analysis and retained
experiment scripts; `benchmarks/results/` and ignored `outputs/` preserve evidence.
The six registered model families include the optional Conv2DTransformerNet;
the default benchmark sweep remains the original five.

Use the repository's existing Python environment from the repository root:

```bash
.venv/bin/python -B pythonTest/2D/train_ed.py --help
.venv/bin/python -B pythonTest/2D/train.py --help
.venv/bin/python -B pythonTest/2D/evaluate.py --help
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m unittest discover -s pythonTest/2D/tests -v
```

Core dependencies are PyTorch, NumPy and SciPy; presentation rendering also uses
Matplotlib. The optional DMRG workflow uses the existing Julia environment.
Relative model/report paths resolve from this 2D directory, not the shell cwd.
Writes are guarded to stay here. Use fresh output paths and preserve JSON/NPZ
companions. Checkpoints contain inference weights and metadata, not complete
optimizer/walker/RNG training state.

Historical documentation is retained verbatim under
[docs/history/pre-documentation-cleanup](docs/history/pre-documentation-cleanup/).
It records earlier milestones and superseded plans; it is not the current API.
