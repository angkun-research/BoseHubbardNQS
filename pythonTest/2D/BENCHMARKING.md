# Reproducible workflows

Run from the repository root using the existing environments. These are usage
examples, not instructions to rerun a research campaign. Detailed callable APIs
are in [docs/API.md](docs/API.md); accepted results are in [STATUS.md](STATUS.md).

## ED and supervised qualification

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -B pythonTest/2D/train_ed.py \
  --arch Conv2DNet --width 32 --Lx 2 --Ly 3 --Ntot 6 --Nmax 4 --pbc \
  --t 1 --U 16.74 --mu 0 --loss log_mse --epochs 10000 --lr .001 \
  --device cpu --dtype float64 --seed 123 --output outputs/example-ed/Conv2D.pt
```

ED is guarded by basis dimension and sparse-memory estimates. Do not bypass
these guards for larger VMC systems. Adam + normalized log-MSE is primary;
absolute unsquared overlap, energy and variance are independent checks. Select
supervised checkpoints by loss, not a favorable evaluation overlap.

## VMC using the shared PyTorch core

After supervised qualification, a fresh small VMC example is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -B pythonTest/2D/train.py \
  --arch Conv2DNet --hidden-dim 32 --Lx 2 --Ly 3 --Ntot 6 --Nmax 4 --pbc \
  --t 1 --U 16.74 --mu 0 --update-method sr --optimizer sgd \
  --sr-regularizer ridge --sr-solver auto --jacobian-method vmap \
  --jacobian-batch-size 128 --max-sr-bytes 4294967296 \
  --epochs 500 --walkers 64 --steps 16 --initial-burnin 4096 --burnin 64 \
  --thin 8 --lr .02 --lr-end .002 --tau .001 --tau-end .0001 \
  --device cpu --dtype float64 --seed 123 --output outputs/example-vmc/Conv2D.pt
```

`--steps` is retained time per walker; M=walkers*steps. `--thin` and burn-in
count local proposals per walker, not lattice sweeps. Initial burn-in applies
before the first update; per-update burn-in is distinct. Initial walkers are
feasible configurations, not equilibrium samples. Training history is measured
before each update; independently evaluate the final saved checkpoint.

The CLI compatibility defaults are VJP/chunk1, ridge and automatic dense solve.
The command above explicitly selects optimized vmap/chunk128. Both compute the
full Jacobian and share CPU/CUDA code. Ridge covariance/solve arithmetic is
float64 on the tensor device. Resource estimates exclude model activations and
vendor workspace; batch size does not bound total retained-sample storage.
SR requires native SGD. The example budget is not a convergence guarantee.

## Frozen density and one-body measurements

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -B pythonTest/2D/evaluate.py \
  --checkpoint outputs/example-vmc/Conv2D.pt --device cpu --dtype float64 \
  --walkers 64 --steps 512 --burnin 4096 --thin 8 --block-length 64 \
  --full-rho --seed 654 --output outputs/example-vmc/frozen654.json
```

Also retain an independent seed321 report. Neither evaluation selects the fit.
The JSON and `.json.samples.npz` companion preserve time/walker axes, ordered
physical pairs and raw estimators. Direction/bond composites are formed within
each sample before blocking. Conditional SEM=max(blockSEM,between-chainSEM),
not a mixing proof. Exact small-sector NQS/ED sums and MC estimates are separate
measurements. Full one-body matrices are intended for tiny systems; selected
pairs and observable-memory guards support larger cases.

## Geometry, references and archived protocols

Default geometry is periodic in both axes. A cylinder must explicitly pass
`--no-pbc-x --pbc-y`; NQS/ED/DMRG comparisons must use the same geometry,
particle number, cutoff, parameters and unique-edge convention. DMRG uses
`dmrg_reference.jl` and the existing Julia environment, with physical ordering
converted from the MPS snake by the validated reference code. See `--help` and
the recorded matched reference commands; do not reinterpret an open-chain MPS
representation as open physical boundaries.

`benchmarks/` contains reusable comparison tools and historical experiment
runners. Some historical runners use fixed dated paths, refuse existing output
or execute work when invoked: do not run them as generic importable APIs.
Their stored commands, source hashes, precision settings and selection rules
are the authority for reproducing the accepted measurements. Preserve failures,
all frozen seeds, checkpoints and raw chains. Use fresh output paths; training
can overwrite its checkpoint/sidecar, while frozen analysis generally rejects
existing destinations. Checkpoints are inference artifacts, not exact resumes.

For slides, follow [slides/README.md](slides/README.md). Historical extended
workflow notes are retained under `docs/history/pre-documentation-cleanup/`.
