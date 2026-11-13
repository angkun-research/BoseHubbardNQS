# Python workflows

The Python implementations provide the numerical engine used from both Python
and Julia. All examples run from the repository root.

## 1D VMC

```bash
.venv/bin/python pythonTest/TrainVMCstates.py \
  --arch Conv1DNet --L 8 --Ntot 8 --Nmax 4 --hidden_dim 32 \
  --epochs 100 --update_method native --optimizer adam --lr 0.001 \
  --n_walkers 64 --n_steps 16 --burnin_steps 16 \
  --device cpu --output data/example-vmc-1d.pt
```

Use `--update_method sr --optimizer sgd` for stochastic reconfiguration.
Dense SR memory grows with parameter count; use the trainer's resource
controls and start small. The 1D ED implementation allocates a dense matrix,
so use `--ed-reference` only when the sector is small.

## 2D supervised learning

```bash
.venv/bin/python -B pythonTest/2D/train_ed.py \
  --arch GroupEquivariantConv2DNet --Lx 3 --Ly 3 --width 8 \
  --loss log_mse --epochs 10000 --lr 0.001 \
  --device cpu --dtype float64 --output outputs/example-ed/best.pt
```

This fits a normalized, uniform full-sector log-MSE using Adam. The physical
overlap and energy are evaluated against the raw ED vector. Checkpoint selection
uses loss, rather than selecting a favorable evaluation fidelity.

Full ED is guarded before allocating the basis or sparse Hamiltonian. The
completed 3x4 GCNN study uses exact symmetry-orbit batching for training memory;
its multiplicities and target moments preserve the full-sector objective.
That benchmark helper is not a new ED solver or optimizer.

## 2D energy-based VMC

```bash
.venv/bin/python -B pythonTest/2D/train.py \
  --arch GroupEquivariantConv2DNet --Lx 3 --Ly 3 --hidden-dim 8 \
  --update-method sr --optimizer sgd --epochs 500 \
  --walkers 64 --steps 16 --initial-burnin 4096 --burnin 64 --thin 9 \
  --lr 0.002 --lr-end 0.0002 --tau 0.01 --tau-end 0.001 \
  --jacobian-method vmap --jacobian-batch-size 128 \
  --dtype float64 --device cpu --output outputs/example-vmc/final.pt
```

`--steps` counts retained steps per walker. Burn-in and thinning count local
proposals, not complete lattice sweeps. Initial walkers are valid configurations,
not equilibrium samples. A fixed update budget does not establish convergence.

The same `vmc.py` handles CPU and CUDA. SR uses the centered, sample-normalized
Jacobian and native SGD; `--update-method native --optimizer adam` selects the
existing energy-gradient route. ED is opt-in, so larger VMC cases do not trigger
an exponentially large reference calculation.

## Independent frozen measurements

```bash
.venv/bin/python -B pythonTest/2D/evaluate.py \
  --checkpoint outputs/example-vmc/final.pt --seed 321 \
  --walkers 64 --steps 256 --burnin 4096 --thin 9 --block-length 32 \
  --device cpu --output outputs/example-measurement/report.json
```

Frozen evaluation keeps time and walker axes, saves raw samples alongside its
JSON report, and supports selected correlations or a full one-body matrix.
Use independent seeds and check block-length stability. The checkpoint contains
inference weights, not the optimizer, walkers and RNG needed for exact resume.

See the [Python API](api/python.md) and [measured benchmarks](benchmarks.md).
