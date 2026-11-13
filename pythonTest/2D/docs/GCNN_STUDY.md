# Minimal group-equivariant CNN accuracy study

Completed at the user's request on 2026-11-13. This studies the established GCNN
with supervised log-MSE and energy-based VMC; the earlier representation/SVD
compression direction remains closed. The user deferred a new symmetry-reduced
ED solver and accepted 2x3 as a replacement for oversized exact-reference cases.

## Results

All declared runs completed. This is one training seed per case, not a
convergence or architecture-capacity study. The width-8 3x4 fit misses the
previous working absolute-overlap criterion of 0.99.

| System and objective | Absolute overlap | Squared fidelity | Exact energy error/site (t) |
| --- | ---: | ---: | ---: |
| 2x3 log-MSE | 0.9999999476 | 0.9999998952 | 4.5630712e-07 |
| 3x3 log-MSE | 0.9976990571 | 0.9954034086 | 0.011931558 |
| 3x4 log-MSE | 0.9842388185 | 0.9687260519 | 0.042972464 |
| 3x3 VMC | 0.9980917331 | 0.9961871077 | 0.0047942699 |

Both rectangular fidelities were independently recomputed by reloading the
saved checkpoints and evaluating every cached ED configuration on GPU 0.
The recomputed overlaps match the reports exactly. On 3x4 only four target
amplitudes were floored at 1e-12; fidelity and energy use the unfloored ED
vector. The ED ground-state energy is -7.642607747766t.

| VMC lattice | Frozen E/site, seed321 (t) | Frozen E/site, seed654 (t) |
| --- | ---: | ---: |
| 3x3 | -0.672888752 | -0.673414446 |
| 4x4 | -0.580353488 | -0.583509129 |
| 5x5 | -0.544315734 | -0.542221964 |

The 3x3 saved state has exact energy/site -0.673493794327t, compared with
ED ground-state energy/site -0.678288064271t. Frozen-chain energy estimates
are close to that saved-state value. The compact JSON retains variances,
block lengths 16/32/64/128, conditional errors, and the final training trend.
No exact fidelity reference is available for 4x4 or 5x5. Their remaining
variance and finite training budget prevent a claim of converged ground-state
accuracy. The 3x4 supervised loss was still falling at update 10000; the miss
does not prove an ansatz capacity limit.

## Ansatz and physical settings

The production addition is the GCNN ansatz and its symmetry/geometry support.
It follows the same model and checkpoint interface as the other NQS families.

| Component | Implementation used by this study |
| --- | --- |
| Ansatz and symmetry | `models.GroupEquivariantConv2DNet`, with D4 helpers in `square_group.py` |
| Hamiltonian, basis and ED | Existing `lattice.py` and `exact_diagonalization.py`, unchanged |
| Supervised optimization | Existing normalized log-MSE objective and Adam; exact orbit batching in benchmark runners |
| VMC sampler and local energies | Existing `vmc.py`, unchanged |
| SR and optimizer | Existing centered ridge SR and native `torch.optim.SGD`, unchanged |
| Checkpoint and frozen measurements | Existing `save_model`/`load_model`, `evaluate.py` and `observables.py` |

The benchmark launchers record a fixed experiment protocol. They do not define
a second VMC implementation. No new ED solver or GCNN-specific optimizer was
introduced. The rectangle-related CLI change accepts the ansatz geometry;
the shared supervised optimizer and loss implementation are unchanged.

One 3x3 lifting convolution, SiLU, one 3x3 D4 group convolution, SiLU, mean over
space and orientation, and a linear log-amplitude head. Width 8 gives 4993
trainable parameters. Constructor defaults remain width 16; previous weights
remain compatible. No new sampler or optimizer implementation is introduced.

All runs use t=1, U=16.74, mu=0, Nmax=4, unit filling and periodic boundaries.
Squares enforce translations and D4. Rectangles retain the same D4-tied local
filters at the user's request, but only translations and D2 are symmetries on
the fixed rectangle. Quarter rotations relate transposed shapes. The length-two
axis of 2x3 aliases opposite convolution offsets, so it is a small accuracy
check rather than a substitute for the larger geometry.

## Protocol

- Supervised: Adam, lr 0.001, 10,000 updates, normalized uniform full-sector log-MSE.
  Rectangles use exact translation/D2 orbit multiplicities and target moments,
  including within-orbit target variance. Select by lowest post-update log-MSE,
  including initialization. Float32 training with TF32 disabled; direct
  float64 full-sector evaluation of the selected saved weights. Report absolute
  overlap and its square separately. Physical metrics always use the raw ED
  vector, not the floored log target.
- VMC: existing centered ridge SR plus native SGD, 500 updates, 64 walkers x16
  retained steps, thin=number of sites, 4096 initial burn-in proposals per walker,
  64 burn-in proposals per update. Float64 forward/Jacobian/solve, TF32 disabled,
  vmap Jacobians with chunk 128. Conservative lr 0.002 to 0.0002 and ridge 0.01
  to 0.001 after stability pilots. Select the final update, not a favorable MC
  fluctuation. Training seed 123 is a single optimization trial.
- Saved-state evaluation: exact full-sector 3x3 energy, variance, overlap and
  squared fidelity; frozen MC seeds 321/654 for each square, 64 walkers x256
  retained steps, 4096 burn-in, thin=number of sites. Reblock energy at lengths
  16, 32, 64, 128. Errors are conditional on stationary chains and independent
  walkers; no ESS, mixing or convergence claim follows from this protocol alone.

## Resource decisions and numerical review

Full ED counts at the declared cutoff/filling are 19855 (3x3), 426 (2x3),
975338 (3x4), 49360405 (3x5), 35561166195 (4x5). The current sparse-ED estimator
budgets about 3.75 GiB for 3x4, 231 GiB for 3x5, and 210 TiB for 4x5. Only 3x4 receives
an explicit raised bound of 1 million states/8 GiB. Larger exact-reference cases
are deferred; absence of fidelity there does not mean poor fidelity.

Claude independently reviewed the ansatz extension, vectorized Jacobians, orbit
objective, supervised runner, VMC orchestration and maintenance cleanup.
Independent Burnside counts are 46 translation/D2 orbits on 2x3 and 21600 on 3x4.
Tests verify rectangular symmetry, transposed-shape covariance, serialization,
operator/local-energy parity, reference/vmap Jacobian parity, primal/dual SR
parity, and full/orbit loss and gradient parity. All 135 CPU tests pass.

The first 5x5 pilot (lr 0.02, ridge 0.001, float32, 512 initial burn-in) failed with
explosive sampled energies followed by nonfinite local energies. It produced
no checkpoint and is retained as a failure, not an energy result. A conservative
float64 pilot (lr 0.002, ridge 0.01, 4096 initial burn-in) completed 10 updates
stably. This changes several settings and does not isolate precision as a cause.

## Evidence and reproduction

[Compact numeric results](gcnn-study-results.json) retain completed cases,
the missed accuracy gate and failed pilot. Full reports, checkpoints and frozen chains are local
under `outputs/2026-11-13-gcnn-study/`, ignored by Git. Training and evaluation
sidecars record commands, source hashes, precision and physical metadata.
Exact versions of all 12 source files pinned by the training reports are
preserved under `outputs/2026-11-13-gcnn-study/source-snapshot/`. The snapshot
also retains the original launchers, tests, study plan and temporary fidelity
audit script. Historical report hashes refer to these pre-cleanup versions;
the maintained launchers differ in formatting, output-path options and metadata
handling. The `models.py` feature-shape docstring was also corrected; its
forward calculation is unchanged by this cleanup.
Subsequent Julia integration also changes sibling import wiring to support the
isolated `pythonTest.bosehubbard2d` package. Numerical formulas and the retained
benchmark records are unchanged; historical hashes still refer to the snapshot.
See [the cleanup receipt](gcnn-study-cleanup.json) for verification.

From the repository root, the new runners are:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B pythonTest/2D/benchmarks/train_rectangular_group.py \
  --Lx 2 --Ly 3 --width 8 --epochs 10000 --device cuda:1 \
  --output outputs/NEW-DIRECTORY/ed-2x3

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B pythonTest/2D/benchmarks/run_group_vmc_study.py \
  --sizes 3 4 5 --device cuda:0 --dtype float64 \
  --output-dir outputs/NEW-DIRECTORY

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -B pythonTest/2D/benchmarks/summarize_group_study.py \
  --study-dir outputs/2026-11-13-gcnn-study \
  --output outputs/gcnn-recollection/results.json
```

Choose a new output directory for a deliberate repeat; launchers refuse
existing run directories. Do not erase retained evidence to rerun a command.
These examples document usage and do not schedule new experiments.
