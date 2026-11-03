# Bounded cylinder benchmark

Run commands from the repository root using the existing Python and Julia
environments. Paths passed to these programs are relative to `pythonTest/2D/`.
Use fresh output names; frozen evaluation, DMRG and comparison refuse existing
destinations. Training can replace a checkpoint and its JSON sidecar, so give
each run a distinct path. The commands below document reproduction; they do
not request another campaign.

The proof stage is **4×4**, fixed particle number 16, inclusive local cutoff
`Nmax=4`, `t=1`, `U=16.74`, `mu=0`. Cylinders explicitly use open x and periodic y:
`--no-pbc-x --pbc-y`. All program defaults remain periodic on both axes for old
command compatibility; tori are separate geometries. Physical indices are
`i=y*Lx+x`; the DMRG snake traverses y within each x column. Bonds are unique
unordered neighbors without self bonds, and onsite interaction is `U*n*(n-1)/2`.

## SR with native PyTorch SGD

This bounded pilot uses 256 retained configurations per update. It is a tool
check, with no assertion that 150 updates, burn-in or thinning is sufficient.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u pythonTest/2D/train.py \
  --arch FCNet --hidden-dim 32 --hidden-layers 2 \
  --Lx 4 --Ly 4 --Ntot 16 --Nmax 4 --t 1 --U 16.74 --mu 0 \
  --no-pbc-x --pbc-y --device cpu --dtype float64 --seed 123 \
  --update-method sr --optimizer sgd --sr-solver auto --lr .02 --tau .01 \
  --epochs 150 --walkers 16 --steps 16 --burnin 512 --thin 16 \
  --output outputs/proof4x4-example/FCNet-seed123.pt
```

`--steps` counts retained steps **per walker**. `--thin 16` makes 16 proposals
between retained states; burn-in remains a proposal count per walker. Post-burn
proposal attempts per measurement are `walkers*steps*thin`, not lattice sweeps.
Default thinning is 1 and preserves old seeded behavior. SR uses centered,
sample-normalized covariance and native `torch.optim.SGD`; Adam is rejected for
SR. The sample-space solve still stores an M×P Jacobian and performs per-sample
backward passes. Resource guards are estimates, not guaranteed device peaks.

Training history measures before updates; use a separate frozen evaluation for
the saved weights. The final checkpoint and sidecar are staged, reloaded and
measured before replacement; their two replaces are not a cross-file atomic
transaction.

## Frozen observables and retained chains

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u pythonTest/2D/evaluate.py \
  --checkpoint outputs/proof4x4-example/FCNet-seed123.pt --device cpu --seed 321 \
  --walkers 64 --steps 256 --burnin 2048 --thin 16 --block-length 32 \
  --reference-sites 9 10 --output outputs/proof4x4-example/frozen-seed321.json
```

The checkpoint supplies geometry, dtype and physical settings. Evaluation does
no optimizer steps. JSON and its `.samples.npz` companion stay under `outputs/`.
NPZ preserves `(retained time, walker, physical site)` occupations, ordered
physical pairs, raw one-body estimators, densities, factorial moments, local
energies and retained proposal indices. Keep these files to check mixing and
blocking without conflating walkers with successive times.

`rho[i,j]=<b_i^dag b_j>` uses the row-action estimator; neither triangle is
silently symmetrized. Selected reference rows also include reverse entries,
diagonals and directed Hamiltonian bonds. For matched bulk comparisons on this
4×4 cylinder, restrict both endpoints to x=1 or x=2 and compare the same physical
pairs. Sites 9 and 10 are interior reference sites; their full rows also contain
boundary endpoints. Full rho (`--full-rho`) is an explicit small-system option.

Check occupation-change rates, adjacent retained equality, autocorrelation,
independent chains and block-length stability. Thinning spends more proposals;
it does not establish equilibration. Conditional SEM assumes stationary,
independent chains and adequately separated time blocks. Incomplete tails,
complete-block means and all-sample means are recorded separately. Noisy VMC
pilots or a stationary log-MSE fit alone do not establish an ansatz capacity
limit.

## Standalone finite-chi DMRG reference

Use the existing root Julia environment; no package updates or installation:

```bash
./julia-local --threads=1 pythonTest/2D/dmrg_reference.jl \
  --Lx 4 --Ly 4 --Ntot 16 --Nmax 4 --t 1 --U 16.74 --mu 0 \
  --no-pbc-x --pbc-y --seed 123 --chi 128 --sweeps 6 --cutoff 1e-9 \
  --spot-chi 256 --spot-sweeps 2 --output outputs/proof4x4-example/dmrg
```

The chi256 spot continues the chi128 MPS for two sweeps. TOML records settings,
versions, source hashes and phase metadata; CSV records energy, physical
coordinates, density, onsite factorial moments, full rho, truncation diagnostics
and runtime. The expensive H-squared variance is omitted by default and labeled
omitted; `--variance` is intended for explicit tiny validation. Times include
imports/JIT where stated, and are distinct from Monte Carlo uncertainty.

The existing validated run is
`outputs/dmrg-4x4-cylinder-chi128-spot256/`; reuse it for matching comparisons.
Chi128 and chi256 energies/site are -0.479479595729 and -0.479481232365.
Their difference is 1.64e-6/site; matched interior rho differs by at most 1.67e-5.
These are finite-chi sensitivity observations, not rigorous error bars. The
independent 3×3 cylinder validation found total-energy error 4.27e-10 and full
rho maximum error 3.08e-9 against ED. Persistent details are in
`benchmarks/results/checker-dmrg-reference-report.json`.

## Compare frozen records with one DMRG phase

```bash
.venv/bin/python -B pythonTest/2D/benchmarks/compare_dmrg_vmc.py \
  --reference outputs/dmrg-4x4-cylinder-chi128-spot256 --phase chi128 \
  --reports outputs/proof4x4-example/frozen-seed321.json \
  --output benchmarks/results/proof4x4-example-comparison.json
```

Multiple report paths may follow `--reports`; output order follows input order.
Without `--phase`, the last recorded phase is selected (chi256 in this reference).
The helper rejects mismatched Lx, Ly, Ntot, Nmax, t, U, mu, per-axis boundaries,
physical indexing, Hamiltonian convention or unique-bond convention. It retains
reference chi metadata and all supplied directed pair indices.

Energy/site and signed error/site use complete-block means when available;
all-sample energy/site is separately labeled. Density and indexed rho differences
carry the elementwise maximum of available block and between-chain SEMs; absent
estimates remain null. Reference DMRG systematic error is not included. The
output makes no automatic ground-state, fidelity, convergence or architecture
ranking claim.

Stay at 4×4 for this proof stage with the one chi256 spot. Consider 6×4 and then
8×4 only if needed and separately authorized; stop size growth when energy/site
and matched bulk correlations are stable within assessed uncertainty. Do not
infer a thermodynamic transition from this bounded reference exercise.
