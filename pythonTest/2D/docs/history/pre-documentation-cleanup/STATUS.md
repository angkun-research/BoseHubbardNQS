# Periodic 2D NQS benchmark status

The main presentation uses **periodic boundaries in both x and y**. Its goals
are ED construction checks, supervised representation, small-system stochastic
VMC validation, and raw neighboring-sector amplitudes. Larger-cylinder studies
remain supporting context. The bounded, user-requested backup benchmark runs
are complete: Jastrow/RBM6×4-cylinder rows and one1D-style convolution
plus Transformer ansatz with periodic small-system benchmarks. The existing
main periodic results remain frozen; no architecture or optimizer grid is run.
All code and artifacts remain under2D; the existing1D code is unchanged.

## Requested backup extension

`Conv2DTransformerNet` extends the existing1D ConvTransformer using a3×3
convolution, SiLU, one four-head encoder and flattened dense/SiLU head.
Width32, FF64, ReLU and zero dropout give a deterministic wavefunction;
there are no positional embeddings or imposed translation invariance.
The original five remain the default benchmark sweep.

| Hybrid benchmark | Absolute overlap | Energy error per site / t |
| --- | ---: | ---: |
|2×3 torus, Adam+log-MSE10000|0.9999999955|3.51e-8|
|3×3 torus, Adam+log-MSE10000|0.9995768953|1.52e-3|
|2×3 torus, fresh SR+nativeSGD500|0.9999978673|4.55e-5|

The fresh VMC exact energy is `-2.580816327645*t`, versus ED
`-2.581089511910*t`; exact variance is `0.019807929*t^2`.
Both frozen seeds321/654 include density, onsite factorial moment, full
one-body matrix and NN composite statistics. No supervised warm start or ED
labels enter VMC. The3×3 fit passes the working overlap gate≥0.99; its energy
error is reported separately. The separate supervised3×3 raw-sector scan enumerates all
`5^9` configurations: `Pfull(9)=0.00578164258`, despite its high canonical
overlap. Full and window-conditional probabilities remain distinct.

The6×4-cylinder comparison uses the same chi128 DMRG reference as the earlier
rows: `E/site=-0.509061402964*t`. Predeclared frozen seed654 gives:

| Added row | Parameters | E/site ± conditional SEM / t | Relative energy error |
| --- | ---: | ---: | ---: |
|Basic Jastrow|430|−0.476818 ±0.001323|6.33%|
|RBM128|15608|−0.386679 ±0.002684|24.04%|

Jastrow retains its original displacement ansatz and baseline exemption. RBM
misses the neural point/precision gate. Each first pilot became nonfinite;
one explicitly reviewed fresh stability retry per family used smaller steps
(`eta=.002→.0005`) and fixed larger ridge (`tau=.01`), with1000updates and
both frozen seeds retained. The failed logs and original protocol are preserved.
These finite-budget optimizer outcomes do not establish architecture capacity
limits, and no further accuracy search was run. See the
[backup records](benchmarks/results/2026-11-01-backup-extension/).

## Independent periodic reference

The3×3 torus has N=9, cutoff4,18unique bonds (nine per direction) and19,855
canonical basis states. Independent occupation-ladder ED gives
`E/t=-6.1045925784417845`, residual `8.30e-14*t`, uniform density1, and
`Gx=Gy=0.425669565922`. The fixed-N neutral gap is `9.015296519740*t`;
this finite cluster does not establish thermodynamic critical scaling.

Fully periodic DMRG was checked against ED with maximum bond limits128/256:

| Bond limit | Total energy error / t | Max full one-body matrix error |
| --- | ---: | ---: |
|128|1.93e-7|1.35e-7|
|256|4.59e-10|3.54e-10|

The refined MPS actually uses maximum bond dimension156 at cutoff1e-12.
An open-chain MPS is the representation; the Hamiltonian includes every
physical x/y wrap. No physical boundary was opened. See the
[periodic reference review](benchmarks/results/2026-11-01-periodic-focus/checker-periodic-reference-review.json).

## NQS and observable checks

Primary Adam+normalized-log-MSE fits use existing trained width32 models, with
checkpoints selected before observable/leakage analysis. Absolute overlaps:

| Ansatz |2×3 torus|3×3 torus|
| --- | ---: | ---: |
| FC | 0.999998 | 0.997320 |
| Conv2D | 0.999998 | 0.997997 |
| RBM | 0.999947 | 0.995000 |
| Transformer | 0.999995 | 0.997980 |
| Jastrow | 0.997298 | 0.992349 |

Jastrow stays a physical baseline. These overlaps describe finite-size
approximations, not exact equality or universal ansatz capacity.

The existing fresh2×3-torus SR+nativeSGD500-update runs reach exact-sector
energies `-2.573496636*t` (FC) and `-2.579806152*t` (Conv2D), compared with
ED `-2.581089512*t`. Absolute overlaps are0.999919 and0.999984. Their fullrho
and density data are benchmarked against ED, with exact model expectations
separated from frozen Monte Carlo measurements.

Main plots now include site-density residuals, every3×3 NN bond, direction
averages for2×3 VMC, and onsite factorial moment D=<n(n-1)>. Uniform mean
density alone is insufficient: e.g. Jastrow preserves uniform density but has
nonzero NN/onsite errors. All MC composites are formed per(time,walker) before
blocking; errors retain covariance and are conditionalSEM, not mixing proofs.
Both frozen seeds321/654 are retained; plots use predeclared654.

See [periodic observable data](benchmarks/results/2026-11-01-periodic-focus/periodic-observable-benchmarks.json)
and [independent review](benchmarks/results/2026-11-01-periodic-focus/checker-periodic-observable-review.json).
The short2×3 torus has three unique x bonds and six y bonds; its two directions
need not have identical coherence. All comparisons match that convention.

## Sector leakage

The3×3 periodic primary checkpoints were independently evaluated on all
`5^9=1,953,125` cutoff4 configurations, covering N=0,…,36 in CPUfloat64.
Raw full-space probabilities, conditional N=7,…,11 probabilities and nearby
global leakage are distinguished. High within-sector overlap does not enforce
zero raw output elsewhere. Exact projection imposes P(N=9)=1 without changing
canonical observables; h(N) with h(9)=0 can alter raw sector weights while
preserving the fitted canonical state.

See [periodic leakage data](benchmarks/results/2026-11-01-periodic-focus/periodic-leakage-results.json)
and [independent review](benchmarks/results/2026-11-01-periodic-focus/checker-periodic-leakage-results-review.json).
These replace the earlier cylinder checkpoints in the main presentation;
old cylinder data are preserved and are not relabeled as periodic results.

## Presentation and records

The [updated49-page PDF](BoseNQS2D-progress-2026-10-31.pdf) has28mainpages and
21supporting pages. Main observable benchmarks are15–17; periodic DMRG/ED is5;
raw-versus-projected weights are22. Both MC seeds and local moments/correlations
are tabulated on43. Cylinder results are supporting pages41–42; page42 now
includes both requested baseline rows. The new hybrid architecture and measured
benchmarks occupy44–49. Only page42 changed among the original43pages.

[Final PDF review](benchmarks/results/2026-11-01-backup-extension/checker-backup-slides-final-review.json),
[backup numerical review](benchmarks/results/2026-11-01-backup-extension/checker-backup-extension-results-review.json),
[reproduction instructions](slides/README.md), [page guide](slides/page-guide.json)
and [quality record](slides/progress-quality-check.json) identify the current
artifact. Earlier PDFs/sources are retained in ignored outputs/slides-snapshots/;
earlier detailed status is archived at `outputs/status-history/STATUS-before-periodic-6356c50727308b85.md`.
Historical campaign details remain in PLAN.md and their result folders.
