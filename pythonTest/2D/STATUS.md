# Current 2D status

The requested minimal GCNN test matrix completed on 2026-11-13. The width-8
model has 4993 parameters. Exact squared fidelities are 0.9999998952 (2x3
log-MSE), 0.9954034086 (3x3 log-MSE), 0.9687260519 (3x4 log-MSE), and
0.9961871077 (3x3 VMC). The 3x4 fit misses the previous 0.99 absolute-overlap
working criterion; its loss was still falling at the fixed budget. The 4x4/5x5
VMC runs and two frozen measurements each completed, without an exact fidelity
reference or convergence claim. Rectangles retain D4-tied local filters with
only D2 lattice symmetry. The larger rectangular exact fits were replaced by
2x3; a new orbit-reduced ED solver remains deferred. See the
[protocol and results](docs/GCNN_STUDY.md) and [scope](PLAN.md).
All 135 CPU regression tests pass. No experimental jobs remain active.

The symmetry-inspired CNN representation-reduction direction was **closed by
the user on 2026-11-06**. Experimental runners, tests and the pooled model were
retired after verified local archival. That representation-reduction direction
remains closed; the new GCNN study does not reopen it.
See [the findings and closure record](docs/CLOSED_REPRESENTATION_STUDY.md).

The earlier building/benchmarking milestone is complete. Julia now exposes
1D and 2D checkpoint inference with the same scalar/batch layout, plus 2D
training and evaluation wrappers. The numerical backends remain separate; see
the [integration contract](docs/JULIA_PREPARATION.md).
The current [presentation](BoseNQS2D-progress-2026-10-31.pdf) has55 pages,
29 main and26 supporting. Its four sections cover ED, supervised NQS,
small-system VMC and raw particle-sector extension. See the
[page guide](slides/page-guide.json) and [layout/data checks](slides/section-revision-review.json).

## Established evidence

The separate 3×3 square-symmetry study adds `GroupEquivariantConv2DNet`:
all 72 spatial operations are verified against ED, the invariant block has
338 orbit basis states, and the width-16 group CNN enforces these spatial
symmetries for any weights. Its single 10,000-update Adam/log-MSE fit gives
absolute overlap **0.9996088830**, energy error/site **0.001840578116t**, and
variance **0.3902421673t²**. It passes the agreed 0.99 working overlap criterion;
it does not pass the stricter tiny-system overlap/energy gate. See the
[symmetry study](docs/SQUARE_SYMMETRY.md) and the separate 58-page
[teaching PDF](BoseNQS2D-symmetry-progress-2026-11-02.pdf) for the construction,
reviewed results and limitations. That milestone recorded 132 passing local
CPU tests; source-only verification also passed with three expected
archival-check skips. The current expanded suite has 135 passing CPU tests.
No new VMC or larger-system claim accompanies this supervised study.

- Independent analytic/operator tests validate the ED Hamiltonian conventions.
  On the3×3 torus, N=9/cutoff4, ED gives E/t=−6.104592578442 and uniform density1;
  Gx=Gy=0.425669565922. Periodic DMRG bond limits128/256 differ from ED in total
  energy by1.93e-7/4.59e-10t.
- The four primary neural families pass the agreed supervised overlap gates
  (tiny2×3:≥0.9999 with energy/site error≤0.001t;3×3 working gate:≥0.99).
  Jastrow remains a fixed physical baseline. Fits use Adam + log-MSE.
- Fresh2×3 torus SR+nativeSGD VMC gives exact saved-state overlaps0.999919
  (FC),0.999984 (Conv2D),0.999997867 (Conv2D+Transformer). Exact density/coherence
  expectations and both frozen MC seeds321/654 are retained separately.
- The optional hybrid's supervised overlaps are0.9999999955 (2×3) and
  0.9995768953 (3×3); its3×3 energy error/site is0.001521t, reported separately
  from the working overlap gate.
- Raw3×3 sector scans enumerate all5^9 configurations. A good fixed-N fit does
  not determine raw amplitudes in other sectors. Exact projection fixes support
  without changing canonical observables; global and window probabilities differ.

## Limits retained in the presentation

Finite-size tests do not establish thermodynamic criticality or a universal
architecture ranking. Conditional MC SEM does not establish mixing. The6×4
backup table uses open-x/periodic-y cylinders against matched chi128 DMRG.
Basic Jastrow's seed654 energy gap is6.33%; RBM128 misses the neural accuracy
and precision gate at24.04%. Initial numerical failures and the single
conservative stability retry per family are preserved. No further accuracy
search is active. These optimizer outcomes do not prove capacity limits.

## Evidence locations

| Scope | Records under benchmarks/results/ |
| --- | --- |
| Small VMC and finite gaps | `2026-11-01-physics-focus/` |
| Periodic ED/DMRG, primary observables and leakage | `2026-11-01-periodic-focus/` |
| Hybrid and requested6×4 rows, retained failures/retries | `2026-11-01-backup-extension/` |

Weights and raw chains live at the paths in those records under `outputs/`.
Current PDF inputs are hash-pinned in its `.provenance.json`. Do not move or
remove them as temporary files. Earlier detailed status and campaign history
are archived in `docs/history/pre-documentation-cleanup/` and `outputs/`.
