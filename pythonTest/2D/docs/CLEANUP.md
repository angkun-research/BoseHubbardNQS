# Documentation and cleanup disposition

## Julia integration after the completed study

The Julia bridge now supports 1D/2D device-aware scalar and batch inference and
isolated calls to the existing 2D workflows. Namespace-relative imports and
duplicate-import guards extend embedded use without moving source/evidence or
altering numerical formulas. Earlier cleanup receipts describe their own
point-in-time source versions; later import changes legitimately change current
file hashes. The GCNN source snapshot and all measured records remain intact.

See [the integration contract](JULIA_PREPARATION.md) and the
[public documentation](../../../docs/site/julia.md). The pre-integration patch
and source snapshot are local under `outputs/integration-baseline-2026-11-13/`.

## Completed GCNN accuracy study (2026-11-13)

The accepted study is recorded in [GCNN_STUDY.md](GCNN_STUDY.md) and its
[compact numeric results](gcnn-study-results.json). The production addition is
the GCNN ansatz and geometry support; shared ED, VMC, local energies, optimizers
and frozen evaluation are unchanged. No new fitting campaign is active.

- Consolidated results, protocols and limitations into one study record and
  replaced the completed research plan with a short scope record.
- Cleaned the three benchmark tools for readability, removed an unused import,
  exposed output paths, and made exact saved-state evaluation use checkpoint
  metadata. The supervised benchmark reloads its selected checkpoint before
  full-sector evaluation. No optimizer, objective or selection rule changed.
- Preserved exact copies of all 12 training-hash-pinned sources, the original
  launchers/tests/docs and temporary audit scripts in the ignored
  `outputs/2026-11-13-gcnn-study/source-snapshot/` before editing. Report hashes
  continue to identify those historical versions.
- Retained every checkpoint, raw chain, ED reference, report, failure and pilot.
  No numerical evidence was deleted. Regression tests remain permanent.

[The cleanup receipt](gcnn-study-cleanup.json) records source/evidence hashes,
verification and the unchanged numerical result record. Cleanup does not commit
or push, merge 1D/2D, alter presentations, or start new training.

## Closed representation-reduction direction (2026-11-06)

The user closed the flatten-head/pooled-CNN representation-reduction work.
Its experimental model, 11 runners/analysis scripts and four test modules
were retired from the active tree. The complete experiment evidence and
historical sources are retained in a verified, Git-ignored local archive;
this explicitly supersedes in-place retention for these two dated campaigns.
Established benchmarks, their tests, accepted slides and all 1D/Julia code
are preserved. See the [closure note](CLOSED_REPRESENTATION_STUDY.md) and
[exact cleanup receipt](closed-representation-cleanup.json).

## Group-CNN cleanup before GitHub

The group-CNN addition is prepared for commit with the numerical implementation,
ED audits, bounded training runner, permanent regression tests, API documentation
and reviewed 58-page presentation retained. No 1D or Julia code is changed.

- The renderer now reads the reviewed `slides/symmetry-deck.json` directly.
  Superseded slide definitions and replacement rules were removed;
  `slides/symmetry_diagrams.py` contains the remaining custom diagrams.
- The one-off `benchmarks/time_group_fit.py` timing smoke was retired. It is
  not needed by the model, training runner, tests, evidence collector or renderer.
  Its exact source and the previous renderer/revision sources remain with the
  old PDF snapshot under ignored `outputs/slides-snapshots/`.
- Removed 17 regenerable preview files (2,111,085 bytes). No numerical archive,
  checkpoint, report/log companion, failed attempt, review or raw chain was
  deleted. Permanent gradient, equivariance, orbit-loss and checkpoint tests
  remain in `tests/`; these are correctness checks rather than temporary runs.
- All 132 CPU tests pass. A fresh source-only export also passes with three
  expected archival-check skips and renders without local checkpoints/results.
  All 58 before/after page images at 800 pixels and extracted PDF text are
  identical. The renderer rejects an unexpected numerical-evidence hash.

Exact local deletion and equivalence receipts are under
`outputs/cleanup-plans/gcnn-cleanup-*.json` and
`outputs/cleanup-plans/gcnn-render-equivalence.json`. Current slide hashes and
review are recorded in [symmetry-quality-check.json](../slides/symmetry-quality-check.json).
Ignored research artifacts stay out of Git. Cleanup does not commit or push.

## Earlier documentation cleanup

This cleanup prepares a reviewable handoff for future Julia use. It does not
merge 1D/2D, change numerical defaults, alter the accepted PDF, or rerun training.

## Completed changes

- Replaced the long milestone-oriented root documentation with concise current
  entry points: README, STATUS, PLAN and BENCHMARKING.
- Preserved their complete previous contents and hashes under
  [history/pre-documentation-cleanup](history/pre-documentation-cleanup/).
  The archive includes superseded plans and historical links; use active docs
  for current behavior.
- Added [API.md](API.md), a machine-readable signature inventory, and
  [JULIA_PREPARATION.md](JULIA_PREPARATION.md). These describe the existing
  implementation; no new adapter is implied.
- Removed regenerable Python bytecode and temporary slide-layout previews.
  Each deletion's path, size, hash and reason is in
  [cleanup-actions.json](cleanup-actions.json).
- Added a [benchmark-tool index](../benchmarks/README.md) distinguishing public
  operations, reusable analysis and dated research/prototype code. Historical
  prototypes are not a second supported VMC implementation.

## Retention rule

Keep core source, tests, the accepted 2D PDF
and its sources, selected/unselected checkpoints, raw chains, report companions,
failure logs, frozen seeds, reference CSVs and reviews. Files whose original
paths/hashes are used by the accepted PDF or historical experiment records are
research evidence, not disposable scratch. Retain them in place so existing
reproduction commands remain meaningful. Dated `outputs/` and
`benchmarks/results/` already separate these artifacts from the public API.

Do not delete a report's NPZ companion, reset a result directory to make a
historical runner launch again, or silently choose only successful attempts.
The current PDF's hash-pinned inputs and the pre-cleanup core/root/1D hashes are
recorded in [pre-cleanup-baseline.json](pre-cleanup-baseline.json).

The independent [cleanup audit](cleanup-audit.json) records candidate and
protected-file decisions. Remaining archival actions, if any, are recorded in
`cleanup-actions.json`; an empty candidate list means no further file was
established to be safely disposable. No source relocation is needed merely to
give the API a clean entry point.

The audit verified the35 disposable deletions (2,630,607 bytes). All32 benchmark
Python files are referenced by evidence/imports or are meaningful tests, so
none was cleared for removal or relocation. Historical preview PDF/provenance
pairs also remain together. The four old root documents are archived; no
checkpoint, raw sample, experiment failure or numerical source was removed.

## Subsequently approved raw-chain retirement

The user subsequently approved removing34 unselected raw archives (300MiB),
then the remaining29 large cylinder archives (472MiB). Six small2×3 validation
chains remain (3.64MiB), together with all checkpoints, reports and logs.
This explicit approval is an exception to the general retention rule above.

The original artifact inventory is a pre-pruning snapshot. Exact deletion
receipts are under `outputs/cleanup-plans/`; retired cylinder inputs and their
hashes are recorded in
[retired-mc-chains.json](../slides/retired-mc-chains.json). Saved slide values and
summaries are unchanged and still render. Reblocking or checking mixing on
discarded cylinder samples now requires fresh sampling.

## Verification

The CPU regression log is [cpu-regression.log](cpu-regression.log). CLI help,
documentation example checks, local-link checks and protected-hash checks are
recorded with the documentation review. Julia round-trip execution is a future
integration check, not a completed validation step in this cleanup.

Future slide previews belong in ignored `outputs/` or
`slides/revision-preview/`; delete/regenerate them as needed. Keep reviewable
source and final evidence distinct from those disposable renderings.

## Intentional retirement of style templates

The user confirmed removing the four original1D PDFs because they served only
as slide-style templates. No restoration is required. Their filenames and
historical hashes are recorded in
[retired-style-references.json](../slides/retired-style-references.json).
The earlier review's missing-template item is resolved by that confirmation.

The accepted2D PDF and its historical provenance stay unchanged. The renderer
now allows these four specifically retired templates to be absent and records
their retirement in future provenance; every other missing input still fails.
The previous renderer is archived with the accepted PDF's hash. No numerical
source, result, checkpoint, 1D function, Julia adapter or model behavior changed.
