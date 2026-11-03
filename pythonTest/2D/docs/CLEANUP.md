# Documentation and cleanup disposition

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
