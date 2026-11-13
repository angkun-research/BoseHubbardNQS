# Closed: symmetry-inspired CNN representation reduction

Closed by the user on 2026-11-06. No further training, pruning, architecture
sweeps or SVD/RG compression experiments are planned in this direction.

## Question and outcome

The goal was to discover whether ordinary CNN training learns symmetry-related
redundancy that permits systematic reduction of the number of convolutional
channels, without imposing point-group symmetry on the filters beforehand.
Tests used the 3×3, N=9 Bose–Hubbard torus, t=1, U=16.74 and occupation cutoff4.
The flatten-head and pooled-CNN studies are both closed.

Six pooled fits (width64/256, seeds123/321/654) completed 10,000 Adam/log-MSE
updates and reached absolute ED overlaps 0.9980–0.9986. This validates those
fits, but does not establish a systematic symmetry-based compression rule.

The initial centered activation-covariance analysis measured data-dependent
responses, not redundancy of the kernel weights. Its rank≈17 under ED weights
was already present before training and must not be interpreted as learned
kernel compression. Raw-weight SVD instead uses W[H,45], retaining all five
one-hot input channels and nine kernel offsets; it excludes bias and applies
no centering, input weighting or row normalization.

| Width | Seed | Initial r99 | Fixed-final r99 | Best-by-log-MSE r99 |
| --- | --- | --- | --- | --- |
| 64 | 123 | 40 | 35 | 35 |
| 64 | 321 | 40 | 34 | 34 |
| 64 | 654 | 40 | 34 | 34 |
| 256 | 123 | 44 | 43 | 44 |
| 256 | 321 | 44 | 44 | 44 |
| 256 | 654 | 44 | 43 | 43 |

Here r99 is the smallest retained rank with at most1% discarded squared
singular-value weight. All18 initial/final/best matrices have float64 numerical
rank45. The wide models show little reduction at this cutoff; no common
strongly reduced kernel dimension was established. Intermediate raw-weight
spectra were not saved, so no kernel-rank-versus-epoch claim can be made.

A channel basis change generally cannot be moved through elementwise SiLU.
Consequently, low kernel-matrix rank does not guarantee a smaller nonlinear
width with the same wavefunction. Compression could still be studied through
other methods, but it is outside this closed direction. This is a limitation
of the tested route, not a proof that symmetry cannot support compression.

## Cleanup and retained evidence

The experimental pooled model, its public API entries, 11 experiment scripts
and four dedicated test modules were retired from the active source tree.
The seven established model families, shared VMC core, existing correctness
tests, accepted presentations, 1D code and Julia code are preserved.

A verified local archive retains all experiment sources/tests, the previous
plan and status, numeric reports, checkpoints, figures, logs and reviews:
`outputs/research-archives/2026-11-06-closed-representation-study.tar.gz`.
It is ignored by Git. The historical dated result/output directories were
removed after every archived file was verified against its SHA256 digest.
No new training, commit or push was performed during closure.

The archive's paths are relative to the historical `pythonTest/2D` directory;
`CLOSURE-MANIFEST.json` lists every file and digest. Extract into a separate
scratch directory for inspection: it includes historical model/API versions
and should not be unpacked over the current working tree.

[Compact numeric results](closed-representation-results.json) and
[cleanup receipt](closed-representation-cleanup.json) remain in the repository.

Post-cleanup verification: all132 remaining CPU regression tests pass. The
established model source and API catalog match their pre-experiment versions.
