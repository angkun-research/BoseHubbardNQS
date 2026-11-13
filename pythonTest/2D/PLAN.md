# 2D research status and scope

The symmetry-inspired CNN representation-reduction direction is **closed**
by the user as of 2026-11-06. This includes flatten-head/pooling comparisons,
feature/kernel SVD analysis and proposed SVD/RG channel truncation.
No further experiments are scheduled in this direction.

The goal, findings, limitations and cleanup are recorded in
[the closure note](docs/CLOSED_REPRESENTATION_STUDY.md). The complete previous
research plan is preserved in the ignored local closure archive identified
there; it is historical scope, not authorization for future runs.

The established 2D ED, supervised-learning, VMC and square-symmetry benchmarks
remain available; see [STATUS.md](STATUS.md) and [BENCHMARKING.md](BENCHMARKING.md).
The seven established model families and shared PyTorch VMC implementation
remain supported. No new direction has been selected.

All 1D and Julia functionality remains separate and unchanged. Future Julia
integration is prepared in [docs/JULIA_PREPARATION.md](docs/JULIA_PREPARATION.md),
but has not been merged or initiated by this closure.
