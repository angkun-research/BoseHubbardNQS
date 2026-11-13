# Completed GCNN study and remaining scope

The requested GCNN test matrix is complete and accepted as the final study
record. Results and reproduction commands are in
[the study report](docs/GCNN_STUDY.md), with exact values in
[the compact results](docs/gcnn-study-results.json).

The same width-8 ansatz was tested using log-MSE/Adam on 2x3, 3x3 and 3x4,
and energy-based VMC/SR+SGD on 3x3, 4x4 and 5x5. Squares have D4 symmetry;
rectangles retain D4-tied filters with D2 lattice symmetry. The shared physics,
ED, sampler, local energies and optimizers are unchanged.

The 3x4 fit misses the previous 0.99 absolute-overlap working criterion.
Exact fidelity and ground-state convergence are not established for 4x4/5x5.
These limits remain part of the final record; no further fitting or architecture
search is active. The larger rectangular ED cases were replaced by 2x3, and
building a symmetry-reduced ED solver remains deferred.

The separate CNN representation/SVD compression direction remains
[closed](docs/CLOSED_REPRESENTATION_STUDY.md). Its original plan and evidence
remain in the verified local archive identified there.

The numerical 1D and 2D backends remain separate. The implemented
[Julia integration contract](docs/JULIA_PREPARATION.md) exposes scalar and batch
inference, device selection, and isolated 2D training/evaluation workflows.
This software integration does not add numerical study results.
