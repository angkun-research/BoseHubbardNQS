# Benchmarks and limitations

These are retained finite-budget experiments. They establish runnable
workflows and measured finite-system accuracy, not convergence or a
universal ranking of network architectures. The tables are generated from
the recorded JSON files; building this page performs no simulation.

## 1D: eight-site exact-sector evaluation

Periodic rings, unit filling, cutoff 4, t=1, U=2 in the 1D convention
`U*n*(n-1)`, mu=0. All four models used 100 native-gradient/Adam updates,
width 32 where applicable, and training seed 123. The exact sector has
5,475 configurations; its ED energy is -7.4597647030t.

| Ansatz | Parameters | Exact model energy/site (t) | Absolute overlap |
| --- | ---: | ---: | ---: |
| FCNet | 2401 | -0.836725466 | 0.926539145 |
| JastrowNet | 130 | -0.792766683 | 0.893763211 |
| Conv1DNet | 8769 | -0.863945417 | 0.945172815 |
| RBMNet | 1352 | -0.853014611 | 0.937768034 |

The overlap column is unsquared. All four states remain above ED.
These short runs do not establish each architecture’s best achievable accuracy.

## 1D: sixteen-site frozen sampling

The same 100-update protocol was evaluated using two fresh chains per
checkpoint, seeds 456 and 789, with 128 walkers, 4,096 burn-in proposals
per walker and 256 retained steps. No exact reference was computed.

| Ansatz | Energy/site, seed456 (t) | Energy/site, seed789 (t) |
| --- | ---: | ---: |
| FCNet | -0.756641865 | -0.778314114 |
| JastrowNet | -0.813138008 | -0.823219657 |
| Conv1DNet | -0.811901748 | -0.824935675 |
| RBMNet | -0.796465755 | -0.803629518 |

These are repeated measurements of the same trained weights, not
independent training seeds. Sampling variation and unresolved mixing
prevent a precise architecture ranking.

[Full 1D protocol and interpretation](https://github.com/angkun-research/BoseHubbardNQS/blob/main/docs/VMC_BENCHMARKS.md).

## 2D: the same minimal GCNN across sizes and objectives

The width-8 GCNN has 4,993 parameters: one lifting convolution, one group
convolution, SiLU and a mean/linear scalar readout. All cases use unit
filling, cutoff 4, t=1, U=16.74 in the 2D `U*n*(n-1)/2` convention,
mu=0 and periodic boundaries. Squares have D4 symmetry; rectangles
retain D4-tied filters and D2 lattice symmetry.

Supervised fitting used 10,000 Adam/log-MSE updates. VMC used 500
centered ridge-SR/SGD updates. Each case has one training seed.

| Case | Absolute overlap | Squared fidelity | Exact energy error/site (t) |
| --- | ---: | ---: | ---: |
| 2x3 log-MSE | 0.9999999476 | 0.9999998952 | 4.5630712e-07 |
| 3x3 log-MSE | 0.9976990571 | 0.9954034086 | 0.011931558 |
| 3x4 log-MSE | 0.9842388185 | 0.9687260519 | 0.042972464 |
| 3x3 VMC | 0.9980917331 | 0.9961871077 | 0.0047942699 |

**The 3x4 fit misses the previous 0.99 absolute-overlap working
criterion.** Its loss was still improving at the fixed update budget;
the miss does not prove a capacity limit. The rectangular saved-checkpoint
fidelities were independently recomputed over the full ED sectors.

## 2D: larger-system frozen energies

Fresh measurements use seeds 321 and 654, 64 walkers, 4,096 burn-in
proposals per walker and 256 retained steps, with thinning equal to the
number of sites. The records retain energy reblocking at lengths
16, 32, 64 and 128.

| Lattice | Energy/site, seed321 (t) | Energy/site, seed654 (t) |
| --- | ---: | ---: |
| 3x3 | -0.672888752 | -0.673414446 |
| 4x4 | -0.580353488 | -0.583509129 |
| 5x5 | -0.544315734 | -0.542221964 |

No exact fidelity reference is available for 4x4 or 5x5. Remaining
local-energy variance and finite training budgets prevent a claim of
converged ground-state accuracy. Block errors are conditional estimates,
not confidence intervals or an effective-sample-size measurement.

The initial aggressive 5x5 pilot failed with nonfinite local energies.
The retained results use the conservative protocol recorded in the full
report. Exact 3x5 and 4x5 fits were replaced with 2x3 because of the
existing full-ED resource limits; a new orbit-reduced solver was deferred.

## Records and reproducibility

- [2D GCNN study and complete protocol](https://github.com/angkun-research/BoseHubbardNQS/blob/main/pythonTest/2D/docs/GCNN_STUDY.md)
- [Machine-readable 2D results](https://github.com/angkun-research/BoseHubbardNQS/blob/main/pythonTest/2D/docs/gcnn-study-results.json)
- [Machine-readable 1D results](https://github.com/angkun-research/BoseHubbardNQS/blob/main/docs/benchmarks/2026-10-30/results.json)

Detailed local checkpoints, raw chains and historical sources are kept
separately from the website. Fidelity and energy refer to the declared
finite lattice and occupation cutoff. Compare [the conventions](physics.md)
before interpreting results across dimensions.
