# Benchmark and research tools

The public callable interface is documented in [../docs/API.md](../docs/API.md).
Production VMC is **only `../vmc.py`**. Scripts here are analysis, launchers,
diagnostics or historical experiments; they are not a second VMC backend.

| Role | Files |
| --- | --- |
| General bounded launchers | `run_baseline.py`, `run_ed_fits.py`, `run_leakage_analysis.py` |
| Matched-reference analysis | `validate_dmrg_reference.py`, `compare_dmrg_vmc.py`, `compare_dmrg_sizes.py`, `assess_refinement.py` |
| Exact/frozen observable helpers | `periodic_observables.py`, `check_torus_ed.py` |
| Dated evidence reproduction | `run_small_vmc_proof.py`, `finalize_small_vmc_proof.py`, `run_periodic_benchmarks.py`, `run_backup_extension.py`, the two `run_*6_stable_retry.py` scripts |
| Development/performance diagnostics | `benchmark_*.py`, `calibrate_conv_sr.py`, `check_training.py`, `prepare_width128.py`, `widen_conv.py` |
| Historical ansatz/optimizer diagnostics | `check_jastrow_*.py`, `check_exact_jastrow_sr.py`, `check_minsr_frozen.py`, `minsr_prototype.py` |
| Analysis regression tests | `test_*.py` |

Retained historical scripts may contain fixed dated paths and fixed experiment
settings. Check the source before invoking them. Do not import them as a Julia
API or rerun an archived campaign during environment setup. MinSR's production
implementation is in `vmc.py`; its earlier prototype is retained only as
research history and an independent comparison reference.

`results/` is evidence: commands, metadata, source/checkpoint/input hashes,
measured summaries, failed attempts and reviews. Some retained Python sources
are pinned by these records or by the accepted PDF provenance. They cannot be
deleted or moved as scratch without breaking that chain. See
[cleanup policy and disposition](../docs/CLEANUP.md).

For current commands and selection rules, use [../BENCHMARKING.md](../BENCHMARKING.md).
Checkpoint weights and raw chains stay at the locations recorded under
`../outputs/`; a JSON report and its raw NPZ companion belong together.
