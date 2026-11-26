# 1D Bose–Hubbard scaling pilot

This pilot compares how much matrix-product-state (MPS) and neural quantum
state (NQS) capacity is needed to reproduce the ground-state energy and
long-distance one-body correlations of periodic, unit-filled rings. The
three couplings are $U/t=2$ (superfluid), $3.275$ (BKT), and $8$
(deep Mott), with $t=1$ and inclusive occupation cutoff `Nmax=6`.
Here $U$ multiplies $n_i(n_i-1)/2$. The legacy VMC code takes $U/2$.

- [Study design and literature](plan.md)
- [Run history, job IDs, and detailed verification](pilot-run.md)
- [Numerical summary](results/pilot-summary.json) and [MPS figure](results/mps-brackets.pdf)
- [Critical coupling and independent DMRG checks](../../CriticalBehavior/RESULTS.md)

## Accuracy criteria

The supervised ED screen uses median infidelity $1-F\le10^{-3}$ over
three seeds in each phase at both $L=8$ and $L=10$. The VMC and MPS
comparisons use the same physical thresholds against verified DMRG
references. The tolerances are $10^{-3}t$ for energy per site and
$0.05|G_{1,\mathrm{ref}}(r)|+5\times10^{-4}$ for $G_1(r)$ at
$r=L/4,L/2$, with an additional signal-bearing distance in the Mott
phase when applicable. A VMC configuration qualifies when at least two
of three training seeds pass. Each pass requires two independent
frozen-checkpoint evaluations whose simultaneous 95% intervals,
expanded for reference uncertainty, satisfy every threshold.

## MPS baseline

The brackets give the tested failing lower cap and passing upper cap
after checking sweep stability. They apply to the tolerances above.

| Sites | Superfluid | BKT | Deep Mott |
| ---: | ---: | ---: | ---: |
| 16 | (20, 24] | (20, 24] | (20, 24] |
| 24 | (24, 28] | (28, 32] | (24, 28] |
| 32 | (28, 32] | (28, 32] | (24, 28] |
| 48 | (32, 40] | (32, 40] | (20, 24] |
| 64 | (32, 40] | (40, 44] | (16, 24] |

Mott stayed at or below a cap of 28 throughout the grid. The superfluid
and BKT feasible power-law slope ranges through $L=48$ overlap.
The $L=64$ lower-cap checks were planned after earlier exploratory
upper-cap results were available. The measured split at $L=64$
leaves the two asymptotic exponents unresolved.

## Neural results

The three VMC comparison slots were FCNet, InvariantConv1D, and
ConvTransformer. FCNet was the dense baseline. The invariant convolution
qualified after a fresh $L=10$ Mott fidelity retest. ConvTransformer
had the lowest worst-phase-and-size median infidelity among the
eligible attention models. Its centered stochastic reconfiguration
runs produced the confirmed crossings below at the stated budgets.
FCNet and InvariantConv1D did not certify at the tested $L=16$
BKT settings. The invariant convolution used ConvTransformer's SR
settings without a separate optimizer calibration.

| Phase | Largest passing cohort | Boundary of the tested protocol |
| --- | --- | --- |
| Superfluid | $L=32$, 16,701 parameters, 256 updates, 2 of 3 seeds | All three $L=48$ seeds missed at 256 updates. A one-seed replay to 512 updates also missed. |
| BKT | $L=24$, 13,501 parameters, 512 updates, 3 of 3 seeds | At $L=32$, one seed passed, one correlation interval still overlapped tolerance, and the third had nonstationary chains. |
| Deep Mott | No certified cohort at the tested $L=16$ budgets | Longer fits, a wider model, and optimizer and proposal changes left the long-distance correlation outside tolerance. |

The $L=16$ superfluid and BKT cohorts and the $L=24$ superfluid
cohort each passed all three seeds.

At $L=24$ BKT, width 20 passed one seed at 256 updates and all
three after 512 updates. A separate width-24 cohort at 256 updates
passed one seed after independent precision checks. The $L=32$ BKT
mixed-training control reduced drift from the same initialization and
still missed energy and $G_1(16)$. It remains a one-seed diagnostic.
The two-point slope diagnostic for the passing $L=24$ BKT cohort was
0.140–0.306 higher than the DMRG slope. The VMC trace measures block
number fluctuations about the unit-filling mean. For a state with
nonuniform mean density, this includes a contribution from the spatial
variation in mean block number. The DMRG reference uses connected
block variances. The slope diagnostic remains outside the acceptance
criteria.

On an A40 at $L=16$ BKT, SR batches of 512 and 2,048 each passed
two of three seeds in roughly 315–328 training-update seconds per
seed. The remaining seed missed at 512 and was inconclusive at 2,048.
The 2,048-sample setting remained the pilot default after passing
all three CPU confirmation seeds.
The 8,192-sample SR arm was inconclusive at its shorter update cap,
and Adam at 2,048 missed all three seeds. The numerical summary
records measured runtime and estimated operation counts. ConvTransformer
has $Lw^2$ weights in its flattened readout, so even a fixed width
$w$ changes the allocated parameter count as the ring grows. The
pilot leaves neural capacity and compute exponents unresolved.

## Rerun individual stages

The launchers target ISAAC-NG. Use Slurm compute nodes with explicit
account, partition, QoS, time, and resource requests. Set absolute paths
for the checkout, a writable Lustre work directory, and a validated
Python environment with PyTorch, NumPy, SciPy, and Matplotlib. For DMRG,
prepare the pinned Julia project and the ITensor depots expected by
the [DMRG launcher](dmrg.sbatch).

| Stage | Entrypoint and inputs |
| --- | --- |
| DMRG observations | [`dmrg.sbatch`](dmrg.sbatch) reads one-based rows from a [cap manifest](dmrg-pilot.tsv). Run it from a Lustre directory. Set `BHNQS_DMRG_SEED` or `BHNQS_DMRG_MAX_SWEEPS` for same-cap stability checks. |
| ED fidelity | [`ed_fidelity.py`](ed_fidelity.py) generates tuning and confirmation manifests and summarizes their results. The [CPU screen launcher](ed-screen.sbatch) runs both stages at $L=8$ for the requested architecture. |
| ED-to-VMC bridge | [`ed_vmc_bridge.py`](ed_vmc_bridge.py) runs sampled and exact-energy fits from matched initializations. |
| VMC fit | [`vmc_train.py`](vmc_train.py) requires the corresponding completed $L=8$ and $L=10$ qualification reports. Set `--device cuda` only in a GPU allocation. |
| Frozen evaluation | [`vmc_evaluate.py`](vmc_evaluate.py) reads a saved weight checkpoint and writes chain-aligned series. `--steps` counts retained snapshots per chain, and `--retain-every` sets the proposal stride. |
| Reference and target reports | [`dmrg_report.py`](dmrg_report.py) loads local measurements and the optional DMRG archive. [`vmc_targets.py`](vmc_targets.py) supplies the shared threshold check and the report CLI for the original FCNet pilot. |

The ED recovery and retest launchers preserve historical replay
recipes. [`vmc-gpu-calibrate.sbatch`](vmc-gpu-calibrate.sbatch) is the
early FCNet GPU calibration. The later A40 study, DMRG refinements,
and size-specific VMC runs have launch records in the cluster archive.

For a DMRG smoke run, set `study_account` to an eligible account,
`study_dir` to an existing Lustre work directory, and `study_depots`
to the parent of the Julia depots. Submit the first manifest row from
the checkout root.

```bash
repo=$(pwd -P)
sbatch --account="$study_account" --partition=short --qos=short --time=00:30:00 \
  --cpus-per-task=1 --mem=4G --chdir="$study_dir" --array=1 \
  "$repo/ScalingStudy/1D/dmrg.sbatch" "$repo" \
  "$repo/ScalingStudy/1D/dmrg-pilot.tsv" "$study_depots"
```

Check the returned job ID with `squeue -j JOB_ID` and use
`sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed` and the log to
verify completion.

For example, the $L=24$ BKT width-20 fit used
`--arch ConvTransformer --length 24 --phase bkt --width 20 --optimizer sr
--lr .03 --sr-lambda .01 --walkers 256 --sample-steps 8 --burn-in 32
--steps 512 --checkpoint-every 256`. The three training seeds were
101, 202, and 303, with sampler seeds 10101, 10202, and 10303.
The final frozen evaluations used 128 chains, 2,048 burn-in proposals
per chain, 4,096 retained snapshots at stride eight, blocking at
32, 64, and 128 retained steps, and correlations at distances 6 and
12. Each evaluation used a fresh seed. The [run ledger](pilot-run.md)
identifies the exact late-run reports, launcher provenance, and the
cluster-local archive for weights and raw chain series.

## Follow-up experiments

1. At $L=10$ Mott, initialize exact-gradient and sampled energy training
   from the same ED-trained checkpoint. Track exact fidelity, energy, and
   $G_1$ against the archived random-start controls.
2. At $L=32$ BKT, test longer per-update burn-in and a smaller nonlocal
   proposal fraction in separate arms. Confirm any candidate on three
   fresh training seeds with independent frozen checks and chain diagnostics.
3. At $L=48$ superfluid, test width 24 at the same 256-update budget
   used for width 20. Record the cost of the wider model alongside accuracy.
4. Sweep capacity at several sizes once a protocol passes reliably.
   Record the smallest tested passing count, training GPU-seconds,
   estimated FLOPs, proposals and effective samples. Retain the failures
   and inconclusive runs when estimating threshold brackets.
