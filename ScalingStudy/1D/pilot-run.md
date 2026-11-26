# ISAAC-NG pilot

See the [pilot summary](README.md) for the final findings and workflow
entry points. This ledger preserves the order of the runs and checks.

Artifact names in this ledger are relative to the ISAAC-NG Lustre
run directory `scaling-1d-20261123`. It holds checkpoint weights,
per-chain traces, launch records, and logs. The early CPU submissions
used partition `short` with QoS `short`. Later entries record
changes in partition and resources.

| Slurm ID | Work |
| --- | --- |
| 6652940 | Exact-fidelity $L=6$ smoke across eight adjustable candidates. Completed with exit code 0. |
| 6652957 | Exact-fidelity $L=8$ timing calibration using the validated smoke environment. All eight trials completed with exit code 0. |
| 6653003 | DMRG backend check at $L=16$, $U_{\rm phys}/t=2$, and $\chi_{\max}=16$. Completed with exit code 0 and converged sweeps. |
| 6653005 | Lower-bond DMRG array after the backend check. |
| 6653004 | Higher-bond DMRG references and cutoff checks after the backend check. |
| 6653227 | DMRG coverage report after both arrays reach a terminal state. |
| 6653007 | $L=8$ ED-to-VMC bridge smoke with sampled and exact-energy arms and a frozen-checkpoint reload. Completed with exit code 0. |
| 6653047 | $L=8$ FCNet screen. Completed and passed all three phases, with median infidelities $7.67\times10^{-5}$, $4.77\times10^{-5}$, and $5.25\times10^{-5}$ in SF, BKT, and MI order. |
| 6653048 | $L=8$ RBMNet screen. Completed and passed SF, with median infidelity $3.80\times10^{-4}$. The BKT and MI medians were $1.20\times10^{-3}$ and $6.77\times10^{-3}$. The MI miss exceeds the retest ceiling, so this RBM variant does not advance. |
| 6653055 | Frozen-checkpoint VMC observable-evaluation smoke at $L=8$. Completed with exit code 0. |
| 6653080 | $L=8$ legacy Conv1DNet screen. Completed and passed all three phases, with median infidelities $6.89\times10^{-5}$, $9.76\times10^{-5}$, and $1.85\times10^{-4}$ in SF, BKT, and MI order. |
| 6653091 | $L=8$ translation-invariant convolution screen. |
| 6653092 | $L=8$ Conv1DPoolNet screen. |
| 6653093, 6653094, 6653095 | $L=8$ TransformerNet, ConvTransformer, and ViTNet tuning arrays. |
| 6653132 | TransformerNet confirmation array after tuning. |
| 6653136 | ConvTransformer confirmation array after tuning. |
| 6653137 | TransformerNet screen report. All three $L=8$ medians passed the tight target, at $4.28\times10^{-5}$, $4.45\times10^{-5}$, and $5.95\times10^{-5}$ in SF, BKT, and MI order. |
| 6653167 | ViTNet confirmation array after tuning. |
| 6653168 | ConvTransformer screen report. All three $L=8$ medians passed the tight target, at $7.41\times10^{-5}$, $3.85\times10^{-5}$, and $9.51\times10^{-5}$ in SF, BKT, and MI order. |
| 6653169 | ViTNet screen report after confirmation. |
| 6653228 | Interim DMRG coverage and reporter verification while the arrays run. |
| 6653177 | Combined $L=8$ gate report after the architecture-specific confirmations. |
| 6653182 | FCNet $L=10$ tuning after the $L=8$ pass. |
| 6653229, 6653230 | FCNet $L=10$ confirmation and gate report after tuning. |
| 6653233 | First $L=16$ FCNet VMC and frozen-evaluation pilot, conditional on a pass in all three $L=10$ phases. |
| 6653286 | Longer-sweep DMRG reruns at lower bond caps to tighten the $L=16,32$ threshold brackets. |
| 6653326 | Joint physical target report after the converged DMRG report and conditional FCNet VMC pilot. |
| 6653331 | Standalone DMRG bond-dimension figure after the coverage report. |
| 6653336 | Interim DMRG report and figure after the lower-cap refinement array. |
| 6653381 | Legacy Conv1DNet $L=10$ small-system tuning for comparison with the invariant convolution. |
| 6653437, 6653439, 6653441 | ConvTransformer $L=10$ tuning, confirmation, and gate report. |
| 6653484, 6653485 | Legacy Conv1DNet $L=10$ confirmation and gate report. |
| 6653567, 6653568, 6653569 | TransformerNet $L=10$ tuning, confirmation, and gate report. |
| 6653707 | Longer lower-cap sweeps at $L=32,64$ for the two gapless points. |

The first DMRG array, 6652933, used the default Julia depot and failed to
load ITensors. The remaining tasks and its dependent report 6652951 were
cancelled. The retry uses the same Julia module and Lustre depots as the
prior [criticality study](../../CriticalBehavior/RESULTS.md).
The preview 6653170 failed because the compute node lacked the login node's
Python path. Its retry and the replacement for the queued report 6653006
use the validated Python environment on Lustre.

The [DMRG manifest](dmrg-pilot.tsv) uses one-based row numbers for the array.
The [ED smoke](ed-smoke.sbatch), [timing calibration](ed-calibrate.sbatch)
and [per-architecture screen](ed-screen.sbatch) run on CPU compute nodes.
The [split screen](ed-screen-split.sbatch) runs each attention tuning
band as its own Slurm task and uses disjoint seeds for confirmation.
Use `squeue -j JOB_ID` for queue state and
`sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed` with the job logs to
verify outcomes.

## Interim MPS result

Preview job 6653336 completed with exit code 0 and produced
`dmrg-preview-6653336.json` and `dmrg-preview-6653336.pdf`.
These are threshold brackets over tested caps at the selected energy and
correlation tolerances. See the [final bracket table](README.md#mps-baseline)
for the resolved values after the additional cap and sweep checks.

| Ring length | Superfluid | BKT | Mott insulator |
| --- | --- | --- | --- |
| 16 | $16<\chi_\star\le32$ | $16<\chi_\star\le32$ | $16<\chi_\star\le32$ |
| 32 | $\chi_\star\le64$ | $\chi_\star\le64$ | $16<\chi_\star\le32$ |
| 64 | Reference pending | Reference pending | $16<\chi_\star\le32$ |

The matched superfluid cutoff checks at cap 256 found absolute shifts from
`Nmax=6` to `Nmax=8` below $3.2\times10^{-9}t$ in energy per site and
$7.2\times10^{-7}$ in the checked correlations at $L=16,32$.
The gapless upper bounds at $L=32$ leave smaller caps unresolved at this
snapshot. Report 6653227 waits for the wider brackets and high-cap references.

## November 24 scheduler recovery

The grouped $L=10$ tuning arrays timed out. Their confirmation jobs could
never satisfy their `afterok` dependencies, and their report jobs were
waiting behind those confirmations. Recovery retains all 69 completed
tuning results and runs only the 39 missing trials with
[one trial per Slurm task](ed-screen-recover.sbatch). Interrupted trial
directories are retained under sibling names ending in
`.interrupted-JOB_ID`. Confirmation also uses one trial per task, with
longer allocations on `campus` for the three attention models. Both
downstream stages use `--kill-on-invalid-dep=yes`.

| Model | Missing tuning trials | Replacement tuning | Confirmation | Report |
| --- | ---: | ---: | ---: | ---: |
| ConvTransformer | 13 | 6655431 | 6655454 | 6655459 |
| TransformerNet | 7 | 6655432 | 6655455 | 6655460 |
| InvariantConv1D | 2 | 6655433 | 6655456 | 6655461 |
| ViTNet | 17 | 6655434 | 6655457 | 6655462 |

The eight obsolete confirmation and report jobs were cancelled. FCNet
completed its $L=10$ gate and its first $L=16$ VMC pilot. Legacy Conv1DNet
also completed its $L=10$ confirmation array.

## Verified completion

The submitting user's queue was empty at the November 24 check. All 39 replacement
tuning tasks, all 36 confirmation tasks, and all four replacement reports
completed with Slurm exit code 0.

FCNet, Conv1DNet, and ConvTransformer passed the $L=10$ fidelity gate in all
three phases. InvariantConv1D, TransformerNet, and ViTNet missed only the
Mott threshold and are eligible for the bounded retest. Their Mott median
infidelities were $1.43\times10^{-3}$, $1.59\times10^{-3}$, and
$1.23\times10^{-3}$, respectively.

The completed DMRG report and figure are `dmrg-coverage.json` and
`dmrg-bond-dimension.pdf`. The smallest passing achieved bond dimensions
at $L=16,32,64$ were $32,32,64$ at both gapless points and $32,32,32$ in
the deep Mott phase. The first FCNet $L=16$ VMC pilot completed and
missed the joint physical target in all three phases.

## Fidelity diagnosis and VMC calibration

All nine Mott confirmations for the three near misses stopped at 3,000
updates at the approximately 8,000-parameter band. The 32,000-parameter
tuning trials reached their FLOP cap after fewer updates, so they do not
settle the capacity question. The bounded retest holds the width, learning
rate, seeds, and original compute hosts fixed. It replays each original
trajectory, verifies its checkpoint scores, and extends to at most 6,000
updates. The replay cost is reported separately.

| Slurm ID | Work |
| --- | --- |
| 6658550 | Generated and validated the nine-trial Mott retest manifest. |
| 6658553–6658561 | Same-width Mott replay and extension. |
| 6658564 | Original merged report, cancelled after a failed prerequisite. |
| 6658551 | V100S VMC calibration at BKT. Completed with exit code 0. Sampling and 100 updates took 8.37, 15.77, and 50.99 seconds at 512, 2,048, and 8,192 retained configurations per update. The original three-CPU run took 3.70 seconds at 512. |
| 6658565 | Three-CPU BKT VMC control with 1,000 updates at the pilot's width and sampling settings. Completed with exit code 0. The two frozen energy estimates were approximately $-0.959t$ per site versus $-0.951t$ after 100 updates and $-1.055t$ for DMRG. The evaluation intervals overlap across the two budgets. |
| 6658598 | Fixed-width, fixed-update BKT GPU comparison at 512, 2,048, and 8,192 retained configurations per update. Completed with exit code 0. The 8,192-sample run reached about $-0.993t$ and $-1.000t$ per site in frozen evaluations, versus $-1.055t$ for DMRG. It still missed the joint target. |
| 6661843 | InvariantConv1D seed 101 retry. Failed with exit code `2:0` after 16m 55s. The original-host retry reproduced the first replay attempt's step-750 infidelity of `0.00970051035255`, which differs from the archived `0.00969435725363`. |
| 6661844 | TransformerNet seed 303 retry after wall-time expiration, with a 10.5-hour allocation requested. |
| 6661845 | First replacement summary. Cancelled while pending after the condo replacement was accepted. |
| 6663940 | Strict-replay summary on `condo-cbatist2`. Cancelled when the four remaining campus replays were replaced with fresh fits. |
| 6663967_0–6663967_5 | Fresh $L=10$ Mott fits on `condo-cbatist2` with QoS `condo`. Tasks 0–2 are TransformerNet and 3–5 are ViTNet, each with seeds 101, 202, and 303. All six completed with exit code `0:0`. |
| 6663976 | Fresh-fit summary. Completed with exit code `0:0` and wrote `ed-l10-mott-fresh-fit-report.json`. |

The original $L=16$ FCNet VMC pilot used three CPU threads, 100 Adam
updates, and 64 batched walkers with eight retained steps per walker
per update. The independent frozen evaluations already put its energy
error well above tolerance. The new CPU control isolates the update
budget while the GPU calibration measures runtime across sampling batches.

Four of the nine strict retest trials completed successfully. Slurm rejected
the four pending fixed-host replays on `condo-cbatist2`. The required original
hosts were `clr0825` and `clr0821` for ViTNet and `ber0726` for TransformerNet.
On November 25, after the condo accepted the replacement array, the worker
cancelled the four pending campus jobs and their dependent summary.
The two completed InvariantConv1D extensions individually crossed the
Mott fidelity threshold. The merger requires every planned seed for an
architecture to produce a validated result, so the repeated seed 101
mismatch leaves its official retest classification pending.

The condo runs start from the same seeded initialization and use the same
width, learning rate, objective, CPU thread count, and extended FLOP and
update ceilings. Each uses one uninterrupted optimizer trajectory with a
score at update 3,000 and a cap of 6,000 updates, subject to the existing
early-stop rule. Read the within-run scores and each fresh three-seed cohort
separately from the archived exact replays. The task manifest is
`ed-l10-mott-fresh-fit-manifest.json` and outputs are under
`ed-l10-mott-fresh-fit/` in the existing Lustre run directory.

All six fresh fits reached 6,000 updates. Both cohorts passed the Mott
threshold on the median best recorded infidelity by 6,000 updates.

| Architecture | Median $1-F$ at 3,000 | Median best $1-F$ through 6,000 | Seeds at or below $10^{-3}$ |
| --- | ---: | ---: | ---: |
| TransformerNet | $1.592\times10^{-3}$ | $7.218\times10^{-4}$ | 2/3 |
| ViTNet | $1.228\times10^{-3}$ | $4.882\times10^{-4}$ | 3/3 |

TransformerNet seed 202 reached its best $1-F=0.06296$ at update 3,000
and ended at $1-F=0.47715$. The other two seeds crossed the gate at the
same width. ViTNet crossed the gate on all three seeds. These results show
that extending the update budget is sufficient to meet the cohort threshold
at the tested width in this Mott case.

## BKT bridge and invariant-convolution confirmation

ConvTransformer is the attention finalist under the worst-median rule.
Its worst median across the two ED sizes and three phases was
$1.384\times10^{-4}$. The corresponding worst median for ViTNet was
$4.882\times10^{-4}$ after the fresh Mott extension. FCNet anchors the
VMC comparison. InvariantConv1D gets a complete fresh Mott cohort to
resolve its eligibility for the invariant convolution slot.

The fresh InvariantConv1D cohort completed with Mott infidelities
$8.35\times10^{-4}$, $8.75\times10^{-4}$, and $8.57\times10^{-4}$.
With the original SF and BKT confirmations, its fresh qualification
report `ed-l10-invariant-fresh-vmc-qualification-report.json` passes
the three-phase $L=10$ gate. The fixed-host replay history remains
available in its original artifacts.

| Slurm ID | Work |
| --- | --- |
| 6666223_0–6666223_2 | Fresh InvariantConv1D $L=10$ Mott fits on `condo-cbatist2`, with the original seeds and fixed-width extended budget. All three completed with exit code `0:0`. |
| 6666224_0–6666224_1 | Eight-update $L=10$ BKT bridge timing on `short` for FCNet and ConvTransformer. Both tasks completed with exit code `0:0`. |
| 6666226 | Fresh InvariantConv1D summary on `condo-cbatist2`. Completed with exit code `0:0` and wrote `ed-l10-mott-invariant-fresh-fit-report.json`. |
| 6666235_0–6666235_5 | Paired $L=10$ BKT bridge on `short`, with sampled Adam at 512 retained configurations per update and exact-sector Adam from the same initial weights. All six completed with exit code `0:0`. |
| 6666236_0–6666236_5 | Sampled-only bridge control on `short`, with 8,192 retained configurations per update and the same initializations as the paired bridge. All six completed with exit code `0:0`. |
| 6666252 | Centered SR CPU checks on `short`. Completed with exit code `0:0`; primal and dual solves agreed within $8.1\times10^{-14}$ in the smoke cases. |
| 6666251 | V100S SR validation and short matched $L=16$ training smoke. Cancelled while pending after an earlier GPU route became available. |
| 6666253_1–6666253_11 | Periodic DMRG reference fits at $L=24,48$ and caps 256 and 384 on `short`, reusing the archived $L=24$ BKT cap-256 fit. All 11 tasks completed with exit code `0:0`. |
| 6666269, 6666271 | High-batch and paired bridge summaries on `short`. Both completed with exit code `0:0` and wrote `high-6666236-report.json` and `production-6666235-report.json` under `ed-l10-bkt-bridge/`. |
| 6666270 | DMRG reference summary. Completed with exit code `0:0` and wrote `dmrg-l24-l48-reference-report-6666253.json`. |
| 6666281 | ConvTransformer Jacobian smoke on CPU. Completed with exit code `0:0`; the vectorized and individual Jacobians agreed within $2.38\times10^{-7}$. |
| 6666282 | ConvTransformer Jacobian and training smoke on V100S. Cancelled while pending after an earlier GPU route became available. |
| 6666321_0–6666321_9 | $L=16$ BKT optimizer calibration on `short` with seed 707, 2,048 samples per update, and 96 updates. All ten completed with exit code `0:0`. |
| 6666336 | Optimizer calibration summary. Completed with exit code `0:0` and wrote `vmc-l16-bkt-sr-calibration-6666321-report.json`. |
| 6666341, 6666342 | Supplemental verification reports for the paired and high-batch bridge arrays. Both completed with exit code `0:0`. |
| 6666363_0–6666363_11 | $L=16$ BKT physical confirmation of the fixed Adam and SR settings on Cascade Lake CPUs in `short`. All 12 tasks completed with exit code `0:0`. |
| 6666380 | CPU confirmation summary on `short`. Completed with exit code `0:0` and wrote `vmc-l16-bkt-cpu-confirm-6666363-report.json`. |
| 6666422_0–6666422_5 | Fixed-optimizer width comparison on `short`. Uses FCNet width 82 and ConvTransformer width 24, with three SR seeds per model. All six completed with exit code `0:0`. |
| 6666429_0–6666429_5 | Fresh higher-precision frozen evaluations of all three original ConvTransformer width-16 SR checkpoints, with two evaluation seeds per checkpoint. All six completed with exit code `0:0`. |
| 6666438 | Summary of the width comparison and its additional training cost. Completed with exit code `0:0` and wrote `vmc-l16-bkt-cpu-wide-sr-6666422-report.json`. |
| 6666439 | Standalone higher-precision summary. Completed with exit code `0:0` and wrote `vmc-l16-bkt-ct16-sr-stage2-6666429-report.json`. |
| 6666466_0–6666466_5 | Matched higher-precision frozen evaluations of the three width-24 ConvTransformer SR checkpoints. All six completed with exit code `0:0`. |
| 6666477 | Standalone higher-precision summary for width 24. Completed with exit code `0:0` and wrote `vmc-l16-bkt-ct24-sr-stage2-6666466-report.json`. |
| 6666478 | FCNet width-82 same-setting training budget diagnostic with seed 101 and 768 updates. Completed with exit code `0:0`. |
| 6666490_1–6666490_18 | DMRG threshold bracket runs for $L=24,48$ at caps 16, 32, and 64 in all three phases. All 18 completed with exit code `0:0`. |
| 6666491_1, 6666501 | $L=48$ superfluid cap-256 stability repeat and its summary. Both completed with exit code `0:0`. |
| 6666502 | FCNet width-82 training budget diagnostic summary. Completed with exit code `0:0` and wrote `vmc-l16-bkt-fc82-sr-768-6666478-report.json`. |
| 6666505 | DMRG threshold bracket summary for the two additional lengths. Completed with exit code `0:0` and wrote `dmrg-l24-l48-brackets-6666490.json`. |
| 6666564_0–6666564_5 | ConvTransformer width-24 SR confirmation in the $L=16$ superfluid and Mott phases, three seeds per phase. All six completed with exit code `0:0`. |
| 6666570 | Summary of the two new phases under the same joint acceptance rule. Completed with exit code `0:0` and wrote `vmc-l16-ct24-sr-sf-mott-6666564-report.json`. |
| 6666622_1–6666622_10 | Same-cap 28-sweep stability checks for ten unresolved $L=24,48$ lower-cap runs. All ten completed with exit code `0:0`. |
| 6666633 | Updated MPS threshold report after the same-cap checks. Completed with exit code `0:0` and wrote `dmrg-l24-l48-brackets-6666622.json` and its PDF. |
| 6666646_0–6666646_2 | InvariantConv1D width-66 SR at $L=16$ BKT, with the ConvTransformer width-24 training and frozen evaluation settings. All three completed with exit code `0:0`. |
| 6666649 | InvariantConv1D width-66 gate and cost summary. Completed with exit code `0:0` and wrote `vmc-l16-bkt-inv66-sr-6666646-report.json`. |
| 6666655_0–6666655_3 | Single-seed ConvTransformer width-24 Mott SR calibration of learning rate and damping, with the original optimizer setting as a matched control. All four completed with exit code `0:0`. |
| 6666659 | Mott calibration summary. Completed with exit code `0:0` and wrote `vmc-l16-ct24-mott-sr-calibration-6666655-report.json`. |
| 6666678_1–6666678_12 | Midpoint MPS cap tests through $L=48$, using cap 24 inside the $(16,32]$ brackets and cap 48 in the $(32,64]$ gapless brackets. All 12 completed with exit code `0:0`. |
| 6666689 | First midpoint summary. Failed with exit code `1:0` because its reader expected a different reference metadata schema. |
| 6666766 | Corrected midpoint summary using the existing DMRG outputs. Completed with exit code `0:0` and wrote `dmrg-l16-l48-midpoint-report-6666678.json`. |
| 6666736_0–6666736_2 | Paired $L=10$ Mott ConvTransformer bridge, with exact-sector and sampled 512-configuration Adam arms on the same three initializations. All three completed with exit code `0:0`. |
| 6666741 | Paired Mott bridge summary and comparison with the archived ED fits. Completed with exit code `0:0` and wrote `ed-l10-mott-bridge/production-6666736-verified-report.json`. |
| 6666744, 6666750 | InvariantConv1D width-66 single-seed Mott diagnostic with the local sampler and its summary. Both completed with exit code `0:0`. |
| 6666746 | Mixed-proposal Slurm smoke. Completed with exit code `0:0`. The enumerated 19-state sector verified a symmetric proposal and detailed balance to $2.17\times10^{-19}$. |
| 6666747, 6666751 | ConvTransformer width-24 single-seed Mott diagnostic with mixed training proposals and its summary. Both completed with exit code `0:0`. |
| 6666773_1–6666773_6, 6666779 | Six 28-sweep midpoint stability repeats and their combined summary. All completed with exit code `0:0`. The report is `dmrg-l16-l48-midpoint-report-6666773.json`. |
| 6666804_0–6666804_5, 6666814 | ConvTransformer width-20 SF and BKT $L=24$ size pilot and its summary. All completed with exit code `0:0`. The report is `vmc-l24-ct20-gapless-sr-6666804-report.json`. |
| 6666819_0–6666819_1, 6666821 | Matched seed-707 Mott ConvTransformer training budget checks to 768 updates with the local and mixed proposals, and their summary. All completed with exit code `0:0`. |
| 6666824_1–6666824_12 | Final pre-holdout MPS bisection through $L=48$, with caps 20, 28, 28, and 40 for SF/BKT and 20, 28, 28, and 20 for Mott. |
| 6666836, 6666873_1–6666873_6 | Final bisection preliminary summary and six same-cap stability repeats. All completed with exit code `0:0`. |
| 6666879 | Final pre-holdout MPS analysis. Completed with exit code `0:0` and wrote `dmrg-l16-l48-pre-holdout-6666873.json` and its PDF. |
| 6666903, 6666904 | Replacement SR and ConvTransformer GPU validation jobs on A40 in `campus-gpu-bigmem`. Both completed with exit code `0:0`. |
| 6666936 | Six matched A40 profiling arms. Completed with exit code `0:0` and wrote `vmc-l16-ct24-a40-calibration-6666936/report.json`. |
| 6667009_0–6667009_1, 6667012 | Initial final-Mott jobs. Slurm completed them, but the launcher omitted `--mix-nonlocal`, so the validator excluded them from the intended mixed-sampler comparison. |
| 6667248_0–6667248_1, 6667250 | Corrected mixed-proposal Mott diagnostics and summary. All completed with exit code `0:0`. The launcher validates the mixture in the dry run and saved configuration. The report is `vmc-l16-mott-final-triage-6667248-report.json`. |
| 6667064, 6667069, 6667077 | Planned-time A40 training grid, 24 independent CPU frozen evaluations, and their combined summary. All completed with exit code `0:0`. The report is `vmc-l16-bkt-a40-equal-time-6667064-report.json`. |
| 6667134_1–6667134_8 | Prospective $L=64$ lower-cap MPS runs after recording fit predictions, at caps 32/40/44 in SF and BKT and 24/28 in Mott. |
| 6667157 | Evaluation of the first $L=64$ MPS lower-cap results. Completed with exit code `0:0` and wrote `dmrg-l64-holdout-report-6667134.json`. Four gapless cap cases need the prespecified same-cap checks. |
| 6667187_0–6667187_5, 6667195_0–6667195_5 | $L=32$ SF width-20 and $L=24$ BKT width-24 training and frozen evaluations. Both split arrays completed with exit code `0:0` for every task. |
| 6667204 | Joint split summary. Completed with exit code `0:0` and wrote `vmc-l32-sf-l24-bkt-ct-6667187-report.json`. SF had one pass, one miss and one inconclusive seed. All three BKT seeds were inconclusive. |
| 6667209, 6667214, 6667221 | Exploratory width-20 $L=24$ BKT seed-202 diagnostic extended to 512 updates. All completed with exit code `0:0`. The report is `vmc-l24-bkt-ct20-seed202-512-6667209-report.json`. |
| 6667283_1–6667283_4, 6667287 | Same-cap 28-sweep checks at $L=64$ for caps 40 and 44 at each gapless coupling. All completed with exit code `0:0`. The final report is `dmrg-l64-holdout-report-6667283.json`. |
| 6667328 | Final MPS figure through $L=64$. Completed with exit code `0:0` and wrote `dmrg-l16-l64-final-brackets-6667283.pdf`. |
| 6667464, 6667466 | Retained-stride smoke attempts. Both stopped before assertions when the test imported Torch before SciPy and loaded a system C++ runtime without `GLIBCXX_3.4.29`. |
| 6667468 | Retained-stride smoke on Cascade Lake with the frozen evaluator's import order. Completed with exit code `0:0` and wrote `vmc-eval-stride-smoke-6667468.log` and `vmc-eval-stride-smoke-6667468/result.json`. |
| 6667478_0–6667478_7, 6667487 | Independent stride-8 frozen checks for three $L=24$ BKT width-24 checkpoints and the separate width-20 seed-202 checkpoint at update 512. All jobs exited `0:0`. The fresh width-24 stage had one pass, one miss and one inconclusive seed. The width-20 diagnostic passed both fresh evaluations. See `vmc-l24-bkt-stride8-stage-6667478-report.json`. |
| 6667524_0–6667524_1, 6667526 | Independent stride-8 checks of the archived $L=32$ SF width-20 seed-202 checkpoint. Both evaluations and the summary exited `0:0`. Seed 202 passed both, yielding two passes and one miss under the predeclared upgrade rule. See `vmc-l32-sf-seed202-stride8-6667524-report.json`. |
| 6667545_0–6667545_1, 6667547_0–6667547_3, 6667553 | $L=24$ BKT width-20 cohort at 512 updates. Every job exited `0:0`, all three update-256 prefixes matched bitwise and all three seeds passed the stride-8 terminal test. See `vmc-l24-bkt-w20-512-cohort-6667545-report.json`. |
| 6667583_0–6667583_2, 6667584_0–6667584_5, 6667590 | Split $L=48$ SF width-20 runs at 256 updates. Every job exited `0:0`. All three seeds missed energy in both terminal evaluations. See `vmc-l48-sf-w20-6667583-report.json`. |
| 6667597 | One-CPU analysis of saved BKT block-number variance. The finite-size two-point $K_{\rm eff}$ diagnostic completed with exit code `0:0` for all 18 evaluations. See `vmc-bkt-two-point-k-6667545-report.json`. |
| 6667650_0–6667650_2, 6667653_0–6667653_5, 6667655, 6667754 | Split $L=32$ BKT width-20 runs at 512 updates. All fits and evaluations exited `0:0`. The first summary used the wrong second seed filenames. Summary-only job 6667754 corrected the lookup and verified zero passes, one miss and two inconclusive seeds. See `vmc-l32-bkt-w20-512-6667650-verified-report.json`. |
| 6667740, 6667798_0–6667798_1, 6667799 | Exploratory $L=48$ SF width-20 seed-202 replay to 512 updates. Every job exited `0:0` and steps 0 and 256 matched bitwise. Both terminal evaluations missed energy. Parallel jobs replaced unstarted jobs 6667742 and 6667743 for scheduling only. See `vmc-l48-sf-w20-seed202-512-6667740-report.json`. |
| 6667760_0–6667760_3, 6667765 | Final independent stride-16 evaluations for the two inconclusive $L=32$ BKT width-20 checkpoints. All jobs exited `0:0`. Seed 202 passed both fresh tests and seed 101 remained inconclusive at $G_1(16)$. Seed 303 retains its formal miss with equilibrium unresolved. See `vmc-l32-bkt-stride16-6667760-report.json`. |
| 6667769 | Bounded fixed-checkpoint mixing diagnostic for $L=32$ BKT seed 303. It exited `0:0`. Independent local energies agreed, and the mixed-start tails had similar means but a wide difference interval and late-window drift. Equilibrium is unresolved and the original formal miss remains recorded. See `vmc-l32-bkt-seed303-mixing-6667769-report.json`. |
| 6667779 | Paired two-point $K_{\rm eff}$ analysis for $L=32$ BKT seeds 101 and 202 alongside the matched $L=24$ width-20 cohort. It exited `0:0` and all four $L=32$ signed intervals lay above zero. See `vmc-bkt-two-point-k-l32-stride16-6667760-report.json`. |
| 6667828, 6667830_0–6667830_1, 6667832, 6667834 | Single-seed $L=32$ BKT mixed-training diagnostic for seed 303. Every job exited `0:0`, the saved 25% mixture and bitwise step-0 match were verified, and both terminal local evaluations missed energy and $G_1(16)$. The matched-anchor comparison remains unresolved for equilibrium. See `vmc-l32-bkt-seed303-mixed-6667828-report.json`. |

The production bridge uses the archived ED widths, 146 for FCNet and 40
for ConvTransformer, with three matched training seeds per architecture
and 256 Adam updates at learning rate 0.003. Score all arms against ED
every 32 updates and independently evaluate the frozen sampled runs.
Both sampled arms use 64 walkers. They retain 8 or 128 steps per walker,
so the higher-batch control also changes the chain length per update.
The $L=16$ optimizer comparison will retain $N_{\max}=6$, hold sampler
settings fixed between Adam and SR, and record compute cost separately
from the matched-update comparison. The normalized SR solve uses a
centered log-amplitude Jacobian and a fixed damping convention.

Median terminal values at 256 updates across seeds 101, 202, and 303.
The ED energy per site is $-1.06212868t$.

| Architecture | Optimizer data | Median $1-F$ | Median $E/(Lt)$ | Median $\lvert\Delta G_1(5)\rvert$ |
| --- | --- | ---: | ---: | ---: |
| FCNet | 512 retained configurations | 0.16149 | -0.989576 | 0.14977 |
| FCNet | 8,192 retained configurations | 0.05358 | -1.037079 | 0.08776 |
| FCNet | Exact sector | 0.00394 | -1.058943 | 0.02004 |
| ConvTransformer | 512 retained configurations | 0.01060 | -1.054959 | 0.01835 |
| ConvTransformer | 8,192 retained configurations | 0.00898 | -1.058337 | 0.04651 |
| ConvTransformer | Exact sector | 0.00624 | -1.059216 | 0.03752 |

FCNet benefited from the larger sampled batch and from the full-sector
gradient at this budget. For ConvTransformer, the larger batch improved
median energy and fidelity, while the median long-distance correlation
error increased. ConvTransformer seed 101 in the 512-sample arm ended at
$1-F=0.06265$. The three-seed data support keeping both physical acceptance
observables and reporting seed variation and sampling diagnostics.

The $L=16$ calibration compares FCNet at width 49 (8,037 parameters)
and ConvTransformer at width 16 (6,705 parameters). For each architecture,
test Adam at learning rate 0.001 and centered SR at learning rates 0.01
and 0.03 crossed with damping 0.01 and 0.001. Hold the sampler fixed at
256 walkers, eight retained steps per walker, and 32 burn-in proposals
per walker per update. Use two independent frozen evaluations per setting
and rank stable SR settings by the worst normalized DMRG target error.
The CPU calibration ran while the GPU validation waited. CPU and GPU
timings will be compared within each backend.

The calibration selected SR learning rate 0.03 for both models, with
damping 0.001 for FCNet and 0.01 for ConvTransformer. The two frozen
energies per site were $-1.05232t$ and $-1.05102t$ for FCNet and
$-1.05500t$ and $-1.05323t$ for ConvTransformer, against DMRG
$-1.05479370t$. The $G_1(8)$ effective sample sizes ranged from about 86
to 154 for those four evaluations. Their intervals did not establish a
joint crossing. The calibration uses a single training seed and serves
to select fixed optimizer settings for confirmation.

The CPU physical confirmation uses 256 updates on three training seeds
per model and optimizer, with the calibration sampler unchanged. Use
the terminal checkpoint for the primary test. Each checkpoint gets two
independent evaluations with 64 chains, 2,048 burn-in proposals per chain,
and 4,096 retained configurations per chain. Blocks of 128, 256, and 512
preserve within-chain order. A training seed passes only if both frozen
evaluations clear the energy and both correlation thresholds, including
the DMRG reference margins. A configuration needs at least two of three
training seeds to pass.

The report verified bitwise-matched initial weights across all six
Adam/SR pairs. Both Adam cohorts had three misses. FCNet SR had two
misses and one inconclusive result. Its six frozen energy means were
between $-1.05174t$ and $-1.05102t$ per site. All three ConvTransformer
SR results were inconclusive. Its six frozen energy means were between
$-1.05445t$ and $-1.05373t$ per site. For seed 303, one energy interval
extended about $1.25\times10^{-5}t$ per site outside the tolerance after
the reference margin. Neither architecture has a certified $L=16$ BKT
crossing at the baseline width and 256-update budget.

The width comparison keeps the optimizer, sampler, updates, training
seeds, and terminal evaluation protocol fixed. Widths 82 and 24 give
16,155 FCNet and 14,665 ConvTransformer parameters respectively. Log
the extra train compute for the larger widths. Independently, the fresh
evaluation stage holds the three width-16 ConvTransformer checkpoints
fixed and uses two new seeds per checkpoint with 128 chains, 2,048
burn-in proposals, and 8,192 retained steps per chain. The two evaluation
stages produce separate classifications at the same physical thresholds.
The higher-precision stage certified ConvTransformer seed 303. Seeds
101 and 202 remained inconclusive on the energy tolerance, leaving the
width-16 cohort at one of three confirmed passes.

The wider FCNet had two misses and one inconclusive result at the same
256-update budget. All three wider ConvTransformer seeds remained
inconclusive in the original evaluation. Its six frozen energy means
were $-1.05426t$ to $-1.05413t$ per site, within the energy tolerance as
point estimates. Their intervals extended beyond the tolerance. The
wider ConvTransformer checkpoints will receive the same higher-precision
evaluation as the width-16 cohort.

The higher-precision width-24 cohort passed all three training seeds.
Both new frozen evaluations per seed met the energy, $G_1(4)$, and
$G_1(8)$ thresholds. This certifies the tested 14,665-parameter setting
at 256 updates and 2,048 retained configurations per update. It is the
smallest passing tested width so far. The 6,705-parameter case remains
inconclusive, so it does not supply a lower bound on the required
parameter count. The 256-update terminal checkpoint is the first
certified checkpoint, with earlier updates unverified at this precision.

The cross-phase confirmation reuses the certified BKT training setting
unchanged for $U/t=2$ and $8$. It applies the higher-precision frozen
evaluation from the start. Both phase references require $G_1(4)$ and
$G_1(8)$. The Mott reference at $r=8$ is still strong enough under the
planned signal check, so the same distances serve both phases at $L=16$.

At these fixed training settings, ConvTransformer passed all three SF
and BKT training seeds and missed all three Mott training seeds. The
six frozen Mott $G_1(8)$ means ran from 0.0524 to 0.0717, compared with
the DMRG reference 0.02491 and tolerance 0.00175. The miss persists
despite the higher-precision protocol. Two seeds had frozen Mott energy
estimates close to one another across evaluations but outside the energy
target. The third showed large energy uncertainty. A phase-specific
optimizer calibration will vary learning rate and damping at fixed
width and sampler before the larger-ring VMC runs.

All four held-out seed-707 Mott calibration settings missed after
256 updates. The lowest worst normalized interval error occurred at
learning rate 0.03 and damping 0.03. Its two frozen $G_1(8)$ estimates
were 0.0604 and 0.0754. The learning-rate and damping changes tested here
did not close the long-distance error. The follow-up compares exact
and sampled energy training in the $L=10$ Mott sector.

The $L=16$ follow-up keeps seed 707, SR learning rate 0.03 and damping
0.03, 256 updates, and the controlled proposal and evaluation budgets.
One ConvTransformer arm uses uniform distinct-site particle transfers
on 25% of training proposals and nearest-neighbor proposals otherwise.
Invalid transfers stay at the current occupations. The frozen
evaluator uses the original local proposals. One InvariantConv1D arm
tests the local sampler at nearly matched parameter count. Both are
single-seed diagnostics.

The mixed-proposal run missed, with frozen $G_1(8)$ of 0.0578 and 0.0662
versus 0.0604 and 0.0754 in its local ConvTransformer control. The local
InvariantConv1D run also missed, with frozen $G_1(8)$ of 0.0856 and
0.0796. Both proposal choices for ConvTransformer will receive the same
longer training budget in a controlled seed-707 comparison.

The two 768-update Mott trajectories reproduced their own archived
256-update checkpoints bitwise. The local arm gave frozen $G_1(8)$
of 0.0589 and 0.0552. The mixed arm gave 0.0474 and 0.0518, compared
with the DMRG value 0.02491. Both missed the joint target. The final
single-seed tests vary either damping after update 256 at width 24 or
width alone at fixed damping, with the same mixed proposal and budget.

The corrected Mott diagnostics both missed. The width-24 damping-decay
arm reproduced the fixed-damping mixed weights bitwise at update 256.
At update 768 its frozen $G_1(8)$ estimates were 0.0579 and 0.0694.
At width 32, the fixed-damping estimates were 0.0539 and 0.0587.
Width 32 improved the frozen energy estimates to $-0.48978t$ and
$-0.48926t$ per site, but the joint error was larger than for the
width-24 fixed-damping mixed control. The bounded Mott calibration
stops here with no certified crossing at the tested settings and
budgets. Larger-ring Mott VMC remains outside this pilot.

The $L=10$ Mott exact-energy Adam bridge reached median $1-F=0.01937$
at 256 updates, versus 0.02431 for sampled Adam and $1.38\times10^{-4}$
for the archived direct-fidelity fits. The corresponding terminal
median energies per site were $-0.46813t$ and $-0.46379t$ for the two
energy arms, compared with $-0.49236t$ from ED. The residual at this
budget persists with exact gradients in the same model.

The $L=24$ pilot advances the verified SF and BKT setups while Mott
calibration continues at $L=16$. ConvTransformer width 20 has 13,501
parameters at $L=24$, the closest supported width to the 14,665 tested
at $L=16$. The first size comparison holds the training sampler and SR
settings fixed and evaluates the $L=24$ energy and $G_1(6),G_1(12)$
against verified DMRG references.

At `Nmax=6`, the ConvTransformer parameter count at width $w$ is
$P(L,w)=(L+8)w^2+35w+1$. Its flattened readout alone contributes
$Lw^2$ weights, and full attention also increases the forward work
with $L$ at fixed width. These architectural size dependencies enter
the observed parameter and compute budgets.

At 13,501 parameters and 256 updates, ConvTransformer passed all three
$L=24$ SF seeds. At BKT it passed seed 303, with seeds 101 and 202
inconclusive. The latter cohort leaves the required capacity unresolved
at width 20. Advance SF to $L=32$ at width 20 (16,701 parameters) and
test width 24 at $L=24$ BKT (19,273 parameters) under the fixed training
and evaluation protocol.

The first $L=32$ SF cohort gave one pass (seed 101), one energy miss
(seed 303), and one inconclusive result (seed 202). For seed 202,
both $G_1(16)$ means, 0.70577 and 0.70430, are below the upper
tolerance near 0.71412, while the simultaneous upper bounds were
0.72081 and 0.71663. The other required intervals passed. The
precision follow-up remeasures only seed 202, keeping both other
statuses fixed. A pass requires both fresh independent ensembles to
meet the original criterion. The original intervals remain archived.

Both fresh seed-202 evaluations passed for $L=32$ SF. The $G_1(16)$
upper bounds were 0.70146 and 0.70389. The checkpoint passed energy
and $G_1(8)$ again, and the final width-20, 256-update cohort met the
two-of-three rule. Seeds 101 and 202 passed and seed 303 missed.
The $L=48$ SF size test keeps the width and training recipe fixed, so
the flattened readout raises total allocated parameters to 23,101.
Its two independent frozen evaluations per checkpoint increase
burn-in to 6,144 proposals per chain, which maintains the 128 lattice
sweeps used at $L=32$ SF.

At $L=48$ SF, the width-20 cohort missed the joint criterion on all
three seeds at 256 updates. All six energy intervals lay outside
the tolerance, with means between $-1.34183t$ and $-1.33921t$ per
site versus the verified DMRG value $-1.34439470t$. Seed 202 had the
smallest energy discrepancy across the two evaluations. Its
single-seed horizon diagnostic repeats the original settings at 512
updates, audits the step-0 and step-256 tensor prefixes, and scores
the terminal state with two independent frozen evaluations.

The horizon replay matched both archived tensor sets bitwise and
missed at update 512. Its two frozen means, $-1.34047t$ and
$-1.34090t$ per site, were farther from DMRG than the same seed's
update-256 means, $-1.34175t$ and $-1.34183t$. The energy ESS dropped
from about 24,000 to 2,839 and 5,534 under the same frozen settings.
The longer horizon remains a separate single-seed diagnostic. The
largest certified SF cohort size in this pilot is $L=32$.

The $L=24$ BKT width-24 cohort initially yielded three inconclusive
results. All six $G_1(12)$ means, 0.58715–0.59188, lie within the
allowed range around the DMRG reference 0.56605. All six upper
interval bounds extend past the upper tolerance near 0.59485.
Some energy intervals also overlap their threshold. A fresh frozen
stage uses two independent ensembles per checkpoint and classifies
all three width-24 seeds together with the same rule. The new
intervals use only the fresh data.

The width-20 BKT horizon diagnostic reproduced seed 202's archived
256-update weights bitwise. At update 512 both frozen energy and
$G_1(6)$ intervals pass. The first $G_1(12)$ interval passes and the
second overlaps the tolerance. The terminal checkpoint remains
inconclusive under that first frozen stage. Its precision follow-up
is reported independently from the width-24 cohort.

The stride smoke reproduces the previous stride-one output bitwise
and checks stride eight against the exact subsequence of the same
proposal trajectory. Final walkers and RNG state match, and local
estimators are measured only at retained states. Each fresh BKT
evaluation retains 4,096 snapshots on each of 128 chains, one
snapshot per eight proposal steps, after 2,048 burn-in proposals
per chain. The SF follow-up uses 4,096 burn-in proposals. The new
evaluations run four times as many measurement proposals as the first
stage and retain half as many snapshots. Blocks of 32, 64, and 128
retained steps correspond to 256, 512, and 1,024 proposal steps.
Compare the two stages by ESS and runtime after both finish.

The fresh $L=24$ BKT width-24 stage certified seed 101, left seed
202 inconclusive on its energy interval, and found an energy miss
for seed 303. The energy upper interval for one seed-202 evaluation
exceeded the allowed boundary by about $6\times10^{-6}t$ per site.
Both fresh width-20 seed-202 evaluations at update 512 passed the
joint test. Their upper $G_1(12)$ interval bounds were 0.59010 and
0.59008. This resolved the exploratory seed before the parallel
cohort follow-up.

The completed width-20, 512-update cohort passed all three $L=24$
BKT training seeds. For each seed, all 18 initialization and
update-256 tensors were bitwise equal to the original fit. This
certifies the tested 13,501-parameter setting with the longer horizon.
The original width-20 cohort at 256 updates had one pass and two
inconclusive results, and width 24 at 256 updates had one pass, one
miss, and one inconclusive result after the fresh precision stage.
The $L=32$ BKT size test retains width 20 and the 512-update training
recipe, raising allocated capacity to 16,701 parameters.

The first $L=32$ BKT terminal stage returned two inconclusive seeds
and one formal miss under the interval classifier. Seed 101 had
overlapping energy intervals and one overlapping $G_1(16)$
interval. Seed 202 had one overlapping $G_1(16)$ interval, with its
upper bound about $10^{-5}$ above tolerance. Both seeds had point
estimates within the joint tolerances. Their final precision check
uses two fresh local-kernel evaluations per checkpoint, with 128
chains, 8,192 burn-in proposals, and 6,144 retained steps at stride
16. The original evaluations stay archived.

The final stride-16 stage certified seed 202 in both fresh
evaluations. For seed 101, both energy and $G_1(8)$ intervals passed,
but one upper $G_1(16)$ interval edge was 0.55685 against the limit
near 0.55641. The local width-20 cohort therefore has one confirmed
pass, one inconclusive seed, and one formal miss with unresolved
equilibrium. No $L=48$ BKT cohort was promoted at this setting.

The two seed-303 ensembles showed nonstationary energy traces. Their
split-half energy means shifted from $-0.78795t$ to $-0.71434t$ and
from $-0.77426t$ to $-0.70813t$ per site. Chain means included groups
near $-1.048t$ and $-0.47t$, and the energy ESS estimates were 181
and 185. The formal energy-miss output stands as a description of
the interval calculation; the split-chain drift prevents certification
of an equilibrium energy. The separate mixed-proposal assessment
holds the frozen weights fixed and tests sensitivity to the proposal
kernel and the walkers' starting configurations.

The bounded diagnostic recovered valid Fock occupations in both
bands. Independent enumeration agreed with the batched local energy
within $6\times10^{-7}t$ per site on the eight anchors. The sampled
high-energy anchors had onsite energies of $1.23$–$1.43t$ per site,
compared with $0.51$–$0.61t$ for the low-energy anchors. Their frozen
log amplitudes overlapped and sometimes exceeded those of the
lower-energy anchors. In the last quarter of the local balanced-start
runs, the low- and high-start energy means were $-0.919t$ and $-0.552t$
per site. The corresponding means with the mixed kernel were
$-0.518t$ and $-0.528t$. The mixed-kernel interval for the difference
was $[-0.129,0.148]t$ per site, and one group's last quarter shifted
by $0.029t$ per site. The diagnostic confirms the local-energy
estimators on these states and shows access to the high-energy
configurations with both kernels. The equilibrium energy remains
unresolved at this budget.

The separate seed-303 mixed-training control used the same
initialization, verified bitwise across all 18 tensors, and the same
512-update budget. Only the proposal kernel changed. On the
resulting checkpoint, all eight archived anchors had independently
verified local energies. The previously high-energy anchors now had
centered log amplitudes from $-14.53$ to $-10.25$, compared with
$+11.84$ to $+13.11$ for the low-energy anchors. In the short
matched-start test, the last-quarter energy means of both groups
were near $-1.049t$ per site under either frozen kernel. The
short test still leaves equilibrium unresolved under its tail
interval and drift rule. These anchor results and the full
ordinary frozen evaluations belong to the separate mixed-training
diagnostic.

Both ordinary frozen evaluations of the mixed-trained checkpoint
missed energy and $G_1(16)$. The energy means were $-1.04825t$ and
$-1.04828t$ per site against $-1.05128t$ from DMRG. Their energy ESS
estimates were 13,336 and 16,513, and the absolute split-half shifts
were at most $3.46\times10^{-4}t$ per site. The observed drift was
much smaller than with local training from the same initialization.
This one-seed control documents improved sampling stability and
insufficient terminal accuracy at the tested training budget.

The width-24 stride-8 $G_1(12)$ ESS ranged from about 17,300 to
24,000, compared with 4,300 to 5,800 for the original measurements.
The new evaluations took 12m39s–13m42s each on Slurm. The original
two-evaluation tasks took about 55 minutes each. The settings change
the proposal path as well as the measurement cadence, so this
compares the cost and precision of the two evaluation schedules.

The two-point ring estimate $K_{\rm eff}=\pi^2[F(L/2)-F(L/4)]/
\log\sqrt{2}$ compares the same finite-size window for both methods.
The VMC trace $F(\ell)$ averages the second moment about $\ell$ over
all windows of length $\ell$. The DMRG reference averages connected
variances over the same windows. At uniform mean density these
definitions agree. Otherwise, the VMC trace includes the spatial
mean of the squared deviation of the window means from $\ell$.
Estimate its uncertainty using the paired per-snapshot difference
within each chain and the original time blocks, and expand each
signed interval by the DMRG numerical margin. At $L=16,24$, DMRG
gives 2.18466 and 2.16745, respectively. The six $L=24$ width-20
evaluations at update 512 yielded signed VMC differences of
0.140–0.306. Each pointwise 95% interval lay above zero after the
DMRG margin, with the closest lower bound at 0.017. At $L=24$ and
width 24 at update 256 the range was 0.271–0.316. At $L=16$ and
width 24 at update 256 it was 0.147–0.243, with four intervals
above zero and two including zero. These are finite-size block-variance
diagnostics alongside the prespecified energy and $G_1$ gate.

The matched-setting extension to $L=32$ uses the four final
local-kernel evaluations of seeds 101 and 202. The two-point DMRG
estimate is 2.15814, with a numerical margin of $1.40\times10^{-4}$.
The VMC estimates exceed it by 0.281–0.417, with all four pointwise
intervals above zero. The $L=24$ width-20 cohort uses the same
training width and horizon, with a shorter evaluation schedule and
one additional seed. Its bias range overlaps the $L=32$ range.
The comparison documents positive matched-window errors at both
lengths without determining their size dependence.

The near-matched parameter comparison at $L=16$ BKT tested
InvariantConv1D width 66 (14,653 parameters) with the same SR and
evaluation settings as ConvTransformer width 24 (14,665 parameters).
All three InvariantConv1D seeds missed the joint criterion. Its six
frozen energies per site ranged from $-1.05332t$ to $-1.05318t$ and
$G_1(8)$ ranged from 0.658 to 0.668. The comparison controls the exposed
settings and number of parameters, and the two models have different
per-amplitude compute costs and receptive fields.

The FCNet width-82 extension reproduced seed 101's archived weights
bitwise at update 256. The trailing 32-update sampled energy means at
updates 256, 512, and 768 were $-1.05148t$, $-1.05254t$, and $-1.05317t$
per site. At update 768, the two frozen evaluations gave $-1.05308t$ and
$-1.05310t$ per site. Both remain outside the frozen energy tolerance
under the joint test. This single-seed control documents improvement
with training budget and a miss at the tested endpoint.

The DMRG cap-256 and cap-384 reference pairs are ready for all three
phases at $L=24,48$. The largest reference energy margin is
$1.15\times10^{-6}t$ per site. At $L=48$ in the superfluid, the cap-384
run converged and the cap-256 check passed comparisons between 28 and
56 sweeps. The reference pair differs by $8.53\times10^{-7}t$ per site
and at most $3.34\times10^{-5}$ across the $G_1$ profile. The ten
lower-cap stability checks resolved the brackets at both added sizes.
Two bisections tightened the brackets before the new $L=64$ checks.

| $L$ | SF bracket | BKT bracket | Mott bracket |
| ---: | ---: | ---: | ---: |
| 16 | $20<\chi_\star\le24$ | $20<\chi_\star\le24$ | $20<\chi_\star\le24$ |
| 24 | $24<\chi_\star\le28$ | $28<\chi_\star\le32$ | $24<\chi_\star\le28$ |
| 32 | $28<\chi_\star\le32$ | $28<\chi_\star\le32$ | $24<\chi_\star\le28$ |
| 48 | $32<\chi_\star\le40$ | $32<\chi_\star\le40$ | $20<\chi_\star\le24$ |
| 64 | $32<\chi_\star\le40$ | $40<\chi_\star\le44$ | $16<\chi_\star\le24$ |

Both gapless cases show an increase across the measured lengths.
Their brackets differ at $L=24$ within the four fit sizes. Mott stays
bounded by 28 over the four fit sizes and varies
nonmonotonically on this grid. At cap 24 the Mott energy error stays
well inside tolerance, and $G_1$ sets the threshold; at $L=48$ the
binding $G_1(8)$ error is about 99.3% of its tolerance. A pure power law
through these four sizes is feasible for SF with slope 0.262–0.631
and BKT with slope 0.380–0.464. The intervals overlap, and omitting
$L=16$ brings the lower limit of the BKT slope to zero. The intervals
are feasibility bounds for this grid and physical target. A pure
monotonic power law is incompatible with the four Mott brackets.

The fit recorded $L=64$ predictions of 34.50–47.96 for SF and
40.65–44.14 for BKT before the new lower-cap submissions. Saturation
immediately after $L=48$ would keep either gapless threshold in
$(32,40]$. The earlier exploratory report had already revealed passing
upper caps at $L=64$, so the new lower-cap test is a partially observed
extension check.

The $L=64$ same-cap checks resolved the gapless brackets. At cap 40,
the binding energy error was 0.950 times tolerance for SF and 1.116
times tolerance for BKT, including the numerical checks. BKT passed
at cap 44 with normalized energy error 0.881. Mott passed at cap 24
with a binding $G_1(8)$ error 0.874 times tolerance. Both gapless
brackets overlap the recorded power-law predictions. Only SF overlaps
the immediate-plateau prediction from $L=48$. The finite-size split
is compatible with shared asymptotic exponents given the overlapping
fit-size slope intervals.

The V100S validation jobs reached one hour pending on priority near
21:09 and 21:15 Eastern on November 25. Dry runs with shorter requests
on eligible public and condo GPU partitions gave later estimates. The
earliest alternate estimate at the second check was 01:10 Eastern on
`ai-tenn-debug`, compared with 23:54 and 00:20 for the submitted jobs.
The jobs retained their `campus-gpu-large` positions while CPU work
continued in parallel.
At the third queue check an A40 slot opened earlier. The worker held
the two accepted replacements, cancelled the pending V100S copies,
then released both A40 jobs. Slurm placed them on `ilpa1209`.

The A40 checks passed. The attention Jacobian differed from individual
sample gradients by at most $2.98\times10^{-7}$. On a single A40, all
six profiling arms completed from matched initial weights at $L=16$
BKT and width 24. The medians below use updates 5 through 32.

| Retained per update | Adam s/update | SR s/update | SR peak GPU memory (GiB) |
| ---: | ---: | ---: | ---: |
| 512 | 0.185 | 0.331 | 0.197 |
| 2,048 | 0.432 | 1.214 | 0.596 |
| 8,192 | 1.667 | 8.061 | 2.459 |

At 8,192 samples, SR spent about 4.35 seconds per steady update on its
dense solve. These profiles measure hardware cost per update. The short
training horizon provides timing calibration for a fixed-time sample
comparison.

The planned-time grid targets about 310 steady A40 update-seconds
for each fixed update cap. At batches 512, 2,048, and 8,192, the SR
caps are 936, 256, and 38 updates. The Adam control uses 718 updates
at batch 2,048. Run three training seeds per arm and apply the same
independent frozen physics test to each terminal checkpoint. Log
actual update time, optimizer operations, and evaluation time to
show how closely the observed costs match the nominal budget.

| Optimizer and batch | Updates | Retained samples per seed | Actual A40 update seconds | Pass / miss / inconclusive |
| --- | ---: | ---: | ---: | ---: |
| SR, 512 | 936 | 479,232 | 315–328 | 2 / 1 / 0 |
| SR, 2,048 | 256 | 524,288 | 316–318 | 2 / 0 / 1 |
| SR, 8,192 | 38 | 311,296 | 308–311 | 0 / 0 / 3 |
| Adam, 2,048 | 718 | 1,470,464 | 322–332 | 0 / 3 / 0 |

The three SR arms used about $7.25\times10^{12}$,
$3.22\times10^{13}$, and $8.18\times10^{13}$ estimated solve FLOPs
per seed, respectively. The report records sampler-forward estimates
and Jacobian work alongside the solves. The 2,048-sample SR setting
remains the default because its CPU cohort passed all three seeds and
its A40 cohort passed two without a certified miss. The 512-sample
arm is a second verified setting at similar A40 runtime. The
8,192-sample arm remains inconclusive at its shorter update cap.

## Repository closeout

Validation job 6669553 checked the Python entry points, DMRG manifests,
and Slurm script syntax, then passed the CPU SR, attention Jacobian,
mixed proposal, and retained-stride smoke checks. Job 6669575 passed all
four existing 1D VMC regression tests on CPU. Evidence export job
6669573 recorded the final bracket and cohort data with source report
names, SHA-256 hashes, and originating job IDs. The [numerical
summary](results/pilot-summary.json) and [MPS figure](results/mps-brackets.pdf)
are the repository copies.
