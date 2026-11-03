# Two-dimensional Bose–Hubbard NQS plan

## Requested backup extension

The bounded runs are complete. Both hybrid supervised fits pass their stated
ED gates; fresh2×3 periodic SR+nativeSGD reaches exact absolute overlap
0.999997867. Both frozen seeds and the full37-sector3×3 raw scan are retained.
The6×4 table now has measured basic-Jastrow and RBM128 results from one
conservative stability retry each; the failed initial attempts are preserved.
Jastrow is a baseline (6.33% energy gap at seed654); RBM misses the neural gate
(24.04%). This closes the requested measurement budget, without a new accuracy
or architecture search. See STATUS.md and the dated backup result folder.

Fill the6×4-cylinder comparison with the original basic Jastrow baseline and
RBM using bounded SR+native-SGD runs and frozen321/654 evaluations against the
same cylinder DMRG reference. The user explicitly requests these rows, which
supersedes the earlier decision to withhold RBM6 until RBM4 passed its gate.
Report misses honestly; Jastrow remains a potentially poor physical baseline.

Add Conv2DTransformerNet as a faithful2D extension of the1D ConvTransformer:
3×3 boundary-aware convolution, SiLU, one four-head Transformer block,
flattened dense-SiLU scalar head. Use zero dropout for a deterministic VMC
wavefunction. Keep the existing five architectures/checkpoints unchanged.
Qualify with Adam+log-MSE on periodic ED-accessible clusters before VMC, then
benchmark fresh small-system SR+SGD and raw neighboring-sector weights. These
hybrid results go in backup slides; main periodic benchmarks remain intact.
No larger-lattice or unconstrained architecture search is requested.

## Current research focus — user clarification

Main-body geometry is now explicitly periodic in BOTH x and y. Use matched
2×3 and3×3 tori for ED, supervised, VMC and raw-sector comparisons. Cylinder
results remain labeled supporting material, not substitutes for torus data.
Small periodic-Hamiltonian DMRG is allowed: an open-chain MPS is a representation
choice, not an open physical boundary. Validate chi128/256 against3×3 torus ED.
Main slides must show actual site-density and NN one-body correlation benchmarks
with ED references, exact learned-state expectations and frozen-MC errors.
Uniform unit-filling density is a useful symmetry check but does not certify
the correlations; include onsite fluctuation and NN coherence comparisons.

Before the requested backup extension, the approved periodic revision had43pages,
28main and15supporting. The extension preserves all28 main pages and adds six
supporting pages, for49total. Main15–17 show measured density/NN/onsite benchmarks;
DMRG128/256 on the3×3 torus agrees with ED to1.93e-7/4.59e-10 in total energy.
The main supervised and leakage datasets now use existing periodic width32
primary fits, and the VMC comparisons use the existing2×3 periodic states.
All current torus records are under2026-11-01-periodic-focus. The earlier
cylinder-focused counts and reports below are retained chronological history.

The main presentation and next work establish four points: independent ED
construction checks; supervised representation of the ED ground state with the
basic ansatzes; actual stochastic VMC validation on ED-accessible clusters; and
the raw network's amplitude outside its trained fixed-particle sector,
especially N0−2 through N0+2. This supersedes the earlier campaign to bring all
architectures to a common larger-cylinder accuracy. Further architecture and
optimizer development is a separate possible direction, not the main milestone.

The earlier cylinder-focused presentation was assembled and independently approved:
25 main pages plus14supporting pages. The completed evidence includes15fresh
ED-specific tests, four-neural supervised overlap checks, two fresh500-update
small-system SR+SGD runs with exact final-state and frozen-sampler checks, and
exact raw-sector scans of all5^9configurations for each of fiveprimarymodels.
The current PDF and reproducible sources are linked from STATUS.md. No further
architecture, large-size or optimizer campaign is active.

Retain all 1D source files and the four original slide PDFs unchanged. Compare the
1D sector-distribution definitions, but distinguish full truncated-Hilbert-space
normalization from normalization over a five-sector window. Report sector
dimensions and mean weight per configuration as a degeneracy control. Fixed-N
supervised fits do not determine relative sector weights: adding a(N−N0) to
the log amplitude preserves the canonical state and changes off-sector mass.
Exact particle projection sets off-sector amplitudes to zero without retraining
and preserves every number-conserving canonical observable. A number penalty
evaluated only on fixed-N examples is zero and cannot train unseen sectors.

Criticality motivates cautious finite-size claims; it is not a theorem that VMC
fails. On each finite cluster report the neutral fixed-N gap when available and
separate energy accuracy from absolute overlap. For a unique ground state,
E−E0 >= Delta_N*(1−F_abs^2). This bound weakens as the gap closes; an accurate
energy density does not certify a full many-body wavefunction. Width-fixed
cylinders are not the isotropic two-dimensional thermodynamic limit.

The worker implements frozen-checkpoint leakage and a bounded actual SR+SGD
small-cluster test if needed; the checker audits normalization, ED references,
sampling and exact final-state metrics. Parent rewrites the slide narrative;
ridge mathematics, one small MinSR comparison and larger-cylinder results stay
as supporting appendices. No larger-lattice or optimizer grid is launched.

## Approved refinement and presentation phase

The requested detailed ridge-SR derivation is now included in the progress PDF
(eight appendix pages, 29–36). Optional real-valued MinSR with the published
power-six soft spectral filter has been added and independently validated;
ridge remains the unchanged default. A bounded, matched 300-update RBM64 4×4
comparison uses the same starting checkpoint, sampling protocol, learning-rate
schedule, seed and precision. Both frozen evaluation seeds are complete: neither
arm meets the energy target and no resolved optimizer advantage is established.
The requested small-system, small-sample illustration is on page 13; density
and nearest-neighbor definitions and lattice diagrams are on pages 14–15,
with Monte Carlo estimator details on page 37.

The bounded width128 trials are now complete and independently audited. The
confirmation energy gaps are 2.22% for FC128d3 at 6×4 and 1.51% for RBM128 at
4×4; neither meets the unchanged 1% target. Later widened FC hidden replicas
remain almost identical on saved probes, so these results do not establish a
hard expressiveness limit. Before a new campaign, resolve the research-scope
choice: optimization of original families, explicit lattice-symmetry variants,
or closure of this tool-stage comparison with remaining gaps documented.

Latest confirmation: the user accepts the centered, normalized SR changes and
the exact regularized sample-space solve when M<P. Retain full-Jacobian dense
direct solves, float64 SR algebra and native SGD. The sample-space ridge is
M*tau; this changes matrix dimension, not the samples used in an update. No
return to the legacy 1D convention is requested. Basic Jastrow is deliberately
a potentially poor physical baseline; further adaptation is out of scope.

User clarification after the progress snapshot: Jastrow is a non-universal
physical baseline, with periodic-boundary focus. Preserve the homogeneous
onsite-plus-displacement-pair form; do not pursue boundary-aware or higher-body
adaptations to force the neural-network accuracy target. Retain the optional
reflection-orbit experiment and its results as historical diagnostics, outside
the primary baseline selection. No higher-body terms were implemented. The
common accuracy refinement priority applies to the four neural architectures.
Compare Jastrow with neural states only under matching boundaries; cylinder
DMRG benchmarks and periodic-baseline comparisons must remain distinct.

The SR audit also identified differences from the original 1D CPU code:
1D uses uncentered, unnormalized J.T@J with a sum-scaled ridge and a CPU-double
direct solve; 2D uses centered, sample-normalized covariance, a covariance-scaled
ridge and a direct solve on the model device, optionally in sample space.
Jacobian vectorization is separate from those mathematical changes. Numerical
same-batch equivalence of the two 2D dense formulations is not evidence of
identical long-run convergence between 1D and 2D conventions. Do not attribute
remaining neural-fit gaps to architecture capacity without isolating optimizer
and regularization effects.

The user selected the next order: tighten Conv2D VMC first, then bring all five
architectures to comparable VMC accuracy, and produce a new 2D slide PDF matching
the style and structure of the four existing 1D slide decks. Keep all new code,
results and slide sources under this directory and all 1D files unchanged.

Working proof-stage targets are relative energy difference from matched DMRG
at most 1%, central direction-averaged NN correlation differences at most 5%,
and central density difference at most 0.01. Require conditional SEM at most
0.25% of the reference energy magnitude, 2.5% of each NN correlation magnitude,
and 0.005 for density. Two-SEM proximity is a diagnostic, never a substitute for
meeting both the point and precision targets. These are provisional practical targets, not fidelity or criticality
criteria. Preserve the existing 4x4/6x4 references and hold lattice size fixed.
Select candidates using frozen measurements; confirm chosen fits with a separate
evaluation seed and inspect block-length stability. Record failed runs and
capacity/budget differences rather than ranking unequal-budget architectures.

First enable explicit inference-checkpoint warm starts, optional recorded
learning-rate/SR-shift schedules, and independently verified bounded Jacobian
batching if it improves throughput. Warm starts use fresh walkers/optimizer and
are not exact training resumes. Keep SR with native SGD and supervised fitting
with Adam. Avoid architecture changes until optimization/capacity evidence
justifies them. After Conv reaches the working benchmark, adapt the other model
budgets and capacities as needed, preserving small-ED qualification.

Measured capacity follow-up: FC32 and RBM32 remain above the 4×4 energy target
after a larger-sample continuation, so bounded width-64 branches are authorized
with explicit weight lineage and independent small-ED qualification. Jastrow's
4×4 energy gap remains about 4.1%; an exact-sector 3×3 energy-SR diagnostic
also reaches a local plateau at a 1.66% gap with a small exact gradient. This is
not a global expressivity bound. One optional cylinder Jastrow sharing mode is
authorized: onsite x-reflection orbits and pair x-coordinate reflection orbits,
retaining periodic-y translations/reflections and correctly reversing occupation
roles under x reflection. Preserve the default displacement-shared version and
its checkpoints. Qualify any new configuration against small ED before accepting
larger-system VMC results; keep primary log-MSE and optional infidelity separate.

The final presentation will use the 1D decks' 16:9 white layout, large black
sans-serif titles, mathematical definitions, model descriptions, comparison
tables, VMC/SR results, takeaways and to-do list. Include matched boundaries,
U/2 convention, unsquared overlaps, uncertainty definitions and finite-size/
cutoff limits. Save reproducible slide source and result provenance beside the
new PDF; do not overwrite the four original PDFs.

Status: the bounded cylinder prototype now includes 4x4 and 6x4 SR+SGD VMC,
matched fixed-chi DMRG, and frozen density/one-body measurements. Hold size at
6x4 for this tool stage; see [STATUS.md](STATUS.md) for current results and the
active refinement campaign. Cylinder qualification requires Transformer
width 64 at the tested budget; the Jastrow primary log-MSE fit misses 0.99 but
an optional infidelity refinement demonstrates overlap capacity above 0.99.
All earlier torus results and criteria remain archived separately.

Historical torus qualification: ED-supervised fitting is the first research gate. FCNet,
Conv2DNet and TransformerNet pass the approved thresholds on 2x3 at width 32;
RBMNet also passes there at width 32 after longer training in both tested seeds.
The user now accepts absolute overlap at least 0.99 for larger ED systems;
all five completed 3x3 candidates meet that fidelity criterion. Energy and
variance remain separate diagnostics. Earlier energy-minimization runs are
exploratory diagnostics and do not measure supervised expressiveness.
The manager has adopted the proposed defaults below under the user's instruction
to continue autonomously; these remain adjustable research settings.

## Scope and isolation

Develop neural-network quantum states for a rectangular two-dimensional square
lattice with nearest-neighbor hopping and onsite repulsion. Put all new source,
tests, documentation, and generated outputs under `pythonTest/2D/`. Do not modify
any existing 1D files, checkpoints, or project configuration. Keep the 2D code
self-contained so that its interaction convention cannot silently inherit the
different convention used by the 1D implementation.

## Agreed physics

\[
H=-t\sum_{\langle i,j\rangle}(b_i^\dagger b_j+b_j^\dagger b_i)
  +\frac{U}{2}\sum_i n_i(n_i-1)-\mu\sum_i n_i.
\]

- Uniform, real nearest-neighbor hopping in both lattice directions; no diagonal
  hopping, external potential, or longer-range interactions.
- `U` is the standard interaction parameter multiplying `n(n-1)/2`. It is twice
  the coefficient named `U` in the existing 1D implementation.
- Support independent `Lx` and `Ly`; if `Ly` is omitted, use `Ly=Lx`.
- Support periodic and open boundaries, with periodic as the default.
- Fix total particle number. Default `Ntot=Lx*Ly` (unit filling) and `mu=0`.
  Within this sector, changing `mu` shifts energy by `-mu*Ntot` without changing
  the eigenvectors.
- `Nmax` is an inclusive occupation cutoff: `0 <= n_i <= Nmax`.
- Focus on the user-selected coupling `U=16.74`. Use `t=1` as the provisional
  default, making the target ratio `U/t=16.74`. This is a study point; finite-size
  calculations alone will not establish the thermodynamic critical coupling.

Use a documented flattening such as `i=y*Lx+x` everywhere. Each unordered
nearest-neighbor bond contributes its hopping term once. Before implementing,
  use the periodic length-two convention: a unique-edge graph counts one bond
between the two sites, whereas a periodic directional sum can count two parallel
bonds. Tests and reference comparisons must use the same convention. Never add
self-hopping for an axis of length one.

## Five architectures

All models return a real, unnormalized log amplitude for occupation
configurations. Ground states for the intended unfrustrated hopping can be
represented with nonnegative amplitudes in the occupation basis. Normalize only
within the specified particle-number sector for comparisons.

| Model | Initial design |
| --- | --- |
| FCNet | Adapt the existing dense network to flattened 2D site features. |
| JastrowNet | Adapt the existing Jastrow ansatz to 2D coordinates and site pairs; document its parameterization and symmetry properties. |
| Conv2DNet | Replace 1D convolutions with 2D convolutions; use circular padding for periodic boundaries and suitable open-boundary padding. |
| RBMNet | Adapt the existing RBM to the 2D configuration encoding. |
| TransformerNet | A minimal encoder with occupation embeddings, 2D positional information, one attention block, pooling, and a scalar log-amplitude head. Start without optional complexity. |

Do not assume every architecture is translation invariant. Record parameter
counts and implemented symmetries. Number-conserving sampling does not imply
that the unconstrained model has zero amplitude outside the selected sector.

## Coordination

- Project manager: primarily review research quality through theoretical and
  fundamental physics questions, including Hamiltonian conventions, finite-size
  effects, cutoff and optimization convergence, reference validity, sampling
  uncertainty, and the physical conclusions supported by the evidence. Maintain
  direction and coordinate implementation and review without taking over routine
  code-quality review.
- Worker (`nqs2d-worker`): create and modify code, implementing one bounded
  milestone at a time in a separate Codex
  pane, with an explicit list of allowed paths and acceptance criteria.
- Code checker (`nqs2d-checker`): independently review correctness,
  runtime/memory behavior, simplicity, and consistency with the existing core
  coding style, main functions, and interfaces in a separate Codex pane; return
  actionable findings and reproduce important checks. Coordinate any simplifying
  edits with the worker before touching the same files.
- Keep generator and validator edits coordinated. The validator should report
  findings before editing files owned by the generator. All agents must preserve
  existing 1D files.

## Milestones and acceptance criteria

### Next phase: larger VMC with matched-cylinder DMRG benchmarks

The user confirmed the next direction on October 31, 2026:

- Scale beyond ED using VMC with SR and native SGD.
- Let FCNet grow in both hidden width and hidden-layer count; retain the current
  two-hidden-layer design as a reproducible baseline.
- Benchmark rectangular cylinders with periodic short direction and open long
  direction. Keep fully periodic tori as a separate study.
- First establish reliable energies, density profiles, and one-body correlations;
  critical exponents and universality fits are a later milestone.
- All new code, configurations and outputs stay under `pythonTest/2D/`; all
  pre-existing 1D source remains unchanged.

The user clarified that this is a proof-of-principle/tool-building stage, not a
publication or extreme-accuracy campaign. The working sequence is now
`(Lx,Ly)=(4,4)`, `(6,4)`, and only if needed `(8,4)`, with open x and periodic y.
Use a fixed modest DMRG bond dimension (initial proposal chi=128), with one
chi=256 spot-check on the smallest cylinder. Stop size growth once energy per
site and matched bulk correlations change little relative to statistical and
reference uncertainty. Do not compare raw total energies across different sizes.
The previous 12x4/width-six proposal is deferred. Increase width as well as length
only if a later goal requires bulk 2D conclusions. Match
NQS and DMRG geometry, bonds, `U/2` convention, `t=1`, `U=16.74`, `mu=0`, unit
filling and occupation cutoff exactly. A narrow cylinder at the bulk study
coupling is a finite-width benchmark, not automatically a critical 2D system.

Implementation order:

1. Add independent axis boundary flags throughout lattice, ED, models, VMC,
   checkpoints and CLIs, preserving legacy both-open/both-periodic behavior.
   Conv padding and Jastrow distances must follow the actual axis boundaries.
2. Add configurable FC hidden depth, with default depth two and compatible
   loading of existing checkpoints. Test capacity growth at documented parameter,
   memory and wall-time budgets instead of assuming a common width is equally
   expressive across architectures.
3. Replace the dense-SR scaling bottleneck with a guarded equivalent sample-space
   solve when the sample count is smaller than the parameter count. For centered
   Jacobian `Jc`, centered energies `Ec`, sample count `M` and shift `tau`, verify
   `delta=Jc.T @ solve(Jc@Jc.T + M*tau*I, Ec)` against the existing parameter-space
   solve. Preserve SR normalization and SGD update semantics. Jacobian storage and
   computation still need explicit guards; a smaller Gram matrix alone does not
   solve every scaling problem.
4. Implement a separate 2D DMRG reference using the existing Julia ITensor stack
   when available. Traverse the short direction in the MPS snake; preserve a
   physical-coordinate mapping for all output arrays. Validate tiny cylinders
   against ED before benchmarking larger grids. Do not edit the 1D DMRG example.
5. Add frozen-model evaluation of density and one-body correlations with
   number conservation, correct bosonic factors, batch bounds, and independent
   small-ED checks. Use central/bulk reference sites on open cylinders; report
   edge profiles separately. The full one-body matrix obeys trace=Ntot and is
   Hermitian; its diagonal equals the density.
6. Check the modest DMRG reference with a fixed bond dimension, sweep stability,
   truncation diagnostics and one higher-bond-dimension spot check; a broad
   convergence campaign is outside this proof-of-principle stage. Establish
   VMC convergence with independent seeds/chains, autocorrelation/blocking,
   effective sample size, sampling length, SR shift and learning-rate sensitivity.
   Retain walker/time axes for statistical diagnostics; do not treat flattened
   correlated samples as independent observations.

Physics review criteria:

- The untwisted hopping Hamiltonian has nonpositive off-diagonal matrix elements;
  real positive amplitudes are appropriate in a connected fixed-number sector.
- At fixed particle number, `<b_i>=0` even for a superfluid-like finite state.
  Use `<b_i^dag b_j>` and, where measured, eigenvalues of the one-body matrix.
- The bulk commensurate transition is in the 3D XY class with z=1. A diverging
  correlation length and closing gap make long-distance correlations important;
  a local Hamiltonian does not imply a short-range-correlated critical state.
- Current Conv2DNet has one local 3x3 convolution followed by a global dense
  head. Circular padding alone does not enforce translation invariance. Treat
  its expected advantage as a hypothesis. Preserve the baseline before testing
  deeper/residual/dilated feature extraction or long-range Jastrow/CNN hybrids.
- Enforce only symmetries of the chosen geometry: a cylinder does not have torus
  translations along the open axis or square-lattice 90-degree symmetry.
  Emergent particle-hole symmetry near the bulk tip is not an exact local
  occupation inversion constraint on the full bosonic Hilbert space.
- Check local occupation distributions and cutoff stability on new sizes; the
  existing small-system cutoff checks do not establish all larger-size errors.
- Without a reliably evaluated NQS–reference overlap, energy agreement does not
  certify fidelity>=0.99. For a finite nondegenerate problem,
  `1-F_abs^2 <= (E_NQS-E0)/gap`; the closing gap weakens this energy-based bound.
  DMRG energy is itself approximate until its reference convergence is established.

Relevant sources:

- [Capogrosso-Sansone et al., 2D QMC benchmark](https://arxiv.org/abs/0710.2703)
- [Rancon and Dupuis, commensurate criticality](https://arxiv.org/abs/1106.5585)
- [Denis and Carleo, neural backflow Jastrow](https://arxiv.org/abs/2404.07869)
- [ITensorMPS convergence and cylinder guidance](https://docs.itensor.org/ITensorMPS/dev/faq/DMRG.html)

Current status: mixed-axis boundaries, FC depth and format-2 checkpoints have
been implemented, with general CLI defaults still periodic on both axes. All
five historical format-1 checkpoints reproduce independently saved raw-forward
gold exactly in float64. Boundary/depth milestone A passed 66 CPU tests. SR
milestone B now passes 70 CPU tests, including both solve spaces, independent
NumPy directions, allocation guards and actual SGD updates; the manager also
reran the four focused SR tests independently. Auto SR chooses the smaller
matrix while guarding Jacobian storage. The standalone DMRG reference and
frozen-model observable evaluation now pass their independent checks; the final
core test count is 83, with eight additional benchmark-helper tests. Completed
4x4/6x4 results are summarized in STATUS.md.
Independent small-matrix
checks verified the dual-SR identity in 15 float64 cases and snake-permutation
equivalence in 16 small one-particle cases. One-body row-action estimators were
checked in 30 entries, including complex test states to expose index reversal.
Those earlier design checks complement the subsequently completed implementation
tests; they do not replace the matched numerical benchmarks.

The manager generated an independent full one-body ED reference for the 3x3
open-x/periodic-y cylinder: 19,855 states, 15 bonds, energy
`-4.385696750796594`, residual `5.75e-14`. Direct ladder-operator sums for the
one-body matrix, density and onsite factorial moment reconstruct the energy
within `1.5e-14`. See
`benchmarks/results/3x3-cylinder-independent-ed-observables.json`.

Bounded 4x4 SR+SGD pilots use all five already ED-qualified architectures,
width 32 where applicable, 150 updates, 256 correlated samples/update,
`lr=0.02`, `tau=0.01`, seed 123, CPU float64. All five complete, but several
trajectories are noisy or poorly optimized; they establish execution rather
than convergence or an architecture ranking. Short final training-side
measurements are not reliable comparisons to DMRG. Better-sampled Jastrow and
Conv pilots (1024 samples, 300 updates, `lr=0.01`, `tau=0.05`) are bounded
optimization diagnostics. Hold size at 4x4 until independent frozen-model
energy and correlation estimates, with statistical diagnostics, make the
comparison useful. Keep all reports under
`benchmarks/results/2026-10-31-cylinder-pilots/` and checkpoints under the
matching ignored `outputs/` directory.

The initial frozen comparison is now recorded in
`benchmarks/results/2026-10-31-cylinder-pilots/frozen-dmrg-comparison.json`.
At 4x4, DMRG gives `E=-7.671673531665546` at chi 128 and
`-7.671699717843744` at chi 256; the energy/site sensitivity is `1.64e-6`,
and the largest bulk one-body change is `1.67e-5`. The independently checked
3x3 DMRG energy differs from ED by `4.27e-10`; full rho differs by `3.08e-9`.
Frozen-observable milestone C passes 77 CPU tests and an independent full-sector
3x3 ED check, including pointwise energy reconstruction. These establish
implementation correctness and a useful modest reference, not VMC convergence.

The initial 4x4 frozen pilots have substantial optimization errors. A sampling
diagnostic finds 89--97% identical adjacent retained configurations and pooled
chain-centered energy lag-one correlations around 0.89--0.94. These are
descriptive diagnostics, not ESS estimates. Add an explicit compatible
`--thin` control for proposal steps between retained training configurations
(default one); retain the distinction between proposals, retained samples and
lattice sweeps. Test exact default seeded behavior and strided-trajectory
equivalence. Compare a bounded thinned run at the same size before attributing
the VMC shortfall to network capacity. No network or size expansion is justified
by the initial noisy trajectories alone. This diagnostic led to the implemented
stride control and completed refined Conv runs: energy/site differences from
matched DMRG are `0.01646*t` at 4x4 and `0.02405*t` at 6x4. Central
direction-averaged nearest-neighbor correlations differ by roughly 2--4%, with
conditional errors formed from per-sample composite observables before blocking.
Hold size at 6x4 for the tool stage. Fixed-chi DMRG central energy changes 1.6%
and central NN correlations 3--4% between these sizes; this is not a claim of
thermodynamic convergence. See `benchmarks/results/dmrg-nqs-4x4-6x4-central-strip.json`.

The matched 3x3-cylinder supervised extension is now complete. FC32, Conv32 and
RBM32 pass the working 0.99 overlap gate using Adam/log-MSE. Transformer32 misses
it, while Transformer64 reaches independently checked overlap `0.990323` with
the same one-block/one-head design. Jastrow's primary log-MSE fit plateaus at
`0.988193`; optional 2000-step Adam/infidelity refinement of the same ansatz
reaches `0.998775`. The positive centered log-target span residual does not bound
achievable overlap, and this candidate demonstrates sufficient overlap capacity.
Primary log-MSE defaults and failed primary-gate records remain unchanged.
Larger-system VMC fidelity is not evaluated or inferred.

The default coordinate convention is fixed: x is long/open, y is short/periodic;
physical index `i=y*Lx+x`. The short-direction MPS snake is
`s=x*Ly+(y if x is even else Ly-1-y)`, with Julia indexing `s+1`.
The 4x4, 6x4 and 8x4 cylinders have 28, 44 and 60 unique bonds respectively.
Tests must cover both mixed orientations even though output defaults use x-long.

Julia 1.13.1 successfully imports the existing ITensors/ITensorMPS environment;
the root Manifest records 1.10.4 and remains unchanged. Boson MPO and primed-bra
contraction API probes also pass. Bounded runtime pilots support the six-sweep
chi-128 references and one two-sweep chi-256 spot check at 4x4. A smaller SR Gram
matrix does not remove Jacobian memory:
the proposed FC capacity grid can already require roughly 629 MiB for a single
1024-sample float64 Jacobian at the larger settings.

### Research order agreed with the user

1. Validate the ED Hamiltonian and numerical reference.
2. Fit the ED ground-state wavefunction by supervised learning with Adam.
3. Increase trainable widths and check seeds/training progress for unsuccessful
   fits; investigate architecture changes when a persistent limitation remains.
4. Use VMC to scale beyond ED only after the ansatz passes supervised checks.
   SR-based VMC uses native `torch.optim.SGD`; combining SR with Adam is rejected.

The user approved the initial supervised acceptance gate: both absolute,
unsquared overlap at least `0.9999` and energy error per site at most
`1e-3*t`. Monitor normalized log-amplitude MSE, exact energy variance, actual
gradient norms, and training trends as additional diagnostics. Use full-sector
normalization consistent with the original 1D supervised loss. A finite-budget
Adam failure is not by itself proof of an expressiveness limit: distinguish
optimization difficulty from a restriction of the represented function class.
JastrowNet has no width parameter, so simply increasing hidden_dim cannot enlarge
that ansatz.

Updated criterion: for the current 3x3-and-larger ED stage, the user defines
fidelity convergence by **absolute overlap >=0.99**, not squared overlap. Energy
errors and variance remain visible, separately from the earlier stricter joint
gate. Historical raw reports retain their original gate definitions. The working
size scope starts at nine sites for the current experiments; it is not a physical
boundary. Keep the stricter small-system checks for the initial 2x3 comparison.

Adam + normalized log-amplitude MSE remains the primary supervised method.
The user authorized Adam + infidelity as a separate diagnostic. Compare loss
choices using the same architecture, seed, budget, and raw ED target. Infidelity
must not silently use floored log-MSE targets. No architecture change is implied.

The authorized diagnostic is complete. Optional `--loss infidelity` is available;
`--loss log_mse` remains the default. All 58 CPU tests passed, including loss
gradients, raw-target independence, selected-checkpoint semantics, and unchanged
default Adam updates. The checker independently reconstructed four selected
Jastrow checkpoints. On 3x3, log-MSE gives absolute overlap `0.992349382` and
energy error/site `0.028694622`; infidelity gives `0.998520270` and `0.010327029`.
Both pass the current 0.99 fidelity criterion. Infidelity improves overlap,
energy and variance in these examples while worsening log-MSE; the primary
method is unchanged. Detailed 2x3/3x3 comparisons, settings and provenance are
retained in `benchmarks/results/jastrow-loss-comparison.json` and README.md.

ED references have been checked using analytical atomic/one-particle/two-site
results, independent tensor-product ladder operators projected into fixed-N
sectors, small dense eigensolves, and residual/cutoff checks. Subsequent 3x3
cylinder DMRG comparisons are recorded above. A small residual validates a solve of the assembled matrix;
it does not replace independent Hamiltonian-construction checks.

Current implementation review status (not supervised qualification):

- Milestone 1 implemented and reviewed: atomic-limit eigensolves, scalar domain
  checks, counting-work preflight, and iterative basis enumeration corrected.
  Fifteen exact tests plus 82 independent matrix/eigenvalue cases and 150 basis
  count/order cases passed.
- Milestone 2 implemented and reviewed: five architectures and inference
  checkpoints; twelve model tests passed. Independent float64 CPU/CUDA outputs
  and parameter gradients agreed for all five architectures on open and periodic
  3x2 rectangles. Checkpoint loading validates physical conventions and sector.
- Milestone 3 implemented and reviewed: all 43 CPU tests passed after output
  staging, finite float64 statistics, scalar-domain, and synchronization fixes.
  Independent sampler/update checks found no physics blockers. CUDA checks passed for
  all five models: ED local energies, native updates, sampling invariants, and
  tiny centered SR solves against CPU.
- Milestone 4 initial baselines complete: 30 short runs across two geometries,
  five models and three seeds; five 500-update 3x3 follow-ups; five 4x4 execution
  smokes without ED; and a paired exact-gradient/sampled-gradient diagnostic.
  These establish functionality and expose remaining optimization error, not
  ground-state convergence or critical behavior.
- Milestone 5 documentation and reproducible commands are in README.md, with
  raw text records under benchmarks/results/ and ignored checkpoints under
  outputs/. No pre-existing tracked files changed. Further scientific work needs
  quantitative size/accuracy/observable targets.

Final independent review found no substantive blockers in the paired diagnostic
or completed baseline records. Sparse Rayleigh gradients passed an independent
directional finite-difference check; new run statuses, source hashes and scoped
memory peaks were checked against source and CLI sidecars. The paired exact and
sampled updates retained similar residual energy errors with nonzero gradient
norms. Sampling alone is not an explanation for that one case; neither an
ansatz capacity limit nor convergence has been established.

The supervised targets are now specified above. Use 2x3 and the ED-feasible
3x3 as initial fit tests. Larger-size VMC studies remain deferred until the
supervised gate is satisfied for the chosen ansatz. Do not infer a critical
point from the completed small-system validation runs.

### First supervised observations

All five models completed 10,000 Adam updates on the 2x3 reference at width 32
where applicable, seed 123, float64, `lr=1e-3`. Best-by-log-MSE checkpoints give:

| Model | Absolute overlap | Energy error per site | Joint gate |
| --- | ---: | ---: | --- |
| FCNet | 0.999998424 | 0.000012045 | Pass |
| JastrowNet | 0.997297954 | 0.016957214 | Fail |
| Conv2DNet | 0.999997787 | 0.000016875 | Pass |
| RBMNet | 0.998617758 | 0.008604880 | Fail, still improving |
| TransformerNet | 0.999995272 | 0.000042999 | Pass |

The checker independently reconstructed saved-weight metrics. Jastrow's
stationary loss is `0.023511369` with gradient norm approximately `9e-10`;
its centered-design lower bound is `0.02298899`. This is evidence of a restricted
log-amplitude class under this objective, not a proof that a different
supervised loss cannot satisfy the physical gate. The user has now authorized
an Adam infidelity-loss diagnostic before any architecture change.
RBM's late loss was still falling, so longer training and initialization checks
were run: at 20,000 updates, width-32 RBM fits pass in seeds 123 and 456
(overlaps `0.99994728` and `0.99997198`, energy errors/site `0.00042118` and
`0.00020365`). These runs establish qualifying examples for this ansatz on 2x3.
Wider and longer FCNet checks improve but do not pass the historical strict
3x3 joint gate. They do satisfy the newly accepted 0.99 overlap criterion.
No new VMC campaign is part of the current infidelity diagnostic.

The 3x3 cutoff reference has also been checked at the wavefunction level:
embedding the `Nmax=4` ED state into the full nine-particle basis gives absolute
overlap `0.99999999015` with the full ED ground state. The full state has
probability `0.99999998037` within the cutoff-four subspace. This error is well
below the supervised overlap tolerance. Measurements are retained in
`benchmarks/results/ed-cutoff-wavefunction-check.json`; no DMRG is involved.

One float64 3x3 Transformer fit was deliberately interrupted for runtime, with
its logs and manager note retained. This is not a failed ansatz result. A
separate float32 fit with independent float64 saved-weight evaluation replaces
it and has completed; the independent check confirms a missed gate. The original
campaign's nonzero trainer exit status must be interpreted with
that explicit interruption note.

The completed 3x3 trials do not yet meet the joint supervised gate: width-32
FC/Conv/RBM fits at 10,000 updates, width-128 FC fits at 10,000 and 20,000,
width-128 Conv with float32 training at 10,000, and the replacement width-32
Transformer fit. These statements refer to the historical strict joint gate.
Under the subsequently approved absolute-overlap threshold of 0.99, all five
completed 3x3 candidates are fidelity-converged. The wider/longer fits improve,
but gradients and loss trends do not justify an impossibility claim. An optional
Adam infidelity-loss cross-check is now authorized; log-MSE remains primary.

### 1. Lattice and exact reference

Implement geometry, bond construction, fixed-number basis enumeration, and a
sparse ED Hamiltonian using `U/2`. Guard ED against oversized basis allocations.

Acceptance: explicit tiny-lattice bond checks; conservation and Hermiticity;
analytic atomic-limit checks; independent hopping matrix-element checks; and
small ED ground states. Verify the 1D limit against existing conventions with
the required factor-of-two conversion of `U`, without changing 1D code.

### 2. Models and inference

Implement the five architectures with a consistent interface and checkpoint
metadata including lattice dimensions, boundaries, cutoff, architecture settings,
and Hamiltonian convention.

Acceptance: valid output shapes, finite log amplitudes and gradients, meaningful
parameter counts, and checkpoint load/save agreement. Keep initial model designs
small and explicit.

### 3. Supervised ansatz qualification

Implement `train_ed.py` following the existing 1D supervised core-function style,
with Adam and a normalized full-sector log-amplitude MSE. Keep all five candidate
models available. Report any regularization of tiny ED amplitudes explicitly;
evaluate physical overlap, energy error and variance against the original ED
state rather than a modified target. Keep checkpoints and training histories
separate from VMC artifacts.

First validate the loss and gradients independently, then train on 2x3 and the
ED-feasible 3x3 reference. Start with manageable widths and multiple seeds;
increase width/training budget where needed. Record the parameter count, gradient
norm and loss trend before diagnosing a capacity limitation. Qualify models using
both user-approved thresholds above. Do not use VMC optimization failure to
reject an ansatz before these supervised checks.

### 4. CPU/GPU VMC scaling after supervised qualification

Implement batched fixed-number sampling, 2D local energies, and training. Start
with the already implemented energy-gradient path as historical diagnostics.
For the research scaling stage, use SR with native SGD and explicit parameter
and memory safeguards after supervised qualification. Chunk connected-state
evaluation to manage memory. Keep ED optional.

Acceptance: local energies agree with the independently constructed ED operator
for arbitrary small-system model states; sampler proposals preserve number and
cutoff and satisfy the implemented Metropolis–Hastings rule; gradients and updates
agree with small reference calculations. Check CPU/GPU agreement within stated
floating-point tolerances. Save inference metadata and training history; describe
whether checkpoints support exact training resumption.

### 5. Accuracy and efficiency baselines

Train all five architectures on common small-system settings and compare exact
sector energies and overlaps where ED is feasible. Then run a larger 2D case
without ED. Provisional starting sizes are 2x2 and 2x3 for exact checks, and 4x4
for the first larger VMC baseline; these are not commitments to ED feasibility
for arbitrary cutoffs.

Report wall time, peak GPU memory where available, parameter count, energy per
site, and local-energy variance. Separate startup cost from steady-state timing
when discussing throughput. Use multiple training seeds for accuracy claims;
keep initial smoke runs clearly labeled. Assess sampling autocorrelation and
effective sample size before attaching uncertainty estimates to energies.
Do not rank architectures from one short run or report local-energy standard
deviation as an error bar on the mean.

### 6. Documentation and review

Document commands, defaults, physical conventions, artifact locations, known
limitations, and reproducible benchmark settings. The validator reviews the
completed implementation and the manager resolves findings before declaring a
milestone complete. Verify that no tracked pre-existing files changed.

## Proposed organization

Names can be adjusted during implementation while retaining isolation:

```text
pythonTest/2D/
  PLAN.md
  README.md
  lattice.py
  exact_diagonalization.py
  models.py
  vmc.py
  train.py
  tests/
  benchmarks/
  outputs/
```

## Working decisions and research review

1. Adopt `t=1` alongside `U=16.74`.
2. Start with `Nmax=4`, with
   cutoff-convergence checks at higher cutoffs before physical accuracy claims.
3. Use unique unordered bonds, explicitly documented for tiny systems. A 2x2
   periodic graph therefore has coordination two, not the coordination four of
   the extended square lattice. Use it to test code, not to interpret criticality.
4. Prioritize ground-state energies and convergence first, then
   extend measurements to correlations, condensate observables, or criticality
   studies. No phase-diagram or critical-point determination is included in the
   first implementation milestone.
5. The initial implementation used a common boundary flag. The October 31
   extension adds independent axis flags for the approved cylinder benchmarks,
   while retaining fully periodic defaults for existing commands.
6. Review finite-size, occupation-cutoff, variational, and sampling errors
   separately. Energy accuracy alone does not establish correlation accuracy.

The selected coupling is consistent with the unit-filling square-lattice QMC
benchmark reporting `(t/U)c = 0.05974(3)` (approximately `U/t=16.74`):
[Capogrosso-Sansone et al., Physical Review A 77, 015602 (2008)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.77.015602).
This comparison concerns the bulk model. It does not certify a finite lattice,
an occupation-truncated Hamiltonian, or a finite-budget variational calculation
as critical. Fixed-number `mu=0` is appropriate for the chosen sector but must
not be interpreted as a grand-canonical chemical potential at the Mott-lobe tip.

### Initial manager checks (October 30, 2026)

Using the new sparse ED implementation at `t=1`, `U=16.74`, `mu=0`, unit
filling and periodic unique bonds:

| Geometry | Nmax | Sector dimension | Ground-state energy |
| --- | ---: | ---: | ---: |
| 2x3 | 2 | 141 | -2.529402278252 |
| 2x3 | 3 | 336 | -2.580945445047 |
| 2x3 | 4 | 426 | -2.581089511910 |
| 2x3 | 5 | 456 | -2.581089553287 |
| 2x3 | 6 | 462 | -2.581089553289 |
| 3x3 | 3 | 13,051 | -6.102729544364 |
| 3x3 | 4 | 19,855 | -6.104592578442 |
| 3x3 | 5 | 22,825 | -6.104595824675 |
| 3x3 | 9 | 24,310 | -6.104595826144 |

The 3x3 eigenvector residual norm was approximately `5.2e-14`. The 2x3 cutoff
check supports starting at `Nmax=4` for that geometry only. Unlike 2x3, periodic
3x3 has coordination four and is a useful next variational reference. At 4x4,
the `Nmax=4` sector contains 183,424,355 states: do not attempt the ED path under
the default resource guards. These are implementation checks, not independent
external ground-state benchmarks or evidence of a phase transition.

For 3x3, the default dimension guard was explicitly raised to 25,000 for the
`Nmax=5` and `9` checks; the memory guard remained in force. `Nmax=9` spans the
entire nine-particle sector, so the cutoff-four total-energy error is about
`3.25e-6` for this finite system. This is much smaller than the expected initial
optimization error and gives a controlled reference for the first baselines.

Periodic 3x3 also contains length-three winding cycles and is not bipartite.
Its coordination-four graph is useful for exact validation, but it is not a
substitute for the even-size sequence (starting at 4x4) in square-lattice
finite-size studies. Separate these geometry effects from variational errors.
