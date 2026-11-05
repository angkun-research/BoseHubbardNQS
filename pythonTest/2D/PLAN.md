# 2D Bose–Hubbard NQS Research Plan

This is the single current research plan, combining the prior project plan
with the SVD/RG representation-reduction study. The active research direction
compares **supervised normalized log-MSE fitting and VMC energy minimization**
using the same ordinary CNN. This document specifies research and implementation
milestones; consolidating it does not launch training or Julia integration.

## Project context and completed foundation

The previous bounded study was the **3×3, N=9 square-symmetry and supervised
GCNN study**: identify microscopic symmetry operators, verify the ED state,
construct a regular-representation group CNN, fit with Adam + normalized
log-MSE, and explain the four stages in a dedicated teaching artifact.
Its restriction against VMC and larger-size experiments applied to that study.
The new research direction below explicitly includes both log-MSE and VMC;
larger lattices and physical-regime sweeps remain later extensions.

The microscopic symmetry group and ground-state sector are documented in
[docs/SQUARE_SYMMETRY.md](docs/SQUARE_SYMMETRY.md). For the square fully periodic
benchmark, the target has fixed charge N0=L², zero momentum, and A1 of D4.
Independent 3×3 ED checks found that all 72 geometric operations preserve the
ground state to 2.2e-15. The 19,855 configurations form 338 spatial orbits;
the zero-momentum/A1 sector has dimension 338 although A1 is a 1D irrep.
Symmetry enforces equality within each orbit, leaving 337 independent real
state parameters after normalization and positivity/global-phase conventions.
The 338×338 reduced Hamiltonian reproduces the full ED vector and energy.

The declared GCNN fit and independent numerical review are complete:
absolute overlap **0.9996088830**, energy error/site **0.001840578116t**.
The prior plan records a dedicated 58-page teaching deck,
`BoseNQS2D-symmetry-progress-2026-11-02.pdf` (the PDF is not present in
this checkout; its provenance JSON remains). It retains the finite-fit errors and comparison limitations. Detailed completed
scope and known misses remain in [STATUS.md](STATUS.md). These results are
prior evidence, not results of the new SVD experiments.

The new baseline is the ordinary `Conv2DNet`, whose symmetry can emerge during
training; the previously studied `GroupEquivariantConv2DNet` imposes symmetry
and is a possible later control. Preserve accepted results, presentation,
checkpoint semantics, current Python defaults, and 1D functionality. Reuse
one shared PyTorch VMC core rather than introducing a duplicate Julia or NumPy
implementation. Julia integration remains a separate, unmerged workstream.

The previous broad refinement/large-system campaign remains closed.
Historical plans under `docs/history/pre-documentation-cleanup/` provide
traceability and do not override the active research direction below.

## 1. Project goal

Study whether both supervised log-MSE fitting and variational Monte Carlo
(VMC) optimization of a deliberately simple
convolutional neural quantum state (NQS) develop a low-dimensional,
symmetry-structured hidden representation, and whether this
representation can be periodically truncated using an
SVD/eigendecomposition procedure without degrading the quality of the
ground-state approximation.

The initial model will be intentionally simple:

\[ `\text{configuration}`{=tex} `\rightarrow`{=tex}
`\text{single CNN layer}`{=tex} `\rightarrow`{=tex}
`\text{nonlinearity}`{=tex} `\rightarrow`{=tex} `\text{flatten}`{=tex}
`\rightarrow`{=tex} `\text{FC}`{=tex} `\rightarrow`{=tex}
`\log`{=tex}`\psi`{=tex}\_`\theta`{=tex}(`\mathbf `{=tex}n). \]

A global-pooling readout will be studied as a control.

The physical testbed is the 2D Bose--Hubbard ground state in regimes
where the Hamiltonian symmetries are known. This provides both a
controlled physics benchmark and a setting in which emergent symmetry,
representation compression, and variational accuracy can be studied
simultaneously.

The central conceptual picture is

\[ `\boxed{
\text{ordinary log-MSE or VMC optimization}
\rightarrow
\text{emergent effective hidden space}
\rightarrow
\text{occasional SVD/RG truncation}
\rightarrow
\text{continued optimization with the same objective}.
}`{=tex} \]

The project is not initially intended to propose a new large-scale
compression algorithm. Its first purpose is to establish, in a
controlled NQS problem, whether the hidden feature space exhibits a
reproducible low-dimensional flow during optimization and whether that
flow can be exploited by an RG/DMRG-like truncation.


### 1.1 Two training objectives and a controlled comparison

Use the same existing `Conv2DNet` for two primary training branches:

1. **Supervised log-MSE:** fit ED ground-state log amplitudes using the
   existing normalized, uniform full-sector log-MSE in `train_ed.py`.
   Define normalized log amplitudes by
   $\ell_\theta(\mathbf n)=\log\psi_\theta(\mathbf n)
   -\tfrac12\log\sum_{\mathbf m}\exp(2\log\psi_\theta(\mathbf m))$.
   Minimize $\mathcal L_{\log\mathrm{MSE}}=|\Omega_N|^{-1}
   \sum_{\mathbf n\in\Omega_N}(\ell_\theta(\mathbf n)-\ell_0(\mathbf n))^2$,
   where $\Omega_N$ is the fixed-particle-number sector and $\ell_0$ is
   the normalized ED target. Record the existing target flooring convention;
   evaluate physical fidelity against the raw ED state.
2. **VMC:** minimize variational energy using the existing sampling and
   optimization procedure. ED amplitudes are used only for evaluation,
   not for training this branch.

The primary question is whether each objective naturally develops a
lower effective activation dimension, and whether subsequent SVD
truncation reduces actual trainable parameters while preserving accuracy.
Passive monitoring alone does not reduce the parameter count. Report
activation effective dimension, retained rank, and actual trainable
parameter count separately, including projection overhead and unchanged
convolution cost.

Use the same lattice, Hamiltonian, occupation cutoff, architecture,
width, and paired initial weights across branches; repeat across seeds.
Compare against optimizer step, wall-clock time, and attained physical
accuracy. An epoch and a VMC iteration are not equivalent work units.
This compares the complete training procedures, including optimizers;
it does not isolate the objective alone without further optimizer controls.

Covariance depends on the configuration distribution. For the primary
small-system comparison, evaluate centered activation covariance over
the full sector using the same fixed ED probability weights
$p_0(\mathbf n)=|\psi_0(\mathbf n)|^2$ for both branches and all checkpoints,
and average uniformly over sites. These weights are evaluation-only
and must not enter VMC updates. Also report uniform-sector covariance
(aligned with log-MSE) and current-model probability covariance (aligned
with VMC sampling) as explicitly labeled secondary diagnostics. This
separates feature changes from changes in sampling distribution. For
larger systems without ED, declare a fixed reference ensemble and check
sampling convergence and autocorrelation.

In both branches, perform passive monitoring first, one-shot truncation
second, and periodic truncation only after those checks. Resume each
branch's original objective after truncation. Task-aware statistics
must specify the objective: normalized log-MSE gradients for supervised
fitting and a valid VMC energy-gradient estimator for VMC. An ordinary
supervised derivative of local energy is not a substitute for that estimator.

------------------------------------------------------------------------

## 2. Motivation

### 2.1 From symmetry to effective representation

A generic CNN already has enough expressive freedom to represent
functions with many additional symmetries, but those symmetries are not
explicitly enforced. A group-equivariant CNN instead restricts the
hypothesis space by imposing known weight-sharing relations.

For a square periodic Bose--Hubbard lattice, relevant exact symmetries
can include lattice translations and the square-lattice point group
(D_4), subject to the Hamiltonian parameters and chosen quantum-number
sector.

This suggests two complementary questions:

1.  **Emergent symmetry:** does an ordinary CNN NQS spontaneously learn
    a representation compatible with the known physical symmetries?
2.  **Emergent dimension:** does the apparently large hidden channel
    space actually collapse onto a much smaller task-relevant subspace
    during log-MSE fitting and VMC optimization?

The second question motivates an RG-like treatment of the hidden
representation.

### 2.2 DMRG/RG analogy

DMRG starts from a large state space, identifies the states most
important for representing the target state, truncates the rest, and
continues optimization in the reduced space.

The proposed neural analogue is

\[ H `\rightarrow`{=tex} r\_{`\rm eff`{=tex}},
`\qquad `{=tex}r\_{`\rm eff`{=tex}}\<H, \]

where (H) is the number of CNN hidden channels and
(r\_{`\rm eff`{=tex}}) is inferred from the learned activation
statistics.

The analogy should not be overstated: channel-space truncation alone is
not a spatial RG transformation. In the first project it should be
described more precisely as an **RG/DMRG-inspired adaptive reduction of
representation space**. A future extension could combine this with
explicit spatial coarse-graining.

------------------------------------------------------------------------

## 3. Minimal NQS architecture

Consider a square (L`\times `{=tex}L) Bose--Hubbard lattice. Let the
input configuration be

\[ `\mathbf `{=tex}n={n\_{`\mathbf `{=tex}r}}, \]

with an appropriate local occupation encoding.

Use a single convolutional layer with (H) hidden channels:

\[ h_c(`\mathbf `{=tex}r) =
`\sum`{=tex}*{c*{`\rm in`{=tex}},`\delta`{=tex}}
K\_{c,c\_{`\rm in`{=tex}}}(`\delta`{=tex})
x\_{c\_{`\rm in`{=tex}}}(`\mathbf `{=tex}r+`\delta`{=tex}) +b_c, \]

followed by a pointwise nonlinearity,

\[ z_c(`\mathbf `{=tex}r)=`\sigma[h_c(\mathbf r)]`{=tex}. \]

The main readout is

\[ z `\rightarrow`{=tex} z\_{`\rm flat`{=tex}} `\rightarrow`{=tex}
`\text{FC}`{=tex} `\rightarrow`{=tex}
`\log`{=tex}`\psi`{=tex}\_`\theta`{=tex}(`\mathbf `{=tex}n). \]

For a real positive ground state where appropriate, a real scalar output
may suffice. More general regimes should retain whatever amplitude/phase
parameterization is already used in the Bose--Hubbard NQS codebase.

### Primary architecture

\[ `\boxed{
\text{Input}
\rightarrow
3\times3\text{ Conv}(C_{\rm in}\rightarrow H)
\rightarrow
\text{SiLU}
\rightarrow
\text{Flatten}
\rightarrow
\text{FC}
\rightarrow
\log\psi
}`{=tex} \]

Here `FC` denotes the existing `Conv2DNet` dense head:
`Linear(H * Lx * Ly, H) → SiLU → Linear(H, 1)`. The baseline uses
one-hot occupations and boundary-aware padding (circular on periodic axes).
The monitored representation is the output of the first SiLU, before
flattening. Projection changes the input dimension of the first dense layer;
the remaining dense head stays in place. A single linear scalar readout
would be a separate architecture ablation.

Start deliberately overparameterized, e.g. (H=32,64,128), depending on
computational cost.

### Control architecture: global pooling

\[ `\boxed{
\text{Input}
\rightarrow
\text{Conv}
\rightarrow
\text{SiLU}
\rightarrow
\text{global pooling}
\rightarrow
\text{FC}
\rightarrow
\log\psi
}`{=tex} \]

This gives a simpler reduced-space readout and provides a useful
comparison with the flatten+FC case.

------------------------------------------------------------------------

## 4. Why analyze activations rather than only convolution weights?

A weight matrix can have many formally available directions that are
rarely or never used by configurations encountered during log-MSE fitting
or VMC sampling.

The relevant hidden representation is therefore better characterized by

\[ z_c(`\mathbf `{=tex}r;`\mathbf `{=tex}n) \]

than by the bare convolution kernels alone.

Collect post-activation hidden features over configurations and lattice
sites, using the explicitly declared weights from Section 1.1. Define the centered channel covariance

\[ C\_{cc'} = `\mathbb `{=tex}E\_{`\mathbf `{=tex}n,`\mathbf `{=tex}r}
`\left[
(z_c-\mu_c)(z_{c'}-\mu_{c'})
\right]`{=tex}. \]

Equivalently, collect activations into a matrix (Z) and perform

\[ Z=U`\Sigma `{=tex}V\^`\dagger`{=tex}. \]

Then

\[ C`\propto `{=tex}U`\Sigma`{=tex}^2U^`\dagger`{=tex}. \]

The eigenvectors/singular vectors define the empirically occupied hidden
directions, while the eigenvalues/singular values quantify their
importance in activation space.

------------------------------------------------------------------------

## 5. Effective hidden dimension

Let

\[
`\lambda`{=tex}\_1`\ge`{=tex}`\lambda`{=tex}\_2`\ge`{=tex}`\cdots`{=tex}`\ge`{=tex}`\lambda`{=tex}\_H
\]

be the eigenvalues of the activation covariance.

### Cumulative retained weight

Define

\[ R(r) = `\frac{\sum_{i=1}^{r}\lambda_i}`{=tex}
{`\sum`{=tex}\_{i=1}\^{H}`\lambda`{=tex}\_i}. \]

For a tolerance (`\epsilon`{=tex}), choose the smallest (r) satisfying

\[ R(r)`\ge1`{=tex}-`\epsilon`{=tex}. \]

The discarded activation weight is

\[ `\epsilon`{=tex}\_{`\rm disc`{=tex}} =
`\frac{\sum_{i>r}\lambda_i}`{=tex} {`\sum`{=tex}\_i`\lambda`{=tex}\_i}.
\]

### Participation-ratio effective dimension

Also monitor

\[ D\_{`\rm eff`{=tex}} = `\frac{\left(\sum_i\lambda_i\right)^2}`{=tex}
{`\sum`{=tex}\_i`\lambda`{=tex}\_i\^2}. \]

This provides a threshold-independent measure of effective channel
dimension.

The first important experiment should measure

\[ D\_{`\rm eff`{=tex}}(t) \]

during both ordinary log-MSE fitting and ordinary VMC training **without performing any
truncation**.

A reproducible decrease such as

\[ 64`\rightarrow45`{=tex}`\rightarrow31`{=tex}`\rightarrow25`{=tex} \]

would indicate that optimization itself is producing a lower-dimensional
effective representation.

------------------------------------------------------------------------

## 6. Periodic RG/SVD truncation

Apply this procedure separately to log-MSE and VMC training.
Do **not** perform a decomposition/truncation after every optimizer update.

The working hypothesis is that the network needs an interval of ordinary
optimization in which the representation can reorganize and approach a
relatively stable subspace.

Use a schedule

\[
`\underbrace{\text{optimization}\rightarrow\cdots\rightarrow\text{optimization}}`{=tex}*{T*{`\rm settle`{=tex}}}
`\rightarrow`{=tex} `\text{SVD/truncate}`{=tex} `\rightarrow`{=tex}
`\underbrace{\text{optimization}\rightarrow\cdots}`{=tex}*{T*{`\rm settle`{=tex}}}
`\rightarrow`{=tex} `\text{SVD/truncate}`{=tex}
`\rightarrow`{=tex}`\cdots`{=tex} \]

At each truncation event:

1.  Accumulate activation statistics under the declared configuration
    distribution for the log-MSE or VMC branch.
2.  Compute (C) or an equivalent activation SVD.
3.  Determine (r) from a chosen tolerance or effective-rank criterion.
4.  Construct the retained basis \[ U_r=(u_1,`\ldots`{=tex},u_r). \]
5.  Project the hidden representation into this basis.
6.  Transform the downstream readout consistently.
7.  Resume optimization using the original branch's objective.

Track the flow

\[
H_0`\rightarrow `{=tex}H_1`\rightarrow `{=tex}H_2`\rightarrow`{=tex}`\cdots`{=tex}.
\]

A possible convergence criterion is

\[ H\_{k+1}=H_k \]

for several consecutive RG events together with stable energy and
discarded weight.

------------------------------------------------------------------------

## 7. How the reduced space connects to the FC layer

This is central to the project.

The original hidden tensor is

\[
z`\in`{=tex}`\mathbb `{=tex}R\^{H`\times `{=tex}L_x`\times `{=tex}L_y}.
\]

The retained channel basis gives

\[ `\tilde `{=tex}z_a(`\mathbf `{=tex}r) = `\sum`{=tex}*{c=1}\^{H}
(U_r)*{ca}z_c(`\mathbf `{=tex}r), \]

so

\[ H`\times `{=tex}L_x`\times `{=tex}L_y `\rightarrow`{=tex}
r`\times `{=tex}L_x`\times `{=tex}L_y. \]

The spatial lattice is unchanged. Only the hidden channel space is
reduced.

### Flatten + FC

Originally,

\[ z\_{`\rm flat`{=tex}}`\in`{=tex}`\mathbb `{=tex}R\^{HL_xL_y}. \]

After reduction,

\[
`\tilde `{=tex}z\_{`\rm flat`{=tex}}`\in`{=tex}`\mathbb `{=tex}R\^{rL_xL_y}.
\]

If

\[ y=A z\_{`\rm flat`{=tex}}+b, \]

then approximately

\[ z\_{`\rm flat`{=tex}} =
(I\_{L_xL_y}`\otimes `{=tex}U_r)`\tilde `{=tex}z\_{`\rm flat`{=tex}}. \]

Therefore

\[ y `\approx`{=tex}
A(I`\otimes `{=tex}U_r)`\tilde `{=tex}z\_{`\rm flat`{=tex}}+b. \]

Define the projected FC matrix

\[ `\boxed{
\tilde A=A(I\otimes U_r).
}`{=tex} \]

The reduced network then uses

\[ y=`\tilde `{=tex}A`\tilde `{=tex}z\_{`\rm flat`{=tex}}+b. \]

Thus the FC layer does not need to be discarded or retrained from
scratch when the hidden basis is truncated.

### Global pooling

For

\[ q_c = `\frac{1}{L_xL_y}`{=tex}
`\sum`{=tex}\_{`\mathbf `{=tex}r}z_c(`\mathbf `{=tex}r), \]

the channel projection and global pooling commute:

\[ `\tilde `{=tex}q=U_r\^`\dagger `{=tex}q. \]

This makes the global-pooling architecture a particularly clean control.

------------------------------------------------------------------------

## 8. Important nonlinearity caveat

The SVD is performed on the post-activation representation

\[ z=`\sigma`{=tex}(h). \]

In general,

\[ U_r\^`\dagger`{=tex}`\sigma`{=tex}(h) `\neq`{=tex}
`\sigma`{=tex}(U_r\^`\dagger `{=tex}h). \]

Therefore the post-activation projection cannot in general be naively
absorbed into a smaller set of pre-activation convolution kernels.

For the first implementation, explicitly represent the projection as a
linear channel map after the nonlinearity:

\[ `\text{Conv}`{=tex}(C\_{`\rm in`{=tex}}`\rightarrow `{=tex}H)
`\rightarrow`{=tex} `\sigma`{=tex} `\rightarrow`{=tex}
P(H`\rightarrow `{=tex}r). \]

This projection is equivalent to a shared (1`\times1`{=tex})
convolution.

This means the first version may reduce the FC/readout cost without
immediately reducing the cost of producing all (H) convolution channels.
Folding the reduced representation back into a genuinely smaller
convolution should be treated as a later extension.

------------------------------------------------------------------------

## 9. Activation magnitude is not sufficient: downstream relevance

A hidden direction can have substantial activation variance but be
ignored by the FC layer.

For example,

\[ `\lambda`{=tex}\_i\>0, `\qquad`{=tex} A u_i`\approx0`{=tex}. \]

Activation-only SVD would regard this direction as important even though
it has essentially no effect on the wavefunction output.

Conversely, a low-variance direction can be strongly amplified by the
downstream readout.

Therefore distinguish:

### A. Weight-space importance

Analyze convolution weights only.

### B. Activation-space importance

Use

\[ C_z=`\mathbb `{=tex}E\[zz\^`\dagger`{=tex}\]. \]

### C. Task-aware importance

Include downstream sensitivity.

A simple sensitivity object is

\[ g_z = `\frac{\partial\mathcal L}{\partial z}`{=tex}, \]

with

\[ C_g = `\mathbb `{=tex}E\[g_zg_z\^`\dagger`{=tex}\]. \]

Possible task-aware reductions can use activation and gradient
information together. More sophisticated versions may use Fisher or
curvature information.

For the first project, it may be more informative to **compare** several
simple criteria than to commit immediately to one new task-aware density
matrix.

The operational test is:

> At a fixed retained dimension (r), which criterion produces the
> smallest degradation in normalized log-MSE (where available),
> variational energy, and wavefunction quality
> immediately after truncation and after re-optimization?

------------------------------------------------------------------------

## 10. Symmetry diagnostics

The Bose--Hubbard system gives a controlled setting because the
Hamiltonian symmetries are known.

For every known transformation (g), evaluate a wavefunction-level
diagnostic such as

\[ `\Delta`{=tex}*g\^`\psi`{=tex} =
`\mathbb `{=tex}E*{`\mathbf `{=tex}n`\sim`{=tex}\|`\psi`{=tex}\|\^2}
`\left`{=tex}\|
`\log`{=tex}`\psi`{=tex}(g`\mathbf `{=tex}n)-`\log`{=tex}`\psi`{=tex}(`\mathbf `{=tex}n)
`\right`{=tex}\|\^2. \]

Where practical, also measure

\[ C_g = `\frac{\langle\psi|U_g|\psi\rangle}`{=tex}
{`\langle`{=tex}`\psi`{=tex}\|`\psi`{=tex}`\rangle`{=tex}}. \]

For a state in the trivial invariant sector,

\[ C_g=1. \]

Candidate symmetries include:

-   lattice translations under periodic boundary conditions;
-   (90\^`\circ`{=tex}), (180\^`\circ`{=tex}), and (270\^`\circ`{=tex})
    rotations;
-   reflections;
-   the full square-lattice (D_4) structure where applicable.

The key questions are:

1.  Does ordinary CNN optimization spontaneously approach the known
    symmetry?
2.  Does the dominant activation subspace become more
    symmetry-structured during training?
3.  Does SVD truncation preserve, improve, or damage wavefunction
    symmetry?
4.  Are symmetry-breaking hidden directions preferentially discarded?
5.  Is the effective dimension correlated with physical regime or
    distance from a phase transition?

Do not infer physical symmetry solely from individual hidden kernels or
hidden irrep amplitudes. The final wavefunction symmetry must be checked
directly.

------------------------------------------------------------------------

## 11. Experimental phases

### Phase 0 --- Establish physical/NQS benchmark

Choose small or intermediate 2D Bose--Hubbard systems where high-quality
reference data are available.

Record:

-   lattice size;
-   boundary conditions;
-   (U/t);
-   filling/particle-number sector;
-   exact symmetries;
-   reference ground-state energy;
-   other reference observables available from exact diagonalization,
    DMRG, QMC, or trusted existing calculations.

Verify both the existing ED log-MSE fitting and NQS/VMC implementations
before introducing SVD.

### Phase 1 --- Ordinary training, no pruning

Train the same single-CNN + FC model in two independent branches:

1. supervised normalized log-MSE fitting to ED amplitudes;
2. VMC energy minimization.

Use paired initial weights and repeat across seeds. Record normalized
log-MSE against ED in both branches wherever full-sector evaluation is feasible.

At checkpoints, **measure but do not modify**:

\[ `\lambda`{=tex}*i(t), `\qquad`{=tex} D*{`\rm eff`{=tex}}(t),
`\qquad`{=tex} `\Delta`{=tex}\_g\^`\psi`{=tex}(t), `\qquad`{=tex} E(t),
`\qquad`{=tex} `\mathrm{Var}`{=tex}(H)(t). \]

Goal: determine whether low-dimensional structure appears naturally
during optimization.

This phase is essential because it separates an emergent property of
log-MSE/VMC optimization from an artifact created by pruning.

### Phase 2 --- One-shot truncation

Take a well-converged model from each training branch and truncate once.

Compare:

1.  weight SVD;
2.  activation SVD;
3.  task-aware/gradient-informed reduction.

For each retained dimension (r), measure the immediate change in

\[ E,`\quad`{=tex} `\mathrm{Var}`{=tex}(H),`\quad`{=tex}
`\Delta`{=tex}\_g\^`\psi`{=tex}, \]

and overlap/fidelity where available.

Also measure the immediate change in normalized log-MSE against ED.
Then fine-tune with the original objective and measure recovery.

This gives a clean compression curve before introducing repeated RG
events.

### Phase 3 --- Periodic RG/SVD training

In each training branch, perform repeated cycles:

\[ `\text{optimize}`{=tex} `\rightarrow`{=tex} `\text{measure}`{=tex}
`\rightarrow`{=tex} `\text{truncate}`{=tex} `\rightarrow`{=tex}
`\text{re-optimize}`{=tex}. \]

Track

\[ H_k,`\quad`{=tex} D\_{`\rm eff`{=tex}},`\quad`{=tex}
`\epsilon`{=tex}\_{`\rm disc`{=tex}},`\quad`{=tex} E,`\quad`{=tex}
`\mathrm{Var}`{=tex}(H),`\quad`{=tex} `\Delta`{=tex}\_g\^`\psi`{=tex} \]

after every RG event, together with normalized log-MSE where available.

### Phase 4 --- Global pooling control

Repeat key experiments using global pooling.

Compare with flatten+FC to determine how much of the observed
compression/symmetry behavior comes from the spatially unconstrained FC
readout.

### Phase 5 --- Symmetry-aware interpretation

Analyze how the retained hidden subspace transforms under known lattice
symmetries.

For a retained basis (U_r), study projected transformations

\[ M_g=U_r\^`\dagger `{=tex}`\rho`{=tex}(g)U_r \]

where an appropriate action (`\rho`{=tex}(g)) on the feature space is
well defined.

Test approximate closure and group relations, but distinguish
hidden-space representation structure from actual wavefunction
invariance.

------------------------------------------------------------------------

## 12. Primary observables

### Fitting and physical quality

-   Normalized full-sector log-MSE against ED, for both branches where feasible
-   Ground-state energy \[ E\_{`\rm VMC`{=tex}} \]
-   Relative energy error \[ `\frac{E_{\rm VMC}-E_0}{|E_0|}`{=tex} \]
-   Energy variance \[
    `\langle `{=tex}H^2`\rangle`{=tex}-`\langle `{=tex}H`\rangle`{=tex}^2
    \]
-   Fidelity/overlap with exact ground state for sufficiently small
    systems
-   Relevant Bose--Hubbard observables already available in the project

### Representation-space quantities

-   Full activation eigenvalue spectrum ({`\lambda`{=tex}\_i})
-   Effective dimension (D\_{`\rm eff`{=tex}})
-   Retained dimension (r)
-   Discarded weight (`\epsilon`{=tex}\_{`\rm disc`{=tex}})
-   Evolution of (D\_{`\rm eff`{=tex}}(t))
-   Stability of retained subspaces between checkpoints

A useful subspace-stability observable is the set of principal angles
between retained subspaces at times (t) and (t+`\Delta `{=tex}t).

### Symmetry quantities

-   (`\Delta`{=tex}\_g\^`\psi`{=tex})
-   (C_g=`\langle`{=tex}`\psi`{=tex}\|U_g\|`\psi`{=tex}`\rangle`{=tex}/`\langle`{=tex}`\psi`{=tex}\|`\psi`{=tex}`\rangle`{=tex})
-   approximate representation matrices (M_g)
-   symmetry behavior before and after truncation

### Engineering quantities

-   trainable parameter count;
-   FC parameter reduction;
-   wall-clock time;
-   memory usage;
-   cost of each SVD/eigendecomposition;
-   cumulative SVD cost;
-   training-time savings, if any;
-   inference/sampling cost.

------------------------------------------------------------------------

## 13. Ablations

Keep the ablation set small but informative.

### Training objective

Compare supervised normalized log-MSE fitting and VMC energy
minimization with paired initializations. Apply the same monitoring,
reduction criteria, and readout controls to both branches.

### Reduction criterion

Compare:

1.  no pruning;
2.  bare weight SVD;
3.  activation SVD;
4.  activation + downstream/task-aware criterion.

### Readout

Compare:

1.  flatten + FC;
2.  global pooling + FC.

### Truncation tolerance

Example sweep:

\[ `\epsilon`{=tex}\_{`\rm disc`{=tex}} =
10^{-1},10^{-2},10^{-3},10^{-4}. \]

Adjust based on observed spectra.

### Truncation cadence

Compare:

-   frequent;
-   intermediate;
-   infrequent;
-   one-shot post-training truncation.

The expectation is that excessively frequent truncation may prevent the
hidden representation from settling.

### Initial hidden dimension

Compare several overparameterized values, e.g.

\[ H=32,64,128. \]

Ask whether the final effective dimension is approximately independent
of sufficiently large (H).

This is a particularly important RG-like test.

### Random seeds

Repeat enough seeds to distinguish reproducible effective-space flow
from optimization noise.

------------------------------------------------------------------------

## 14. Core research questions

### RQ1 --- Do log-MSE fitting and VMC generate low-dimensional hidden representations?

Does

\[ D\_{`\rm eff`{=tex}}(t) \]

decrease and stabilize under each objective? Compare effective ranks
and compressibility at comparable physical accuracy.

### RQ2 --- Is the effective dimension intrinsic?

Starting from different sufficiently large (H), seeds, and
parameterizations, does the learned representation approach a similar
effective rank?

### RQ3 --- Can periodic truncation preserve NQS accuracy?

Can

\[ H`\rightarrow `{=tex}r \]

be performed with negligible change in ground-state energy, variance,
and symmetry diagnostics?

### RQ4 --- Is activation-space truncation superior to weight-space truncation?

At equal retained dimension or parameter count, does activation SVD
preserve the NQS better?

### RQ5 --- Does downstream relevance improve the truncation?

Does a gradient/Fisher/task-aware criterion outperform activation
variance alone?

### RQ6 --- How does readout architecture affect the effective space?

Does flatten+FC retain directions that global pooling removes, and how
does this affect symmetry and compressibility?

### RQ7 --- Does the effective representation encode the known physical symmetry?

As training proceeds and the representation is truncated, does the
retained space become increasingly compatible with the known
translation/(D_4) structure?

### RQ8 --- Does the effective dimension depend on physical regime?

For example, does (D\_{`\rm eff`{=tex}}) differ systematically between
weakly and strongly interacting regimes or near a phase transition?

This should be treated as an extension after the basic method is
validated.

------------------------------------------------------------------------

## 15. Working hypotheses

**H1.** An overparameterized single-layer CNN NQS develops an activation
covariance spectrum with a substantially smaller effective dimension
than the nominal hidden dimension.

**H2.** The effective dimension becomes relatively stable only after
sufficient log-MSE or VMC optimization; overly frequent SVD truncation disrupts
this process.

**H3.** Activation-space truncation preserves the optimized NQS better
than truncation based solely on convolution-weight SVD.

**H4.** Incorporating downstream loss sensitivity further improves
truncation because high-variance but unused hidden directions can
otherwise be incorrectly retained.

**H5.** Once (H) is sufficiently large, the converged effective
dimension is less sensitive to the initial hidden width than the raw
parameter count.

**H6.** Periodic SVD truncation can reduce the effective model size
under either objective while maintaining fitting accuracy, variational
energy, and known ground-state symmetries.

**H7.** Known physical symmetries provide an independent diagnostic of
whether the retained hidden space is physically meaningful rather than
merely numerically low rank.

**H8.** Both log-MSE fitting and VMC may develop compressible hidden
representations, with potentially different ranks and recovery curves.
This is a hypothesis to test, not an assumed outcome.

------------------------------------------------------------------------

## 16. Important pitfalls

### Hidden-feature symmetry is not wavefunction symmetry

Nontrivial hidden symmetry sectors can combine to produce an invariant
final scalar. Always test the final NQS directly.

### Full-rank triviality

A bank of (3`\times3`{=tex}) filters can easily span the full
9-dimensional local kernel space. Bare kernel-subspace closure can
therefore become trivial. Activation statistics and task relevance are
more informative.

### Hidden-channel gauge freedom

Individual channels are not physically meaningful. Arbitrary
invertible/orthogonal mixing of hidden channels can change their
appearance without changing the represented function. Prefer
subspace-level and function-level diagnostics.

### Activation SVD can retain useless directions

Large activation variance does not guarantee downstream relevance. The
FC layer may assign essentially zero weight to that direction.

### Low-variance directions can be important

A small activation direction may have large downstream sensitivity. This
motivates task-aware criteria.

### Nonlinearity prevents naive kernel folding

Post-activation SVD projection generally cannot simply be moved before
the nonlinearity.

### SVD itself can perturb optimization

Frequent truncation may continually rotate/change the optimization
landscape. Allow substantial ordinary optimization between RG events.

### Sampling noise

When covariance is estimated from VMC samples, check convergence versus
sample size and autocorrelation. Exact full-sector diagnostics avoid Monte
Carlo sampling error but still depend on the declared configuration weights.

### Degenerate singular/eigenvalue sectors

Near-degenerate eigenvalues make individual eigenvectors unstable even
when the subspace is stable. Analyze principal subspaces rather than
individual vectors.

### Symmetry-sector choice

Do not assume every nominal square-lattice symmetry applies in every
Hamiltonian/boundary-condition/quantum-number setup. Establish the exact
symmetry group of each benchmark first.

------------------------------------------------------------------------

## 17. Computational strategy

The expensive decomposition is performed only at occasional RG
checkpoints.

Ordinary training remains the existing GPU log-MSE fitting or VMC
optimization, with separate runs for the two objectives.

For a hidden covariance

\[ C`\in`{=tex}`\mathbb `{=tex}R\^{H`\times `{=tex}H}, \]

the eigendecomposition is small for the proposed (H`\sim`{=tex}32-128).
The dominant computational cost is likely to be collecting sufficiently
accurate activation/gradient statistics rather than diagonalizing (C).

For deeper future architectures, avoid blindly decomposing every layer
at every checkpoint. Possible extensions include:

-   layer-specific RG triggers;
-   effective-rank monitoring;
-   randomized/truncated SVD;
-   asynchronous truncation;
-   compression only when expected future training savings exceed
    decomposition overhead.

These are not required for the first project.

------------------------------------------------------------------------

## 18. Literature positioning

The project overlaps with several established areas:

-   channel/filter pruning;
-   low-rank/SVD neural-network compression;
-   activation-aware compression;
-   Taylor-expansion pruning;
-   Hessian/Fisher/curvature-aware pruning;
-   dynamic low-rank training;
-   tensor-network-inspired neural architectures;
-   equivariant/G-CNN architectures;
-   neural-network RG interpretations.

Therefore the initial project should **not** claim that SVD pruning or
task-aware pruning is new.

The potentially distinctive scientific framing to test is:

> During ordinary log-MSE or VMC optimization of an overparameterized CNN NQS, does
> a reproducible low-dimensional and symmetry-structured effective
> hidden representation emerge, and can periodic DMRG-like truncation
> follow this representation flow while preserving the physical ground
> state?

Novelty should be assessed only after a dedicated literature review and
initial results.

------------------------------------------------------------------------

## 19. Minimal implementation milestones

### Milestone 1 --- Baseline

Implement/verify:

\[
`\text{Conv}`{=tex}`\rightarrow`{=tex}`\text{SiLU}`{=tex}`\rightarrow`{=tex}`\text{Flatten}`{=tex}`\rightarrow`{=tex}`\text{FC}`{=tex}
\]

for the existing 2D Bose--Hubbard NQS/VMC framework.

Establish normalized log-MSE, energy, variance, and ED fidelity for
separate log-MSE and VMC baselines.

### Milestone 2 --- Passive representation monitoring

Add hooks that accumulate:

\[ C_z, `\quad`{=tex} {`\lambda`{=tex}*i}, `\quad`{=tex}
D*{`\rm eff`{=tex}}, \]

without modifying training.

Plot their evolution versus log-MSE optimizer step or VMC iteration,
wall-clock time, and achieved physical accuracy.

### Milestone 3 --- Symmetry monitoring

Implement direct wavefunction tests for known translations and (D_4)
operations.

Track symmetry error versus training time.

### Milestone 4 --- One-shot activation truncation

At convergence:

1.  diagonalize (C_z);
2.  choose (r);
3.  insert/project onto (U_r);
4.  transform FC weights;
5.  measure immediate log-MSE/energy/symmetry damage;
6.  re-optimize.

### Milestone 5 --- Compare reduction criteria

Compare weight SVD, activation SVD, and a simple task-aware criterion at
equal (r).

### Milestone 6 --- Periodic RG flow

Introduce several SVD events during each of log-MSE and VMC training.

Measure

\[
H_k, D\_{`\rm eff`{=tex}}, E, `\mathrm{Var}`{=tex}(H), `\epsilon`{=tex}\_{`\rm disc`{=tex}},
\]

and symmetry diagnostics after every event.

### Milestone 7 --- Global-pooling control

Repeat the central experiment with global pooling.

### Milestone 8 --- Physical-regime comparison

Only after the method is stable, compare several (U/t), lattice sizes,
or physical phases.

------------------------------------------------------------------------

## 20. Criteria for a successful first project

The project is scientifically useful if several of the following are
observed:

1.  (D\_{`\rm eff`{=tex}}`\ll `{=tex}H) emerges reproducibly during
    ordinary log-MSE or VMC optimization.
2.  (D\_{`\rm eff`{=tex}}) becomes approximately independent of
    initial (H) once the model is sufficiently overparameterized.
3.  Periodic truncation preserves normalized log-MSE, ground-state
    energy, and energy variance within declared tolerances in each branch.
4.  Activation/task-aware truncation clearly outperforms bare
    weight-based truncation.
5.  The effective hidden space stabilizes between RG events.
6.  Known physical symmetries are preserved or become cleaner under
    successful truncation.
7.  The final reduced model has substantially fewer effective degrees of
    freedom without loss of physical accuracy.
8.  Effective dimension or symmetry content changes systematically
    across physically distinct Bose--Hubbard regimes.

A negative result is also informative: if the activation spectrum
remains broad, the retained subspace is unstable, or truncation
repeatedly harms log-MSE or VMC optimization, that would constrain the usefulness
of the DMRG analogy.

------------------------------------------------------------------------

## 21. Suggested first figures

1.  **Normalized log-MSE and energy vs training step** for both
    objectives, with RG/SVD events marked.
2.  **Activation eigenvalue spectrum** at several training checkpoints.
3.  **Effective dimension (D\_{`\rm eff`{=tex}}(t))** vs training step.
4.  **Retained hidden dimension (H_k)** after successive RG events.
5.  **Energy error vs retained dimension (r)** for weight, activation,
    and task-aware truncation.
6.  **Energy before truncation / immediately after / after
    re-optimization**.
7.  \*\*Symmetry error (`\Delta`{=tex}\_g\^`\psi`{=tex}(t))\*\* for
    representative (D_4) and translation operations.
8.  **Compression/parameter count vs log-MSE and variational error**,
    comparing the two training objectives.
9.  **Effective dimension vs initial hidden width (H)**.
10. If stable enough, **effective dimension vs (U/t)**.

------------------------------------------------------------------------

## 22. Immediate next experiment

Start with one small periodic square lattice for which the ground-state
benchmark is trustworthy.

Use

\[ H=64 \]

and train the same single-CNN + SiLU + flatten + FC NQS in two
independent branches from paired initializations: **ED log-MSE fitting**
and **VMC energy minimization**, both **without any pruning**.

At fixed checkpoints in each branch:

1.  estimate the centered activation covariance (C_z);
2.  save its full eigenvalue spectrum;
3.  compute (D\_{`\rm eff`{=tex}});
4.  compute several candidate retained ranks (r(`\epsilon`{=tex}));
5.  evaluate the wavefunction symmetry errors;
6.  save normalized log-MSE, ED fidelity, energy, and energy variance.

Use the shared diagnostic distribution in Section 1.1 and repeat across
paired seeds. Compare against optimizer step, time, and physical accuracy.

The first question to answer is simply:

\[ `\boxed{
\text{Do log-MSE fitting and VMC each produce a stable low-dimensional,
symmetry-structured hidden representation?}
}`{=tex} \]

Only after answering this should periodic SVD truncation be enabled.

------------------------------------------------------------------------

## 23. Longer-term extensions

If the minimal study is successful:

-   dynamically grow as well as shrink hidden dimension;
-   develop a principled task-aware/Fisher reduced density matrix;
-   compare with hard (D_4)-equivariant CNNs;
-   perform irrep decomposition of the retained hidden space;
-   study approximate/broken symmetries;
-   extend to multiple CNN layers;
-   introduce spatial coarse-graining so the construction becomes closer
    to genuine RG;
-   compare effective dimension with correlation
    length/entanglement/criticality;
-   use the method as adaptive architecture selection for NQS;
-   investigate whether RG flow of learned representations provides
    useful physical diagnostics beyond compression.

------------------------------------------------------------------------

## 24. One-sentence project summary

**Train an intentionally overparameterized single-layer CNN NQS for the
2D Bose--Hubbard ground state using both supervised log-MSE fitting and
VMC energy minimization, observe whether each naturally generates a
low-dimensional and symmetry-structured activation space, and
periodically truncate that space by SVD/eigendecomposition in a
DMRG-inspired RG procedure while monitoring log-MSE, variational accuracy,
symmetry, and computational complexity.**

------------------------------------------------------------------------

## 25. Separate workstream: preparation before Julia integration

Retain the following preparation and review requirements from the prior plan.
This consolidation does not authorize a Julia merge, root-package edits, or
changes to 1D functionality. Preserve benchmark/provenance dependencies and
accepted artifacts. The checklist describes the preparation scope; it does
not claim every item is complete or reopen completed cleanup work.

1. Document the current 2D public functions, returns, shapes, failure modes and
   device/precision choices in [docs/API.md](docs/API.md).
2. Keep one shared PyTorch VMC core with tested derivative/solver options;
   do not introduce a duplicate Julia or NumPy VMC implementation.
3. Remove disposable caches/previews, archive unneeded scratch material and
   superseded prose, and preserve every benchmark/provenance dependency.
4. Review the existing PythonCall/1D adapter and explicitly document geometry,
   interaction, indexing, checkpoint and raw-log-amplitude differences in
   [docs/JULIA_PREPARATION.md](docs/JULIA_PREPARATION.md).
5. Verify the documentation examples, import/CLI entry points, CPU regression
   tests, protected inputs and local links after cleanup.
6. Stop at a reviewable handoff. A future user-authorized merge should first add
   collision-free Python namespaces and metadata-dispatched inference, then
   batch observables/VMC, followed by Julia↔Python parity checks. No such adapter
   or Julia integration test is claimed in this phase.
