# Physics and conventions

## Hamiltonian

The 2D implementation uses

$$
H=-t\sum_{\langle i,j\rangle}(b_i^\dagger b_j+b_j^\dagger b_i)
+\frac{U_{2D}}2\sum_i n_i(n_i-1)-\mu\sum_i n_i.
$$

The existing 1D implementation uses $U_{1D}n_i(n_i-1)$ instead. To match
interaction strengths, set $U_{1D}=U_{2D}/2$. Loading a model or changing its
device does not convert this physical convention.

The cutoff `Nmax` is inclusive: $0\le n_i\le N_{\max}$. Fixed-sector ED and VMC
use $\sum_i n_i=N_{\mathrm{tot}}$. Larger cutoffs and thermodynamic extrapolation
are separate physical checks, not consequences of a high finite-system fidelity.

## Geometry and indexing

For 2D, physical zero-based site index is `i = y*Lx + x`. A Julia lattice matrix
with shape `(Lx, Ly)` has this order when flattened with `vec(image)`. A batch
of flat configurations has shape `(Lx*Ly, walkers)` in Julia.

The boundary flags `pbc_x` and `pbc_y` are independent. The 2D graph uses unique
unordered hopping edges, with no self bonds. A periodic axis of length two
has one edge; the historical 1D short-ring convention includes parallel bonds.
Use open chains or periodic length at least three for cross-dimension parity
tests after converting the interaction coefficient.

## Network output and normalization

`nn_value` and `nn_values` return **unnormalized real log amplitudes**. They are
neither probabilities nor normalized wavefunctions. For a complete sector,

$$
\log Z=\operatorname{logsumexp}_{\mathbf n}(2f_\theta(\mathbf n)),\qquad
P(\mathbf n)=\exp(2f_\theta(\mathbf n)-\log Z).
$$

A raw network has no particle-number projection. Inference on configurations
outside its training sector is allowed but unconstrained. VMC proposals
preserve a chosen particle number; this does not prove the raw network vanishes
in other sectors. Never normalize each walker batch independently and treat
those values as a globally normalized state.

## Symmetry

GroupEquivariantConv2DNet uses a lifting convolution, a group convolution,
pointwise SiLU activations and a mean readout over sites and orientations.
Its internal features are equivariant and its scalar output is invariant.

On periodic squares, it enforces translations and D4 rotations/reflections.
On periodic rectangles, the same D4-tied local filters enforce translations
and D2 lattice symmetry. A quarter turn relates transposed shapes, not two
states on the same non-square lattice. Both axes must be periodic for this
ansatz; other families support their documented boundary choices.

## Training objectives: log-MSE and VMC

Changing the training objective does not change the ansatz or its inference
API. Both produce the raw log amplitude $f_\theta$. The distinction is where
the training signal comes from:

| Workflow | Objective | Training information | Update examples |
| --- | --- | --- | --- |
| ED-supervised fitting | Normalized log-amplitude MSE | Exact small-sector ED state and all configurations in that sector | Adam |
| Energy-based VMC | Variational energy | Network samples and Hamiltonian matrix elements; no ED state needed | Native energy gradient with Adam, or SR with SGD |

For a sector $\mathcal S$, define

$$
\ell_\theta(n)=f_\theta(n)-\tfrac12\operatorname{logsumexp}_{m\in\mathcal S}
\bigl(2f_\theta(m)\bigr).
$$

Log-MSE minimizes

$$
\mathcal L_{\log\mathrm{MSE}}=
\frac1{|\mathcal S|}\sum_{n\in\mathcal S}
\bigl(\ell_\theta(n)-\ell_{ED,\epsilon}(n)\bigr)^2.
$$

The ED log target is normalized by the same operation. The legacy 1D helper
starts from $\log(|\psi_{ED}|+\epsilon)$; the 2D trainer starts from
$\log\max(|\psi_{ED}|,\epsilon)$. Both default to $\epsilon=10^{-12}$, but their
small-amplitude regularizations are not identical. The 2D trainer retains the
raw ED vector for physical overlap/energy diagnostics and rejects zero or
mixed-sign components instead of hiding them with an absolute value. The
legacy 1D helper takes absolute values and casts targets to float32. A low log-MSE is not
itself an energy-convergence certificate.

For VMC,

$$
E_\theta=\mathbb E_{n\sim P_\theta}[E_{\mathrm{loc}}(n)],\qquad
E_{\mathrm{loc}}(n)=\sum_m H_{nm}\exp\bigl(f_\theta(m)-f_\theta(n)\bigr),
$$

where $P_\theta(n)\propto\exp(2f_\theta(n))$ within the sampled sector.
Native updates use the sampled energy gradient; SR preconditions the update
using log-amplitude parameter derivatives. The historical 1D and current 2D
SR implementations use different conventions. For $M$ samples, let
$J_{sk}=\partial_{\theta_k}f_\theta(n_s)$, $e_s=E_{\mathrm{loc}}(n_s)-\bar E$
and $J_c=J-\operatorname{mean}_{s}J$. The documented ridge variants solve

$$
\begin{aligned}
\text{1D:}\quad &(J^\mathsf TJ+\tau I)\delta=J^\mathsf T e,\\
\text{2D:}\quad &\left(\frac{J_c^\mathsf TJ_c}{M}+\tau I\right)\delta
=\frac{J_c^\mathsf T e}{M}.
\end{aligned}
$$

The 1D matrix is uncentered and unnormalized; the 2D matrix is a sample
covariance. Both SR directions above omit the extra factor of two used in the
native energy gradient. With SGD the update is $\theta\leftarrow\theta-\eta\delta$.
The 1D diagonal shift is absolute, so its strength relative to the Gram matrix
changes with sample count. **Do not transfer `tau` or learning-rate values
between dimensions without retuning.** The 1D dense solve also lacks the 2D
solver's pre-allocation memory guards.

Adam/SGD choose how the update is applied: VMC and Adam are different kinds of
choice, not competing objectives. The 2D SR route requires SGD; both guides
show SR+SGD examples. The 1D trainer also accepts SR+Adam, which sends the
preconditioned direction to Adam as its gradient; this is a different update
from SR+SGD. The 2D SR implementation rejects Adam.

An optional ED reference during VMC is a diagnostic and does not turn it into
supervised training. VMC avoids an ED target but still needs convergence,
mixing and uncertainty checks. The positive-real ansatz used here does not
represent an arbitrary signed or complex wavefunction.

## Accuracy and uncertainty

The absolute overlap is $|\langle\psi_{ED}|\psi_\theta\rangle|$; squared
fidelity is its square. We label both explicitly. Energy, overlap and variance
are separate diagnostics. Local-energy variance or standard deviation is not
an error bar on the mean. Conditional block errors assume suitable mixing and
stationarity; finite-size measurements do not establish thermodynamic phases.
