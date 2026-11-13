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

## Accuracy and uncertainty

The absolute overlap is $|\langle\psi_{ED}|\psi_\theta\rangle|$; squared
fidelity is its square. We label both explicitly. Energy, overlap and variance
are separate diagnostics. Local-energy variance or standard deviation is not
an error bar on the mean. Conditional block errors assume suitable mixing and
stationarity; finite-size measurements do not establish thermodynamic phases.
