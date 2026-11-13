# BoseHubbardNQS

Learn and evaluate neural quantum states for the Bose–Hubbard model using
PyTorch, variational Monte Carlo, and a Julia interface.

The numerical work lives in Python. Julia loads the same trained networks,
evaluates individual configurations or batches of walkers, and launches the
existing 2D training and measurement workflows.

| Capability | 1D | 2D |
| --- | --- | --- |
| Fixed-particle-number ED references | Supported for small sectors | Supported with dimension and memory guards |
| Supervised NQS training | ED amplitude/log-amplitude experiments | Normalized log-MSE with Adam |
| Energy-based VMC | CPU/CUDA, native gradients and SR | One CPU/CUDA core, native gradients and SR |
| Julia inference | Scalar and batched configurations | Scalar and batched configurations with geometry metadata |
| Spatial symmetry | Architecture-dependent | Includes a D4-tied group-equivariant CNN |

## Choose a starting point

- [Install and run a small example](getting-started.md).
- [Train and measure a state in Python](python.md).
- [Use the Julia interface](julia.md).
- [Evaluate batches on CPU or CUDA](batching.md).
- [Inspect measured accuracy and its limits](benchmarks.md).

## Neural quantum states

For a configuration $\mathbf n=(n_1,\ldots,n_L)$, a network returns a real,
unnormalized log amplitude $f_\theta(\mathbf n)$. Within a chosen particle sector,

$$
\psi_\theta(\mathbf n)=
\frac{\exp f_\theta(\mathbf n)}
{\sqrt{\sum_{\mathbf m}\exp(2f_\theta(\mathbf m))}}.
$$

ED-supervised fitting compares the network with an exact small-system state.
VMC instead samples the network and minimizes its energy. Both use the same
ansatz; changing the training objective does not require a second network
implementation.

The supported 1D VMC families are FCNet, JastrowNet, Conv1DNet and RBMNet.
The 2D implementation includes FCNet, JastrowNet, Conv2DNet, RBMNet,
TransformerNet, Conv2DTransformerNet and GroupEquivariantConv2DNet.

Read [the conventions](physics.md) before comparing energies or transferring
settings between dimensions. The current 1D and 2D interaction parameters use
different factors of two.
