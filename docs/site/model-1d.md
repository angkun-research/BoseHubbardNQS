# 1D Bose–Hubbard model

This guide trains a state in Python and evaluates the saved network in Julia.
Run shell commands and Julia sessions from the repository root, after following
[Getting started](getting-started.md). The examples use small systems to show
the API; their short training budgets do not establish ground-state convergence.

## Hamiltonian and network output

The 1D implementation uses

$$
H_{1D}=-t\sum_{\langle i,j\rangle}(b_i^\dagger b_j+b_j^\dagger b_i)
+U\sum_i n_i(n_i-1)-\mu\sum_i n_i.
$$

Occupations satisfy $0\leq n_i\leq N_{\max}$, and sampling fixes
$\sum_i n_i=N_{\mathrm{tot}}$. A network returns a real log amplitude
$f_\theta(\mathbf n)$, with $\psi_\theta(\mathbf n)\propto\exp f_\theta(\mathbf n)$.
For VMC the sampling weight is proportional to $\exp(2f_\theta)$.
The one-hot input is $X_{ic}=\mathbf1[n_i=c]$, with $c=0,\ldots,N_{\max}$;
write $v=\operatorname{vec}X$ for the flattened input.

The interaction coefficient here has **no factor of one half**. Match the 2D
convention with $U_{1D}=U_{2D}/2$; see [physics conventions](physics.md).

## Supported NQS ansatzes

These four families are supported by `TrainVMCstates.py` and Julia's `load_nn`.
All produce an unnormalized scalar log amplitude.

| `--arch` | Construction | Meaning of `--hidden_dim` |
| --- | --- | --- |
| `FCNet` | Two hidden tanh layers and a linear scalar head | Width of both hidden layers |
| `JastrowNet` | Onsite occupation table plus distance-shared pair tables | Unused |
| `Conv1DNet` | Circular width-3 convolution, then a flattened dense head; SiLU activations | Convolution channels and dense hidden width |
| `RBMNet` | Real binary-hidden restricted Boltzmann machine | Number of hidden units |

### FCNet

With trainable weights and biases,

$$
h_1=\tanh(W_1v+b_1),\qquad
h_2=\tanh(W_2h_1+b_2),\qquad
f_\theta=w^\mathsf T h_2+b.
$$

The flattened input and dense layers are tied to the trained site count.

### JastrowNet

$$
f_\theta(\mathbf n)=\sum_i a(n_i)
+\sum_{i<j}V_{d(i,j)}(n_i,n_j),\qquad
d(i,j)=\min(j-i,L-(j-i)).
$$

The implementation learns occupation tables rather than restricting the pair
term to a single coefficient multiplying $n_i n_j$. The 1D pair tables are not
explicitly symmetrized under exchange of the occupation arguments. Distance
sharing alone should therefore not be interpreted as a guarantee of every
lattice symmetry.

### Conv1DNet

For $\phi=\operatorname{SiLU}$ and a circular convolution $K*X$,

$$
u=\phi(K*X+c),\qquad
f_\theta=w^\mathsf T\phi(W\operatorname{vec}u+b)+b_{\mathrm{out}}.
$$

The flattened dense readout does not enforce translation invariance. Its input
size is fixed by $L$. The convolution remains circular even when `--no-pbc`
selects an open Hamiltonian; the Jastrow distances above also remain ring-based.
The Hamiltonian boundary choice and the network's structural prior are separate.

### RBMNet

$$
f_\theta(v)=a^\mathsf T v+
\sum_{j=1}^{H}\operatorname{softplus}\!\left(b_j+\sum_k v_kW_{kj}\right),
\qquad \operatorname{softplus}(z)=\log(1+e^z).
$$

This expression is used directly as the log amplitude, with no extra square
root or factor of one half. The one-hot visible units are the components of $v$.

The source also contains older experimental classes such as `Conv1DPoolNet`,
`TransformerNet`, `ConvTransformer` and `ViTNet`. They are not registered choices
in the supported 1D VMC trainer or Julia checkpoint loader described here.

## Train in Python

There are two different objectives: fit an ED state with **log-MSE**, or
minimize the energy with **VMC**. Adam and SGD are optimizers; native gradients
and SR are update choices within VMC. See the
[objective comparison](physics.md#training-objectives-log-mse-and-vmc).

### Fit a small ED state with log-MSE and Adam

The existing supervised helper functions are in `TrainEDstates.py`. Its main
program has settings embedded in the source, so use the following explicit
Python recipe rather than passing it VMC CLI flags. This example trains the
same four-site Conv1D architecture used below and saves a Julia-compatible
checkpoint:

```bash
.venv/bin/python - <<'PYTHON'
import sys
from pathlib import Path
import torch

# The legacy 1D supervised helpers use sibling imports.
sys.path.insert(0, str(Path("pythonTest").resolve()))
from TrainEDstates import make_ed_dataset, train_model_on_ed, compare_predictions
from NeuralNetworkClass import Conv1DNet, save_model

torch.manual_seed(123)
L, Ntot, Nmax, width = 4, 4, 2, 8
device = "cpu"  # or "cuda:0" with a working CUDA PyTorch environment
x, y, energy, basis, psi_ed = make_ed_dataset(
    L=L, Ntot=Ntot, Nmax=Nmax, t=1.0, U=8.37, mu=0.0,
    pbc=True, target="logabs")
model = Conv1DNet(n_sites=L, channels=Nmax + 1, hidden_dim=width)
model = train_model_on_ed(model, x, y, target="logabs",
                          lr=0.001, epochs=20, device=device)
compare_predictions(model, x, y, target="logabs", device=device)
path = Path("data/docs-1d/conv-logmse.pt")
path.parent.mkdir(parents=True, exist_ok=True)
save_model(path, model.cpu().eval(), "Conv1DNet", L, Nmax, width,
           extra={"training_method": "ed_log_mse", "target": "logabs"})
PYTHON
```

`target="logabs"` is essential: it makes this a normalized log-amplitude MSE
with Adam, not the helper's legacy amplitude-target mode. The target is
$\log(|\psi_{ED}|+\epsilon)$, with $\epsilon=10^{-12}$ by default, and both
predicted and target log amplitudes are normalized over the complete sector.
The ED matrix is dense: keep this recipe to small sectors. The helper fits its
final weights at the requested budget; it does not implement the 2D trainer's
best-loss checkpoint-selection/reporting policy. The recipe prints full-sector
MSE and normalized absolute overlap against the regularized ED target as a
sanity check; this is not an independent raw-ED fidelity report. Keep `epochs >= 10`
with this legacy helper: its progress-print interval is `epochs // 10`.

To change architecture, use `FCNet(n_sites=L, channels=Nmax+1, hidden_dim=width)`,
`JastrowNet(n_sites=L, channels=Nmax+1)` or
`RBMNet(n_sites=L, channels=Nmax+1, n_hidden=width)` from
`NeuralNetworkClass`. Save the matching architecture name and width metadata.
Use a new output path to keep previous checkpoints.

### Small energy-based VMC example

This trains a four-site Conv1D state using native energy gradients and Adam:

```bash
.venv/bin/python pythonTest/TrainVMCstates.py \
  --arch Conv1DNet --L 4 --Ntot 4 --Nmax 2 --hidden_dim 8 \
  --t 1 --U 8.37 --mu 0 --pbc \
  --epochs 20 --update_method native --optimizer adam --lr 0.001 \
  --n_walkers 16 --n_steps 8 --burnin_steps 8 --eval_every 10 \
  --ed-reference --device cpu --seed 123 \
  --output data/docs-1d/conv-vmc.pt
```

The 1D ED reference builds a dense matrix. This example's sector is small;
remove `--ed-reference` for larger systems. That flag adds a reference energy:
**the optimization remains energy-based VMC**, not supervised ED fitting.

To change ansatz, replace `--arch` with any name in the table. `--hidden_dim`
controls the width or hidden-unit count; Jastrow ignores it. Choose a distinct
output filename for each run. The output path is relative to your shell's
working directory and its parent directories are created automatically.

### Stochastic reconfiguration and GPU options

For SR, replace the update and optimizer options with:

```text
--update_method sr --optimizer sgd --lr 0.005 --tau 0.001
```

These replace the corresponding options in the complete command; they are not
standalone shell commands. `tau` regularizes the SR solve. This 1D implementation uses an uncentered, unnormalized Gram
matrix; its `tau` and learning rate are not directly comparable to the 2D
convention. See the [SR equations](physics.md#training-objectives-log-mse-and-vmc).
Start with small networks because dense SR memory grows quadratically with
parameter count and the 1D solve has no pre-allocation memory guard.

Use `--device auto` to let the 1D trainer select an NVIDIA GPU, or
`--device cuda:0` to request one explicitly. The trainer's `auto` policy queries
GPU utilization and free memory, requires at least 10 GB free, and falls back to
CPU if no GPU qualifies or the query fails. The Julia loader's `auto` policy
simply selects CUDA when PyTorch reports availability, otherwise CPU.
Use `--evaluation_batch_size 128` to chunk model evaluations during sampling
and updates. This is distinct from the number of walkers.

| Training option | Purpose and default |
| --- | --- |
| `--L`, `--Ntot`, `--Nmax` | Sites, fixed particle number, inclusive cutoff; defaults `4`, `4`, `4` |
| `--arch`, `--hidden_dim` | Model family and capacity; defaults `FCNet`, `64` |
| `--t`, `--U`, `--mu` | Hamiltonian parameters; defaults `1`, `2`, `0` in the 1D convention |
| `--pbc` / `--no-pbc` | Periodic / open hopping; periodic by default |
| `--update_method` | `sr` (default) or `native` |
| `--optimizer` (alias `--native_opt`) | `sgd` or `adam`; omitted selects SGD for SR and Adam for native |
| `--epochs`, `--lr`, `--tau` | Updates, learning rate, SR regularization; defaults `100`, `0.005`, `0.001` |
| `--n_walkers`, `--n_steps`, `--burnin_steps` | Parallel walkers, retained steps and burn-in steps per epoch; defaults `64`, `32`, `32` |
| `--eval_every` | Progress interval; default `10` |
| `--evaluation_batch_size` | Optional positive evaluation chunk size; omitted evaluates the full batch |
| `--ed-reference` | Optional small-sector ED energy diagnostic; disabled by default |
| `--device`, `--seed` | Execution device and random seed; defaults `auto`, `123` |
| `--output` | Inference checkpoint; default `data/bose_hubbard_vmc_nqs.pt` under the repository |

These are the principal training options; run
`.venv/bin/python pythonTest/TrainVMCstates.py --help` for the complete CLI.

The supervised recipe and VMC CLI are separate entry points. `--ed-reference`
on the VMC command only measures a reference energy; it does not switch the
training objective to log-MSE.

## Use the trained state in Julia

Start Julia with the checkout's Python interpreter:

```bash
./julia-local
```

The following loads the checkpoint produced by the complete Python command:

```julia
using BoseHubbardNQS

model = load_nn("data/docs-1d/conv-vmc.pt"; device="auto")
sigma = [1, 1, 1, 1]                         # occupation counts, not one-hot
logpsi = nn_value(model, sigma)
same_logpsi = ansatz_log_amplitude(model, sigma)

configs = hcat(sigma, [0, 2, 1, 1])          # sites × walkers, shape (4, 2)
values = nn_values(model, configs; chunk_size=128)
```

`logpsi` is a Julia `Float64`; `values` is a `Vector{Float64}`. The 1D forward
calculation remains float32. These are raw log amplitudes, not normalized
probabilities. Scalar and batch calls infer the saved occupation cutoff.

### Loading and inference keyword choices

| Julia keyword | Choices and behavior |
| --- | --- |
| `load_nn(...; device=nothing)` | Default CPU behavior; `"cpu"`, `"auto"`, `"cuda"` or `"cuda:0"` select execution placement. `auto` means CUDA if available, otherwise CPU. |
| `load_nn(...; map_location="cpu")` | Checkpoint deserialization location; does not by itself move the constructed model onto a GPU. Usually keep the default and choose `device`. |
| `load_nn(...; arch=nothing)` | Metadata override: one of the four registered names. Normally leave unset. |
| `load_nn(...; n_sites=nothing, nmax=nothing, hidden_dim=nothing)` | Optional architecture overrides. They must remain compatible with the saved weight shapes; normally infer them from the checkpoint. |
| `nn_value(...; nmax=nothing)` | Infer cutoff by default. An explicit integer must match the loaded model's cutoff. Also accepted by `ansatz_log_amplitude`. |
| `nn_values(...; nmax=nothing, chunk_size=1024)` | Same cutoff rule; positive integer chunk size limits configurations per forward pass. |

For example, the metadata for this particular checkpoint could be specified
explicitly, although normal loading does not need the overrides:

```julia
cpu_model = load_nn("data/docs-1d/conv-vmc.pt";
                    arch="Conv1DNet", n_sites=4, nmax=2, hidden_dim=8,
                    map_location="cpu", device="cpu")
values_cpu = nn_values(cpu_model, configs; nmax=2, chunk_size=1)
```

An unavailable explicit CUDA device raises an error. Existing
`load_nn(path)` / `nn_value(model, sigma)` calls keep their CPU behavior.
Changing the loader's dimension or cutoff does not retrain or resize a model.

The log-MSE checkpoint uses the same loader and inference API:

```julia
supervised = load_nn("data/docs-1d/conv-logmse.pt"; device="cpu")
logpsi_ed_fit = nn_value(supervised, sigma)
values_ed_fit = nn_values(supervised, configs; chunk_size=128)
```

The objective used during training does not change the meaning of the network
output or the Julia call signatures.

## Reopen the same checkpoint in Python

In a Python session started with `.venv/bin/python` from the repository root:

```python
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("src/python").resolve()))
import Bose_Hubbard_NQS as nqs

model_python = nqs.load_model("data/docs-1d/conv-vmc.pt", device="cpu")
logpsi_python = nqs.predict_scalar(model_python, [1, 1, 1, 1])
configs_python = np.array([[1, 0], [1, 2], [1, 1], [1, 1]], dtype=np.int64)
values_python = nqs.predict_values(model_python, configs_python, chunk_size=128)
```

This is the same inference bridge Julia uses; the helper's batch input also
has sites on rows and walkers on columns. Replace the filename with
`data/docs-1d/conv-logmse.pt` to reopen the supervised fit. No conversion is
required between languages. Device kernels can introduce small floating-point
differences; compare values with dtype-appropriate tolerances.

Continue with [the Julia API](api/julia.md), [batching and CUDA](batching.md),
or [the retained accuracy benchmarks](benchmarks.md).
