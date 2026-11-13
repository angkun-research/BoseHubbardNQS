# 2D Bose–Hubbard model

This guide covers all seven supported 2D ansatzes, Python training using ED or
VMC, and Julia inference with the saved checkpoints. Run commands from the
repository root after [installing the environment](getting-started.md).
The short examples demonstrate the API, not converged ground states.

## Hamiltonian, geometry and output

$$
H_{2D}=-t\sum_{\langle i,j\rangle}(b_i^\dagger b_j+b_j^\dagger b_i)
+\frac U2\sum_i n_i(n_i-1)-\mu\sum_i n_i.
$$

The lattice has $L_xL_y$ sites, with independent boundary flags for each axis.
The occupation cutoff is inclusive, $0\leq n_i\leq N_{\max}$, and ED/VMC fix
$\sum_i n_i=N_{\mathrm{tot}}$. Omitting `Ly` selects a square; omitting `Ntot`
selects unit filling. The hopping graph uses unique unordered nearest-neighbor
edges. Compare dimensions only after matching the
[interaction and short-axis conventions](physics.md).

Physical site order is $i=yL_x+x$ with zero-based coordinates. Networks take
one-hot $X_{ic}=\mathbf1[n_i=c]$ and return $f_\theta(\mathbf n)\in\mathbb R$;
$\psi_\theta(\mathbf n)\propto\exp f_\theta(\mathbf n)$. Network inference is
unprojected: it accepts valid occupation counts in any particle sector.

## Supported NQS ansatzes

All seven names work with `train_ed.py`, `train.py`, checkpoint loading and
Julia's `load_nn_2d`. ED fitting and VMC use the same model classes.

| `--arch` | Architecture | Capacity and boundary notes |
| --- | --- | --- |
| `FCNet` | Tanh multilayer perceptron | Configurable hidden width/depth; any axis boundaries |
| `JastrowNet` | Onsite and pair occupation tables | Width unused; displacement sharing by default; optional reflection sharing on cylinders |
| `Conv2DNet` | One 3×3 convolution and flattened dense head | Hidden width sets channels and head width; any axis boundaries |
| `RBMNet` | Real binary-hidden RBM | Width sets hidden-unit count; any axis boundaries |
| `TransformerNet` | Occupation/position embeddings, one attention block and mean pooling | One head; hidden width; any axis boundaries |
| `Conv2DTransformerNet` | 3×3 convolution, attention block and flattened dense head | Four heads; hidden width must be divisible by four; any axis boundaries |
| `GroupEquivariantConv2DNet` | D4 lifting convolution, group convolution and invariant pooling | Hidden width; both axes must be periodic |

For ED training, capacity is `--width`; for VMC it is `--hidden-dim`.
The Python constructor setting is `hidden_dim`, except RBM uses `n_hidden`
and Jastrow needs neither. Checkpoints retain these settings.

### FCNet

Let $v=\operatorname{vec}X$, $h_0=v$ and

$$
h_{\ell}=\tanh(W_{\ell}h_{\ell-1}+b_{\ell}),\qquad
f_\theta=w^\mathsf T h_d+b.
$$

There are two hidden layers by default. `--hidden-layers` changes $d$ for
FCNet only. Dense input weights fix the site count and impose no lattice
symmetry.

### JastrowNet

The default `pair_sharing="displacement"` uses

$$
f_\theta(\mathbf n)=\sum_i a(n_i)+
\sum_{i<j}\overline V_{d_x(i,j),d_y(i,j)}(n_i,n_j),\qquad
\overline V_d(p,q)=\tfrac12\bigl(V_d(p,q)+V_d(q,p)\bigr).
$$

Each distance is the absolute coordinate difference, minimized around the
corresponding axis when that axis is periodic. Unlike a single $n_i n_j$
coefficient, each displacement has a full occupation-pair table.

The optional `--pair-sharing reflection_orbit` requires **open x, periodic y**.
It uses onsite x-reflection classes and reflected unordered x-pair/minimal-y
classes. Pair canonicalization can swap occupation roles; only same-x or
self-reflection pair classes are transpose-symmetrized. This retains distinct
edge/bulk tables. It is a separate table-sharing choice within JastrowNet.

### Conv2DNet

For $\phi=\operatorname{SiLU}$,

$$
u=\phi(K*X+c),\qquad
f_\theta=w^\mathsf T\phi(W\operatorname{vec}u+b)+b_{\mathrm{out}}.
$$

The 3×3 convolution pads each axis circularly if periodic, or with zero
features if open. The flattened readout does not enforce translation
invariance. Its weight dimensions depend on the number of sites.

### RBMNet

$$
f_\theta(v)=a^\mathsf T v+
\sum_{j=1}^{H}\log\!\left(1+\exp\left[b_j+\sum_k v_kW_{kj}\right]\right).
$$

This is the log amplitude directly, evaluated with stable `softplus`.
The constructor's `n_hidden` and the CLI's width determine $H$.

### TransformerNet

At site $(x_i,y_i)$, the token is

$$
z_i=W_eX_i+b_e+p^{(x)}_{x_i}+p^{(y)}_{y_i},\qquad
f_\theta=w^\mathsf T\left[\frac1{L_xL_y}\sum_i\mathcal T(z)_i\right]+b.
$$

Here $\mathcal T$ is one standard transformer encoder layer with one attention
head, residual connections, layer normalization and a ReLU feed-forward block.
For attention, $\operatorname{Attn}(Q,K,V)=\operatorname{softmax}(QK^\mathsf T/
\sqrt{d_k})V$. The feed-forward width defaults to twice the hidden width;
Python construction also accepts `feedforward_dim`. Dropout is zero.
Learned x/y positions mean mean pooling alone does not impose lattice symmetry.

### Conv2DTransformerNet

$$
u_i=\operatorname{SiLU}(K*X+c)_i,\qquad
f_\theta=w^\mathsf T\operatorname{SiLU}
\left(W\operatorname{vec}\mathcal T(u)+b\right)+b_{\mathrm{out}}.
$$

The convolution uses the axis padding described above. The encoder has four
attention heads, ReLU feed-forward width $2h$ and zero dropout. There are no
positional embeddings or mean pooling. The flattened dense head fixes the
site count and does not enforce translation invariance.

### GroupEquivariantConv2DNet (GCNN)

The orientation index $h\in D_4$ runs over four rotations and four reflected
rotations. With a 3×3 displacement stencil $\Delta$ and
$\phi=\operatorname{SiLU}$, the lifting and group layers have the form

$$
F^{(1)}_{h,\alpha}(r)=\phi\!\left(
 b_\alpha+\sum_{\delta\in\Delta,c}
 K_{\alpha c}(h^{-1}\delta)X_c(r+\delta)\right),
$$

$$
F^{(2)}_{h,\alpha}(r)=\phi\!\left(
 d_\alpha+\sum_{a\in D_4,\delta\in\Delta,\beta}
 W_{\alpha\beta}(h^{-1}a,h^{-1}\delta)
 F^{(1)}_{a,\beta}(r+\delta)\right),
$$

$$
f_\theta=b_{\mathrm{out}}+
\sum_\alpha w_\alpha\frac1{8L_xL_y}
\sum_{h\in D_4,r}F^{(2)}_{h,\alpha}(r).
$$

Coordinates wrap periodically. Internal features are equivariant; the mean
readout makes the scalar log amplitude invariant. On squares, the guaranteed
lattice symmetries are translations and D4. On rectangles, the same D4-tied
local filters guarantee translations and **D2**: 180° rotations and axis
mirrors. A quarter turn exchanges rectangular shapes; it is not a symmetry of
a fixed non-square lattice. Open or mixed boundaries are rejected for GCNN.

The convolutional weights and pooled head can have the same parameter shapes
across sizes, but the current checkpoint loader restores the **saved geometry**.
`load_nn_2d` has no lattice-size override. A new training run constructs a model
for its requested geometry; loading alone does not transfer a checkpoint to a
new lattice or establish its accuracy there.

## Train in Python

**Log-MSE** fits an ED target; **VMC** minimizes a sampled energy. These are
objectives/workflows, whereas Adam/SGD are optimizers and SR/native gradients
are VMC update choices. See the
[objective comparison](physics.md#training-objectives-log-mse-and-vmc).

### Fit a small ED ground state

This fits the log amplitude of a 2×3 fixed-sector ED state using the GCNN:

```bash
.venv/bin/python -B pythonTest/2D/train_ed.py \
  --arch GroupEquivariantConv2DNet --Lx 2 --Ly 3 --Ntot 6 --Nmax 2 \
  --width 8 --t 1 --U 16.74 --mu 0 --pbc \
  --loss log_mse --epochs 20 --lr 0.001 --eval-every 10 \
  --dtype float64 --device cpu --seed 123 \
  --output outputs/docs-2d/ed-gcnn.pt
```

This small 2×3 example has one unique hopping bond along its length-two axis;
the GCNN's ±1 convolution offsets on that axis reach the same neighboring site.
It demonstrates the interface rather than a generic large-lattice geometry.

The output is `pythonTest/2D/outputs/docs-2d/ed-gcnn.pt` from the repository
root. All 2D CLI relative checkpoint/report paths start under `pythonTest/2D/`.
Training outputs must stay inside that directory and existing outputs are
protected against overwrite; use a new filename for each run.

`log_mse` uses normalized log amplitudes over the full fixed sector, with Adam.
Checkpoint selection uses loss, while raw-vector ED overlaps/energies are
reported separately. ED has dimension/memory guards and is for feasible small
sectors. For larger systems use VMC without requesting an ED reference.

### Minimize energy with VMC

This uses the same ansatz with sampled energy gradients and Adam:

```bash
.venv/bin/python -B pythonTest/2D/train.py \
  --arch GroupEquivariantConv2DNet --Lx 2 --Ly 3 --Ntot 6 --Nmax 2 \
  --hidden-dim 8 --t 1 --U 16.74 --mu 0 --pbc \
  --update-method native --optimizer adam --epochs 20 --lr 0.001 \
  --walkers 16 --steps 8 --initial-burnin 64 --burnin 8 --thin 1 \
  --dtype float64 --device cpu --seed 123 \
  --output outputs/docs-2d/vmc-gcnn.pt
```

For the existing ridge SR + SGD route, replace the update options with
`--update-method sr --optimizer sgd --lr 0.002 --tau 0.01`. For a small model,
`--sr-solver auto --jacobian-method vmap --jacobian-batch-size 32` are available
controls. SR resource guards still apply. `--steps` is retained steps per walker;
burn-in and thinning count local proposals rather than complete lattice sweeps.
These short illustrative sampling settings are not mixing recommendations.

The SR examples above use the default `--sr-regularizer ridge`. The optional
`--sr-regularizer minsr` instead uses a soft spectral pseudoinverse, ignores
`tau`, and rejects a `tau` schedule. Consult `train.py --help` for its spectral
cutoff controls; the ridge equations in this guide do not describe MinSR.

### Select architecture, boundaries and device

Replace `--arch` in either complete command with any of the seven names above.
Use `--width` for ED and `--hidden-dim` for VMC; choose a multiple of four for
Conv2DTransformerNet. Jastrow ignores width. `--hidden-layers` affects only FCNet.

For a cylinder with any of the six non-GCNN families, replace `--pbc` with
`--no-pbc-x --pbc-y`. Use `--no-pbc` for both axes open. `--device auto` selects
CUDA if available, otherwise CPU; `--device cuda:0` requests a specific GPU.
`--dtype float64` and `float32` are supported. To disable TF32 in CUDA runs,
add `--disable-tf32`.

| Setting | ED fitting | Energy-based VMC |
| --- | --- | --- |
| Capacity | `--width 8` | `--hidden-dim 8` |
| Objective/update | `--loss log_mse`, Adam | `--update-method native --optimizer adam` or `sr --optimizer sgd` |
| Common physical options | `--Lx`, `--Ly`, `--Ntot`, `--Nmax`, `--t`, `--U`, `--mu` | Same |
| Boundary options | `--pbc` / `--no-pbc`, axis overrides `--pbc-x` / `--no-pbc-x` and y equivalents | Same |
| Numerical options | `--dtype` defaults to `float64`; also `--device`, `--seed`, `--epochs`, `--lr` | `--dtype` defaults to `float32`; also sampling/SR controls |

Use `--help` on each CLI for the complete option set. See
[Python workflows](python.md) for independent frozen measurements and
[benchmarks](benchmarks.md) for measured accuracy and convergence limits.

## Use the trained state in Julia

Launch `./julia-local` from the repository root, then:

```julia
using BoseHubbardNQS

model = load_nn_2d("pythonTest/2D/outputs/docs-2d/ed-gcnn.pt";
                   device="auto")
sigma = ones(Int, 6)
logpsi = nn_value(model, sigma)
same_logpsi = ansatz_log_amplitude(model, sigma)
configs = hcat(sigma, [0, 2, 1, 1, 1, 1])     # 6 sites × 2 walkers
values = nn_values(model, configs; chunk_size=128)

geometry = (model.metadata["Lx"], model.metadata["Ly"])
symmetries = model.metadata["spatial_symmetries"]
```

To evaluate the VMC-trained state instead, use the same calls with
`"pythonTest/2D/outputs/docs-2d/vmc-gcnn.pt"`. The scalar is a Julia `Float64`
and the batch result a `Vector{Float64}`. Forward precision follows the saved
dtype; converting an output to Float64 does not increase float32 accuracy.

For one image, Julia's `(Lx, Ly)` layout flattens in the saved x-fast order:

```julia
image = ones(Int, 2, 3)
logpsi_image = nn_value(model, vec(image); nmax=2)
values_chunked = nn_values(model, configs; nmax=2, chunk_size=1)
```

A matrix passed to `nn_values` is always a sites-by-walkers batch, not an image.
No per-batch normalization or particle-sector projection is applied.

### Loading and inference keyword choices

| Julia function/keyword | Choices and behavior |
| --- | --- |
| `load_nn_2d(path; device=nothing)` | CPU by default; `"cpu"`, `"auto"`, `"cuda"`, `"cuda:0"` select placement. `auto` means CUDA if available, otherwise CPU. |
| Architecture, size, cutoff, dtype and boundaries | Read from checkpoint metadata. There are no `arch`, `Lx`, `Ly`, `nmax` or `dtype` overrides on this loader. |
| `nn_value(...; nmax=nothing)` | Infer cutoff by default; an explicit integer must match. Also accepted by `ansatz_log_amplitude`. |
| `nn_values(...; nmax=nothing, chunk_size=1024)` | Same cutoff rule; chunk size must be a positive integer. |
| Return handle | `NQS2D` with `.model` (Python object) and `.metadata` (native Julia dictionary). |

Julia paths are relative to the Julia working directory, unlike the Python 2D
CLI convention. The explicit repository-prefixed paths above account for this.
Unavailable explicit CUDA devices raise an error.

### Optional: start the existing Python workflows from Julia

The wrappers run the same Python programs in isolated subprocesses. For example,
this creates a separate small checkpoint with the same geometry as above:

```julia
fit = train_ed_2d(arch="GroupEquivariantConv2DNet",
                  Lx=2, Ly=3, Ntot=6, Nmax=2, width=8,
                  t=1.0, U=16.74, mu=0.0, pbc=true,
                  loss="log_mse", epochs=20, eval_every=10,
                  dtype="float64", device="cpu",
                  output="pythonTest/2D/outputs/docs-2d/julia-ed-gcnn.pt")
trained = load_nn_2d(fit["paths"]["checkpoint"]; device="cpu")
```

`train_vmc_2d` and `evaluate_2d` expose the corresponding VMC and frozen
measurement CLIs. Workflow keywords follow CLI names with hyphens replaced by
underscores: `hidden_dim`, `update_method`, `pbc_x`, `initial_burnin`, etc.
Preserve capitalization for `Lx`, `Ly`, `Ntot`, `Nmax` and `U`. Workflows default
to `device="auto"`, whereas inference loaders default to CPU. Results contain
`"report"`, `"paths"`, `"stdout"` and `"stderr"`. Checkpoints are inference
weights/metadata, not exact optimizer/walker/RNG resume state.

## Reopen the same checkpoint in Python

Start `.venv/bin/python` from the repository root. This evaluates the same
saved state used by the Julia examples:

```python
from pathlib import Path
import torch
from torch.nn import functional as F
from pythonTest.bosehubbard2d.models import load_model

path = Path("pythonTest/2D/outputs/docs-2d/ed-gcnn.pt").resolve()
model_python, metadata = load_model(path, device="cpu")
parameter = next(model_python.parameters())
occupations = torch.ones((1, 6), dtype=torch.long, device=parameter.device)
x = F.one_hot(occupations, num_classes=metadata["Nmax"] + 1).to(parameter.dtype)
with torch.no_grad():
    logpsi_python = model_python(x).cpu().item()
```

Here the direct model input is one-hot `(batch, sites, Nmax+1)`. The Julia
interface accepts occupation counts and performs this encoding internally.
To reopen a checkpoint produced by `train_ed_2d` above, replace `ed-gcnn.pt`
with `julia-ed-gcnn.pt`; the file format is identical. Absolute paths avoid the
Python loader's 2D-relative-path rule. Use the namespaced imports above in a
fresh interpreter rather than mixing them with legacy bare 2D imports.

See [the Julia API](api/julia.md) for all exports and
[batching and CUDA](batching.md) for input validation and device behavior.
