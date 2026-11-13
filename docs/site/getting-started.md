# Getting started

Run the commands below from the repository root. Start on CPU with a small
system; enable CUDA after confirming the selected Python environment supports it.

## Python environment

```bash
git clone https://github.com/angkun-research/BoseHubbardNQS.git
cd BoseHubbardNQS
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

For CUDA, install a PyTorch build compatible with your driver using the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/).
To reproduce the tested Linux/Python 3.13 CUDA 12.8 environment, install
`docs/requirements-cuda.txt` into a clean virtual environment. It pins the
required packages and excludes unused image/audio packages. Installing into an
existing environment does not remove leftover dependencies from older installs.

```bash
.venv/bin/python -m pip install -r docs/requirements-cuda.txt
```

Check the actual interpreter that Julia will use:

```bash
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

The tested CPU CI dependency set is in `docs/requirements-ci.txt`. The website
has a separate environment in `docs/requirements-docs.txt`.

## Produce a small 1D checkpoint

```bash
.venv/bin/python pythonTest/TrainVMCstates.py \
  --arch FCNet --L 4 --Ntot 4 --Nmax 4 --epochs 20 \
  --device cpu --ed-reference --output data/example-1d.pt
```

This is a small runnable demonstration, not a converged benchmark. Use a new
output path to preserve previous results.

## Produce a small 2D checkpoint

```bash
.venv/bin/python -B pythonTest/2D/train_ed.py \
  --arch GroupEquivariantConv2DNet --Lx 2 --Ly 3 --width 8 \
  --epochs 20 --device cpu --dtype float64 \
  --output outputs/docs-example/ed-2x3.pt
```

The 2D output path is relative to `pythonTest/2D/`. The resulting file is
`pythonTest/2D/outputs/docs-example/ed-2x3.pt` when viewed from the repository root.
Twenty updates illustrate the workflow and do not reproduce the long-run
[accuracy results](benchmarks.md).

## Julia environment

The `julia-local` launcher selects this checkout's `.venv/bin/python` for
PythonCall. Install the Julia dependencies once:

```bash
./julia-local -e 'using Pkg; Pkg.instantiate()'
```

Then evaluate the generated checkpoint:

```julia
using BoseHubbardNQS

model = load_nn("data/example-1d.pt")
sigma = [1, 1, 1, 1]
logpsi = nn_value(model, sigma)
```

See [the Julia guide](julia.md) for 2D, batches and device selection. Set
`JULIA_PYTHONCALL_EXE` before loading PythonCall if you use another environment;
the interpreter must contain the required Python packages.

## Switch between Python and Julia

These are two language environments connected to **the same PyTorch runtime**.
Python trains and saves a checkpoint; Julia calls that Python interpreter
through PythonCall and loads the same file. A checkpoint produced by the Julia
2D workflow wrappers can also be reopened from Python. No weight-format
conversion or second training run is needed.

| Task | Python | Julia |
| --- | --- | --- |
| 1D log-MSE fitting | Supervised helper recipe in the [1D guide](model-1d.md#fit-a-small-ed-state-with-log-mse-and-adam) | Load its checkpoint with `load_nn` |
| 1D energy-based VMC | `TrainVMCstates.py` | Load its checkpoint with `load_nn`; no dedicated 1D training wrapper |
| 2D log-MSE / VMC / measurements | `train_ed.py` / `train.py` / `evaluate.py` | `train_ed_2d` / `train_vmc_2d` / `evaluate_2d` delegate to those Python programs |
| Scalar / batch inference | [1D reopening example](model-1d.md#reopen-the-same-checkpoint-in-python) / [2D reopening example](model-2d.md#reopen-the-same-checkpoint-in-python) | `nn_value` / `nn_values` for either dimension |

Keep the working directory at the repository root and use `.venv/bin/python`
for Python and `./julia-local` for Julia. This avoids accidentally using a
system Python with a different PyTorch/CUDA installation. Install Julia
packages in the Julia project and Python packages in `.venv`; Julia inference
uses PyTorch's device support, not a separate Julia CUDA model.

If you prefer a custom interpreter, select it **before starting Julia**:

```bash
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE="$PWD/.venv/bin/python" julia --project=.
```

The launcher sets these variables for you. To confirm the interpreter actually
used by a Julia session:

```julia
using PythonCall
pyconvert(String, pyimport("sys").executable)
```

For end-to-end training and loading commands, use the [1D guide](model-1d.md)
and [2D guide](model-2d.md). Both distinguish log-MSE from energy-based VMC and
state the keyword, path and output conventions.
