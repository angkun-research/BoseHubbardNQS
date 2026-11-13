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
