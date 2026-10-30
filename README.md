# BoseHubbardNQS

Neural-network quantum states for the one-dimensional Bose–Hubbard model, with Python training and a Julia inference wrapper.

The Python pipeline supports exact-diagonalization (ED) supervised training, particle-number leakage analysis, and batched variational Monte Carlo (VMC) on CPU or GPU. The VMC trainer supports `FCNet`, `JastrowNet`, `Conv1DNet`, and `RBMNet`, with either stochastic reconfiguration (SR) or a standard energy-gradient update. Additional pooling and transformer architectures have separate experimental ED scripts.

Run all commands below from the repository root. See [the VMC benchmark report](docs/VMC_BENCHMARKS.md) for measured eight- and sixteen-site results, hardware, commands, and limitations.

## Physics and conventions

The implemented Hamiltonian is

\[
H=-t\sum_{\langle i,j\rangle}(b_i^\dagger b_j+b_j^\dagger b_i)
  +\sum_i\left[U n_i(n_i-1)-\mu n_i\right].
\]

Here `U` is the coefficient of `n(n−1)`. If your convention is `(U_model/2) n(n−1)`, pass `U = U_model/2`; for example, `--U 2` corresponds to `U_model = 4`. `Nmax` is the inclusive occupation cutoff, so each site has `Nmax + 1` levels. `Ntot` fixes the total particle number. Periodic boundary conditions are the VMC default; use `--no-pbc` for open boundaries.

For `target="logabs"` and VMC checkpoints, the model returns an unnormalized real log amplitude, not a probability. Within a chosen basis, `logZ = logsumexp(2 * logpsi)` and `P(s) = exp(2 * logpsi(s) - logZ)`. A normalization computed in one particle-number sector cannot be used as a global normalization across other sectors. VMC hopping proposals preserve `Ntot`; they do not demonstrate that the unconstrained network has zero amplitude outside that sector.

## Environment

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
mkdir -p data
```

The Python dependencies are PyTorch, NumPy, SciPy, and Matplotlib. GPU use requires a CUDA-capable PyTorch installation compatible with the local driver. Check availability with:

```bash
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Dependencies are currently unpinned. The benchmark report records the environment used for its results.

## VMC training

Start with a small system and an ED reference:

```bash
.venv/bin/python pythonTest/TrainVMCstates.py \
  --arch FCNet --L 4 --Ntot 4 --Nmax 4 \
  --epochs 100 --device cuda:0 --ed-reference \
  --output data/vmc_L4_FCNet.pt
```

This uses the trainer defaults: SR with SGD, `hidden_dim=64`, `lr=0.005`, `tau=0.001`, 64 walkers, 32 sampling steps, and 32 burn-in steps per epoch. `--device cpu` runs on CPU; `--device auto` selects a GPU using the utility's free-memory threshold or falls back to CPU.

For larger systems, use the standard energy-gradient update with Adam to avoid a dense SR solve:

```bash
.venv/bin/python pythonTest/TrainVMCstates.py \
  --arch Conv1DNet --L 16 --Ntot 16 --Nmax 4 --hidden_dim 32 \
  --epochs 100 --n_walkers 64 --n_steps 16 --burnin_steps 16 \
  --update_method native --optimizer adam --lr 0.001 \
  --evaluation_batch_size 4096 --seed 123 --device cuda:1 \
  --output data/vmc_L16_Conv1DNet.pt
```

Use `--arch FCNet`, `JastrowNet`, `Conv1DNet`, or `RBMNet`. `hidden_dim` sets the dense/convolutional width or RBM hidden-unit count; JastrowNet ignores it. `--optimizer` accepts `sgd` or `adam` for either update method; defaults are SGD for SR and Adam for `native`. `--native_opt` is an alias for `--optimizer` in this training script. See all options with:

```bash
.venv/bin/python pythonTest/TrainVMCstates.py --help
```

`--evaluation_batch_size` bounds individual network forward/Jacobian batches. It does not bound all retained samples, the complete autograd graph, or the dense parameter-space SR matrix. SR uses storage proportional to the square of the number of parameters. The ED implementation allocates a dense Hamiltonian before calling a sparse eigensolver; omit `--ed-reference` for large systems. The training script does not run ED unless requested.

The trainer saves an inference checkpoint after its final update and a fresh energy measurement. It contains architecture dimensions, weights, settings, energy history, final sampled energy and local-energy standard deviation, and an ED energy when requested. The reported standard deviation is the spread of local energies, not an error bar on the mean. Checkpoints do not include optimizer or walker state for exact training resumption. Use distinct output paths to retain separate runs; the default is `data/bose_hubbard_vmc_nqs.pt`.

Two GPUs can run independent jobs by assigning `--device cuda:0` and `--device cuda:1` with different output paths. The trainer runs each model on one device; it does not distribute one training run across GPUs.

## ED-supervised training and analysis

```bash
.venv/bin/python pythonTest/TrainEDstates.py
```

Edit the configuration in the script's `__main__` block before running. The current checked-in settings are `L=3`, `Ntot=3`, `Nmax=3`, `t=1`, `U_model=3.3578`, `U=U_model/2`, `mu=0`, periodic boundaries, `target="logabs"`, `hidden_dim=128`, `FCNet`, and 10,000 epochs. It saves `data/bose_hubbard_nqs.pt`. The historical eight-site results in [CLAUDE.md](CLAUDE.md) use different settings and are not the current defaults or the new VMC benchmark.

Other research entry points:

| Script | Purpose |
| --- | --- |
| [BoseHubbardED.py](pythonTest/BoseHubbardED.py) | Fixed-sector basis, Hamiltonian, ground state, and observables |
| [TrainEDstatesFull.py](pythonTest/TrainEDstatesFull.py) | Full-Fock-space training with outside-sector suppression |
| [TrainEDstates3Sector.py](pythonTest/TrainEDstates3Sector.py) | One model trained on three adjacent particle-number sectors |
| [cross_sector_overlap.py](pythonTest/cross_sector_overlap.py) | Separate sector models and inter-sector overlap analysis |
| [leakage_analysis.py](pythonTest/leakage_analysis.py) | Full-space sector probabilities, leakage, and optional particle-number filters |
| [SystemSizeInvTest.py](pythonTest/SystemSizeInvTest.py) | Pooling-network ED experiment |
| [OtherNNtests.py](pythonTest/OtherNNtests.py) | Transformer-based ED experiments |

Full-space analysis enumerates `(Nmax+1)^L` configurations and becomes expensive quickly. Keep these experiments small even when VMC training at a larger `L` is practical.

## Regression checks

```bash
.venv/bin/python pythonTest/vmc_gpu_test.py --self-test --device cpu
.venv/bin/python pythonTest/vmc_gpu_test.py --self-test --device cuda:0
```

The four regression tests cover local energies against ED and the scalar implementation, proposal invariants and symmetry, sampling/cache/gradient behavior, and SR/native update calculations. `vmc_gpu_test.py` also has a training mode; it computes ED by default, so pass `--skip-ed` for larger systems. For saving checkpoints, use `TrainVMCstates.py`.

## Julia inference integration

The existing Julia wrapper and ITensor reference examples have not been changed or revalidated as part of the October 30, 2026 documentation and Python benchmark work. Python benchmark results do not establish Julia compatibility.

The [julia-local](julia-local) launcher sets `JULIA_CONDAPKG_BACKEND=Null`, points `JULIA_PYTHONCALL_EXE` at this repository's `.venv/bin/python`, and launches Julia with `--project=.`. From the repository root:

```bash
./julia-local -e 'using Pkg; Pkg.instantiate()'
```

The module exports `load_nn`, `nn_value`, and `ansatz_log_amplitude`. A metadata-based load avoids hardcoding the training dimensions:

```julia
using BoseHubbardNQS

model = load_nn("data/vmc_L4_FCNet.pt"; map_location="cpu")
sigma = [1, 1, 1, 1]
println(nn_value(model, sigma))
println(ansatz_log_amplitude(model, sigma))
```

This illustrates the existing API; it was not executed during this documentation update. Use a checkpoint whose site count matches `length(sigma)`. If supplying explicit architecture or dimension arguments, they must match the checkpoint. CPU inference is the documented path.

The existing example can be launched with:

```bash
./julia-local example/Bose_Hubbard_NQS_test.jl
```

That example explicitly expects `data/bose_hubbard_nqs.pt` trained as an eight-site FCNet with `Nmax=8` and `hidden_dim=128`. It does not match the current three-site ED training defaults. [BoseHubbardITensor.jl](example/BoseHubbardITensor.jl) provides an existing DMRG reference; it was not used for the new benchmarks.

To develop the Julia package from another project, add this checkout with `Pkg.develop(path="/absolute/path/to/BoseHubbardNQS")` in that project's Julia environment and configure PythonCall to use this checkout's Python environment before importing the package. The `julia-local` launcher's `--project=.` resolves relative to the working directory.
