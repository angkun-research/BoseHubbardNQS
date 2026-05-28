# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural network quantum states (NQS) for the Bose-Hubbard model. The project has two parallel implementations:

- **Julia** (`src/`): A `BoseHubbardNQS` module wrapping a Python neural network via `PythonCall.jl`. Loads saved PyTorch models and evaluates wavefunction amplitudes for spin configurations.
- **Python** (`pythonTest/`): Full training pipeline — exact diagonalization (ED) of the Bose-Hubbard Hamiltonian, multiple NN architectures (FCNet, JastrowNet, Conv1DNet, RBMNet), and training/evaluation utilities.

Both target the same physics: finding ground states of `H = -t sum_<ij> (b_i^dag b_j + h.c.) + sum_i [U n_i (n_i-1) - mu n_i]`.

## Key Files

| File | Role |
|------|------|
| `src/BoseHubbardNQS.jl` | Julia module entry point; exports `load_nn`, `nn_value`, `ansatz_amplitude` |
| `src/spin_nn.jl` | Julia-Python bridge: loads `.pt` models and calls `predict_scalar` |
| `src/python/simple_spin_nn.py` | Simple SpinNet definition, train/save/load/predict functions (Julia bridge target) |
| `pythonTest/BoseHubbardED.py` | Exact diagonalization: basis generation, Hamiltonian building, `eigsh` solver, observables (density profile, one-body density matrix) |
| `pythonTest/NeuralNetworkClass.py` | NN architectures: FCNet, JastrowNet, Conv1DNet, RBMNet |
| `pythonTest/TrainEDstates.py` | Training loop: creates ED dataset, trains model, evaluates overlap |
| `pythonTest/utils.py` | GPU device selection via nvidia-smi |
| `example/BoseHubbardITensor.jl` | ITensor-based DMRG reference for validation |
| `example/spin_nn_test.jl` | Julia-side usage example for saved spin net |

## Running the Code

### Julia
```bash
./julia-local -e 'using BoseHubbardNQS; println(load_nn())'
```
The `julia-local` script sets up the Python virtualenv before launching Julia. Run with `--project=.` to use the local Manifest.toml.

### Python
```bash
.venv/bin/python pythonTest/TrainEDstates.py
.venv/bin/python pythonTest/BoseHubbardED.py   # ED only, no training
```

### Dependencies
- **Julia**: ITensorMPS, ITensors, PythonCall (see `Project.toml`)
- **Python**: torch, numpy, scipy, matplotlib (see `requirements.txt`)
- Install Python deps: `.venv/bin/pip install -r requirements.txt`

## Architecture Notes

- Basis states use one-hot encoding: shape `(batch, n_sites, nmax+1)`.
- The Julia bridge (`spin_nn.jl`) lazy-loads the Python module via `PythonCall` and caches it in `_pynet`.
- ED uses `scipy.sparse.linalg.eigsh` for the lowest eigenpair; Hilbert space dimension grows combinatorially with L, Ntot, Nmax.
- DMRG reference (`example/BoseHubbardITensor.jl`) produces the same ground energy (-7.45979) as ED for L=8, Ntot=8, Nmax=8, t=1, U=2.

## Training & Evaluation Results

Physics: L=8, Ntot=8, Nmax=8, t=1.0, U=2.0, lr=1e-3, epochs=10000, target=logabs.

| Metric | FCNet | JastrowNet | Conv1DNet | RBMNet |
|--------|--:|:--:|-:|-:|
| MSE | 6.398577e-04 | 5.374164e-03 | 2.953705e-04 | 4.151604e-03 |
| Overlap | 9.994203e-01 | 9.972393e-01 | 9.999396e-01 | 9.973432e-01 |
| P([1,1,1,1,1,1,1,1]) | 1.382670e-01 | 1.341546e-01 | 1.356353e-01 | 1.354035e-01 |
| P([2,1,1,1,1,1,1,1]) | 2.027709e-01 | 1.773731e-01 | 3.148663e-01 | 8.556325e-01 |
| P([0,1,1,1,1,1,1,1]) | 4.303387e-03 | 4.623400e-03 | 1.645630e-03 | 9.668646e-04 |

## Log-psi Loss Function, Overlap & Probability

When `target="logabs"`, the model outputs `logpsi(s) ≈ log|ψ(s)|` for each basis state `s`. The full-basis normalization and downstream metrics use `logsumexp` consistently:

### Loss function

The true ground state is `φ_true`, with `log_phi_true[s] = log|φ_true[s]| + C_true` (some unknown constant `C_true`). The model outputs `logpsi[s] = log|ψ(s)| + C_model`. Both are defined up to an additive constant, so we subtract the normalization constant to compare them:

```
normalize(v) = v - 0.5 * logsumexp(2*v)
```

The normalization ensures `∑_s exp(2 * normalized_logpsi[s]) = 1`. The loss is:

```
L = mean( (normalize(logpsi) - normalize(log_phi_true))^2 )
```

This is **not** `|ψ(s) - φ(s)|²` — it is `|exp(log|ψ|/Z) - exp(log|φ|/Z_true)|²` after exponentiation and L2 normalization. The `*2` inside `logsumexp` converts log-amplitude to log-probability (`|ψ|² = exp(2*log|ψ|)`), and `/2` converts back.

### Overlap (fidelity)

```
Overlap = |∑_s exp(normalize(logpsi)[s]) · exp(normalize(log_phi_true)[s])|
         / ( ||exp(normalize(logpsi))|| · ||exp(normalize(log_phi_true))|| )
```

Since both are L2-normalized, this equals `|⟨ψ̂|φ⟩|` where `ψ̂` is the model's L2-normalized wavefunction. Values closer to 1 indicate a better approximation.

### Probability of a basis state

```
P(s) = |ψ(s)|² / ∑_i |ψ(i)|²
     = exp(2*logpsi(s) - logsumexp_i(2*logpsi(i)))
```

In code, compute `logsumexp_i(2*logpsi(i))` over the **entire** training basis once, then for each test state `s`:

```python
prob = exp(2 * logpsi(s) - full_logsumexp)
```

**Do not** use `logsumexp` computed over a different Hilbert sector (e.g. training on Ntot=8 and evaluating a state with Ntot≠8). The model only learned the training sector, so the normalization constant is meaningless outside it.
