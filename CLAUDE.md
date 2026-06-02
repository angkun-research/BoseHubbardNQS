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
| `pythonTest/leakage_analysis.py` | Measures `P(N')` across all `(Nmax+1)^L` occupation states; reports leakage fraction, sectored overlap, and plots distribution |
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

## Leakage Analysis

`leakage_analysis.py` evaluates a trained NQS on **all** `(Nmax+1)^L` occupation states (no fixed-particle-number constraint) and measures how amplitude leaks outside the training sector.

### Key functions

| Function | What it does |
|----------|-------------|
| `generate_all_basis(L, Nmax)` | Generates all Fock states ordered by increasing `N'`, grouped by sector |
| `evaluate_logpsi_chunks(model, ...)` | Forward pass in 4096-batch chunks (memory-managed) |
| `compute_sector_probabilities(model, ...)` | `P(N') = Σ_{s: Σnᵢ=N'} exp(2·logψ(s) - Z)` where `Z = logsumexp(2·logψ)` over all states |
| `compute_leakage_fraction(model, ...)` | `leakage = 1 - P(N_train)` |
| `compute_sectored_overlap(model, ...)` | Fidelity per sector vs ED ground state |
| `plot_leakage_distribution(...)` | Bar chart of `P(N')` with training sector highlighted (log scale) |

### CLI usage

```bash
# Train fresh and analyze
.venv/bin/python pythonTest/leakage_analysis.py --architecture FCNet --L 8 --Nmax 8 --train_Ntot 8

# Analyze a saved checkpoint (no overlap)
.venv/bin/python pythonTest/leakage_analysis.py --model data/FCNet_net.pt --architecture FCNet --no_overlap --output fig_leakage_fcnet.pdf

# Compare architectures
for arch in FCNet JastrowNet Conv1DNet RBMNet; do
    .venv/bin/python pythonTest/leakage_analysis.py --model data/${arch}_net.pt --architecture $arch \
        --L 8 --Nmax 8 --train_Ntot 8 --output fig_leakage_${arch}.pdf
done
```

### Interpreting leakage

- **Low leakage** (< 1%) means the model's capacity is concentrated in the training sector — the unconstrained architecture naturally respects the U(1) symmetry despite no explicit constraint.
- **High leakage** means the model wastes amplitude on wrong sectors. This does not necessarily mean poor physics: if the model still achieves high fidelity on the training sector, leakage is harmless capacity waste.
- The leakage pattern across sectors is itself a diagnostic (Benedetti et al.): organized leakage (e.g. symmetric peaks) vs random suggests how the ansatz distributes capacity.
- For `L=8, Nmax=8` the full basis is `9^8 ≈ 43M` states — ~1.2 GB for float32 arrays. The script uses 4096-batch evaluation and `int8` state storage to manage memory.

## Related Literature

### Symmetry, Particle Number Conservation & Leakage in NQS

- **[Denis & Carleo, Quantum 2025 (arXiv:2404.07869)](https://quantum-journal.org/papers/q-2025-06-17-1772/pdf/)** — Neural backflow Jastrow ansatz for the 2D Bose-Hubbard model. Explicitly addresses U(1) symmetry breaking, superfluidity, and boson condensation. Key result: standard architectures (RBM, FFN) leak amplitude into wrong particle-number sectors; the backflow transformation builds symmetry into the ansatz. Best variational energies reported for 2D Bose-Hubbard up to 20×20 lattices.

- **[Hul et al. (arXiv:2605.07779, May 2026)](https://arxiv.org/abs/2605.07779)** — "Neural network quantum states in the grand canonical ensemble." Architecture that represents symmetric bosonic wavefunctions in Fock space with variable particle number. Trains with a chemical potential to control <N>. The inverse approach to studying leakage: embraces it as the feature, using it to access observables like condensate fractions and radial density profiles from first principles.

- **[Choo et al., PRL 121, 167204 (2018)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.167204)** — Symmetries and many-body excitations with NQS. Shows how to construct U(1)-symmetric NQS via symmetry-projected states: train unconstrained, then apply particle-number projection operator to extract the correct sector.

- **[Benedetti et al., PRX Quantum 5, 030314 (2024)](https://link.aps.org/doi/10.1103/PRXQuantum.5.030314)** — Symmetry breaking in geometric quantum machine learning. Even equivariant networks can develop symmetry-breaking behavior during training due to optimization noise.

- **[Lange et al., Quantum Sci. Technol. 9, 040501 (2024)](https://iopscience.iop.org/article/10.1088/2058-9565/ad7168)** — Comprehensive review of NQS architectures and applications. Section 2.5.1 discusses U(1) symmetry in autoregressive NQS; covers symmetry projection as a design choice.

- **[Medvidović & Robledo Moreno, Eur. Phys. J. Plus 139, 631 (2024) (arXiv:2402.11014)](https://arxiv.org/abs/2402.11014)** — Review of NQS for many-body physics. Covers variational Monte Carlo approaches, time evolution, and overlap optimization. Discusses symmetry projection for recovering wave function symmetries.

- **[Benedetti et al. (arXiv:2401.10293)](https://link.aps.org/doi/10.1103/PRXQuantum.5.030314)** — Symmetry-organised complexity: a framework for measuring how an ansatz distributes capacity across symmetry sectors. The leakage pattern itself is a diagnostic — organized leakage across sectors is a feature, not a bug.

### Fixing Ntot — autoregressive as variational baseline

Writing `ψ(n₁,...,nₗ) = ∏ₖ a(nₖ|n₁...nₖ₋₁)` is just **Bayes' rule** — a tautology for any nonzero function. The "autoregressive" decomposition imposes no physics; different orderings produce different conditionals but the same final wavefunction. For fixed Ntot, a "gauge block" masks amplitudes so that `nₖ ∈ [0, min(Nmax, remaining)]` at each step, guaranteeing zero leakage. This is a hand-built constraint in the forward pass, not an emergent property of the network.

The only use of autoregressive NQS is as a **variational baseline**: if unconstrained networks (FCNet, RBMNet, etc.) achieve the same energy on the same physics, then leakage costs nothing — the network learns the correct sector anyway. If the autoregressive model beats it, the unconstrained model is wasting capacity on wrong sectors.

### Bosonic NQS Architectures

- **[Bosonic NQS review (Phys. Rev. C 109, 034004)](https://link.aps.org/accepted/10.1103/PhysRevC.109.034004)** — Neural network solutions for bosonic quantum systems in 1D. Benchmarks symmetric vs. unconstrained architectures.

- **[arXiv:2101.07243]** — General framework for gauge-invariant autoregressive NQS (applied to U(1) lattice gauge theory, toric codes, anyonic chains). The gauge block zeros out symmetry-violating configurations at each autoregressive step. For the global U(1) Ntot constraint, the same framework applies trivially (run remaining budget as a hand-built mask). Useful as a variational baseline only.

- **[Choo et al., PRL 121, 167204 (2018)]** — Symmetry projection: train unconstrained, then apply `|ψ_projected⟩ = (1/2π) ∫ e^{-iαN̂} dα |ψ_unconstrained⟩` to extract the correct sector.

- **[arXiv:2405.15488]** — Unsupervised deep neural network approach for bosonic systems. Adaptive momentum optimizer for faster convergence on Bose-Hubbard.

### Methodology

- **Carleo & Troyer, Science 355, 602 (2017)** — Original NQS paper. Neural-network ansatz for quantum many-body states via variational Monte Carlo.

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
