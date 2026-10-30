# Quick VMC benchmarks — October 30, 2026

Eight VMC training runs compare the four architectures supported by `TrainVMCstates.py` at eight and sixteen sites. All completed with finite saved-model energies, using the existing Python implementation at commit `682c4731f676c0aed385afecc9f504c73ec4cfb4`. No Python or Julia source was changed. Julia integration and DMRG were not exercised.

These are short, single-training-seed baselines with common hyperparameters. They measure the behavior of a fixed 100-epoch budget, rather than converged ground-state accuracy or the best achievable performance of each architecture. Full-precision measurements, commands, training logs, and GPU regression logs are preserved in [results.json](benchmarks/2026-10-30/results.json).

## Hardware and protocol

| Setting | Value |
| --- | --- |
| GPUs | Two NVIDIA RTX A6000 cards, approximately 48 GiB each |
| Driver / PyTorch CUDA runtime | 580.178.04 / 12.8 |
| Python / PyTorch | 3.13.3 / 2.11.0+cu128 |
| NumPy / SciPy / Matplotlib | 2.3.3 / 1.17.1 / 3.10.9 |
| CPU threading | `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` |
| Lattice | 1D periodic ring, `L=Ntot=8` or `16`, `Nmax=4` |
| Hamiltonian | `t=1`, `U=2`, `mu=0`; onsite term is `U n(n−1)` |
| Model width | `hidden_dim=32`; ignored by JastrowNet |
| Training | 100 epochs, native energy-gradient update, Adam, `lr=0.001` |
| Sampling per epoch | 64 walkers × 16 retained steps = 1,024 correlated samples; 16 burn-in steps per walker before each epoch |
| Training seed / network chunk size | 123 / 4,096 |
| GPU assignment | FCNet and RBMNet on GPU 0; JastrowNet and Conv1DNet on GPU 1 |

Each GPU ran one training job at a time, with the two GPU queues running concurrently. The cards also serve a workstation; they were not reserved exclusively. Width is held constant where applicable, but parameter counts differ substantially.

Wall time is one end-to-end invocation of the unchanged training CLI, including Python/PyTorch startup, CUDA initialization, training, the trainer's final measurement, and checkpoint saving. It excludes the separate evaluations below and ED reference construction. There were no timing repeats or explicit warm-up runs; these numbers are not isolated kernel timings or steady-state throughput measurements.

## Eight sites: exact sector evaluation

The `Ntot=8`, `Nmax=4` sector has 5,475 basis states. After training, each saved model was evaluated on the entire sector. With `a = exp(logpsi - max(logpsi))` normalized to unit norm, the reported energy is `aᵀ H a`; it has no Monte Carlo sampling error. The ED reference is **−7.4597647030**, or **−0.9324705879 per site**, for this same cutoff and Hamiltonian.

| Architecture | Parameters | Wall time (s) | Exact model energy | Energy/site | Gap to ED | Overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FCNet | 2,401 | 5.37 | −6.693804 | −0.836725 | 0.765961 | 0.926539 |
| JastrowNet | 130 | 8.88 | −6.342133 | −0.792767 | 1.117631 | 0.893763 |
| Conv1DNet | 8,769 | 6.03 | −6.911563 | −0.863945 | 0.548201 | 0.945173 |
| RBMNet | 1,352 | 5.20 | −6.824117 | −0.853015 | 0.635648 | 0.937768 |

Overlap means `|⟨a|psi_ED⟩|`, without squaring. It is computed after sector normalization. The ED reference is specific to `Nmax=4`, whereas the historical supervised-training table in `CLAUDE.md` uses `Nmax=8`.

Conv1DNet has the lowest exact eight-site energy in this short run, with the most parameters. All four models remain above ED after 100 epochs. This is a useful training sanity check, not evidence of convergence.

## Sixteen sites: independent sampling evaluations

The fixed-number sector contains 183,424,355 states, so ED was omitted. Each saved model was evaluated twice from fresh initial walkers, with seeds 456 and 789. Each evaluation used 128 walkers, 4,096 burn-in proposals per walker, and 256 retained steps per walker: 32,768 correlated samples per repeat. The two evaluations reuse the same trained weights; they are not independent training runs.

| Architecture | Parameters | Wall time (s) | Energy, seed 456 | Energy, seed 789 | Mean energy/site | Local-energy SD, seeds 456 / 789 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FCNet | 3,681 | 5.42 | −12.106270 | −12.453026 | −0.767478 | 4.5402 / 4.1736 |
| JastrowNet | 230 | 22.01 | −13.010208 | −13.171515 | −0.818179 | 3.0654 / 3.0725 |
| Conv1DNet | 16,961 | 6.21 | −12.990428 | −13.198971 | −0.818419 | 3.0402 / 2.9219 |
| RBMNet | 2,672 | 5.31 | −12.743452 | −12.858072 | −0.800048 | 3.4175 / 3.4527 |

Mean energy/site is the arithmetic mean of the two total-energy estimates, divided by 16. SD is the population standard deviation of sampled **total local energies**, not a standard error or confidence interval. Samples are correlated; autocorrelation times and effective sample sizes were not estimated. The burn-in length was increased from an initial 256-step evaluation because the estimates changed noticeably. Both the preliminary and longer evaluations are retained in the raw record. Longer burn-in alone does not establish equilibration.

JastrowNet and Conv1DNet give similar sixteen-site estimates within the observed sampling variation. The difference between their mean total energies is about 0.004, much smaller than their repeat-to-repeat variation, so these results do not separate their accuracy. No sixteen-site ground-state reference or rigorous gap is claimed. The short training budget and remaining evaluation variability also preclude a firm ranking of the other models.

## Reproduce training

Run from the repository root in the environment described above. The following two shell blocks can run in separate terminals. They use distinct checkpoints and do not overwrite the pre-existing `data/bose_hubbard_vmc_nqs.pt`.

GPU 0:

```bash
mkdir -p data/benchmarks/2026-10-30
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bhnqs-mpl
for L in 8 16; do
  for arch in FCNet RBMNet; do
    .venv/bin/python -u pythonTest/TrainVMCstates.py \
      --arch "$arch" --L "$L" --Ntot "$L" --Nmax 4 --hidden_dim 32 \
      --epochs 100 --n_walkers 64 --n_steps 16 --burnin_steps 16 \
      --eval_every 25 --seed 123 --t 1 --U 2 --mu 0 --lr 0.001 \
      --update_method native --optimizer adam --device cuda:0 \
      --evaluation_batch_size 4096 \
      --output "data/benchmarks/2026-10-30/L${L}_${arch}.pt"
  done
done
```

GPU 1:

```bash
mkdir -p data/benchmarks/2026-10-30
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bhnqs-mpl
for L in 8 16; do
  for arch in JastrowNet Conv1DNet; do
    .venv/bin/python -u pythonTest/TrainVMCstates.py \
      --arch "$arch" --L "$L" --Ntot "$L" --Nmax 4 --hidden_dim 32 \
      --epochs 100 --n_walkers 64 --n_steps 16 --burnin_steps 16 \
      --eval_every 25 --seed 123 --t 1 --U 2 --mu 0 --lr 0.001 \
      --update_method native --optimizer adam --device cuda:1 \
      --evaluation_batch_size 4096 \
      --output "data/benchmarks/2026-10-30/L${L}_${arch}.pt"
  done
done
```

The trained checkpoints and per-run files remain locally under `data/benchmarks/2026-10-30/`, which is ignored by Git. The documentation's `results.json` preserves measurements and logs without requiring those binary checkpoints. A repeated training command overwrites the corresponding benchmark checkpoint, so use a new directory if retaining multiple trials. Exact floating-point reproduction across environments is not guaranteed by the seed.

## Reproduce saved-model evaluation

The following Python snippet can be run from a temporary file with `.venv/bin/python`, from the repository root. It uses the existing model and sampling functions. Set `L`, `arch`, and `device` for the desired row; for all four rows use the GPU assignment above. The eight-site branch constructs ED; do not extend it to sixteen sites.

```python
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd() / "pythonTest"))
from TrainVMCstates import ARCH_BUILDERS
from vmc_utils_gpu import generate_initial_state, Obtain_Sampling_batch_gpu

L, arch, device = 16, "FCNet", torch.device("cuda:0")
torch.set_num_threads(1)
torch.cuda.set_device(device)
path = Path(f"data/benchmarks/2026-10-30/L{L}_{arch}.pt")
ckpt = torch.load(path, map_location="cpu", weights_only=True)
model = ARCH_BUILDERS[arch](L, 5, 32).to(device).eval()
model.load_state_dict(ckpt["state_dict"])

if L == 8:
    from BoseHubbardED import exact_ground_state, basis_to_one_hot

    E0, psi0, basis, H = exact_ground_state(8, 8, 4, 1., 2., 0., True)
    x = torch.from_numpy(basis_to_one_hot(basis, 8, 4)).to(device)
    with torch.no_grad():
        logs = torch.cat([model(b) for b in x.split(4096)]).double()
        a = (logs - logs.max()).exp().cpu().numpy()
    a /= np.linalg.norm(a)
    energy = float(a @ H @ a)
    print("energy, energy/site, gap, overlap:",
          energy, energy / L, energy - E0, abs(a @ psi0))
else:
    assert L == 16
    means = []
    for seed in (456, 789):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        states = [generate_initial_state(L, L, 4) for _ in range(128)]
        settings = dict(psi=model, L=L, Nmax=4, t=1., U=2., mu=0.,
                        pbc=True, evaluation_batch_size=4096)
        _, _, states, _ = Obtain_Sampling_batch_gpu(
            initial_states=states, n_steps=4096, burnin=True, **settings)
        logs, energies, _, _ = Obtain_Sampling_batch_gpu(
            initial_states=states, n_steps=256, **settings)
        mean = energies.mean().item()
        means.append(mean)
        print(seed, "energy:", mean,
              "local-energy SD:", energies.std(unbiased=False).item())
        del logs, energies
    print("mean energy/site:", np.mean(means) / L)
```

## Validation and scope

The existing four-test regression suite passed on CPU during the initial project review and on **both GPUs** during this benchmark update:

```bash
.venv/bin/python pythonTest/vmc_gpu_test.py --self-test --device cpu
.venv/bin/python pythonTest/vmc_gpu_test.py --self-test --device cuda:0
.venv/bin/python pythonTest/vmc_gpu_test.py --self-test --device cuda:1
```

The suite checks local energies against ED and scalar math, proposal conservation/symmetry, sampling/cache/gradient behavior, and SR/native update calculations. The architecture comparisons above use the native update only; passing SR regression tests is not a larger-system SR performance benchmark.

Further accuracy studies should use longer training, multiple training seeds, a validated DMRG reference at sixteen sites, and sampling diagnostics before reporting error bars or architecture rankings. These are follow-up experiments, not part of the quick baseline reported here.
