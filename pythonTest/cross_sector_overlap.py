#!/usr/bin/env python3
"""
Cross-sector overlap matrix for Neural Quantum States.

Trains independent NQS models on neighboring particle-number sectors
(e.g. N-1, N, N+1 for unit filling) and computes a 3x3 overlap matrix.
Each entry (i,j) evaluates both models within the ket's sector only,
with independent L2 normalization within that sector.

Usage:
    python cross_sector_overlap.py --arch FCNet --epochs 10000
    python cross_sector_overlap.py --arch Conv1DNet --Nmax 8 --lr 1e-3
"""

import argparse

import numpy as np
import torch

from scipy.special import logsumexp

from NeuralNetworkClass import FCNet, JastrowNet, Conv1DNet, RBMNet
from BoseHubbardED import generate_basis, basis_to_one_hot
from TrainEDstates import make_ed_dataset, train_model_on_ed
from utils import pick_best_device

# ── Architecture builders ──────────────────────────────────────────────────

ARCH_BUILDERS = {
    "FCNet":      lambda L, Ch, hd: FCNet(n_sites=L, channels=Ch, hidden_dim=hd),
    "JastrowNet": lambda L, Ch, hd: JastrowNet(n_sites=L, channels=Ch),
    "Conv1DNet":  lambda L, Ch, hd: Conv1DNet(n_sites=L, channels=Ch, hidden_dim=hd),
    "RBMNet":     lambda L, Ch, hd: RBMNet(n_sites=L, channels=Ch, n_hidden=hd),
}

# ── Evaluation (log-space, for numerical stability) ────────────────────────

@torch.no_grad()
def evaluate_logpsi(model, basis_states, device, batch_size=4096, global_Nmax=8):
    """Evaluate a model on basis states and return log-amplitudes (log|psi|).

    Args:
        model: trained PyTorch nn.Module (kept in eval mode)
        basis_states: list of occupation tuples or np.ndarray (N, L)
        device: torch device
        batch_size: forward pass batch size
        global_Nmax: global occupancy cutoff (channels = Nmax + 1)

    Returns:
        logpsi: np.ndarray shape (N,) float32, log|psi(s)|
    """
    model.eval()
    N = len(basis_states)
    L = len(basis_states[0]) if isinstance(basis_states[0], (list, tuple)) else basis_states.shape[1]
    channels = global_Nmax + 1

    # Convert list of tuples to numpy array for indexing
    if isinstance(basis_states[0], (list, tuple)):
        states_arr = np.array(basis_states, dtype=np.int8)
    else:
        states_arr = basis_states

    logpsi = np.zeros(N, dtype=np.float32)
    buf = np.empty((batch_size, L, channels), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bs = end - start
        batch = states_arr[start:end]

        # One-hot encode into buffer (in-place)
        for s in range(L):
            buf[:, s, batch[:, s]] = 1.0
        x_tensor = torch.tensor(buf[:bs], dtype=torch.float32, device=device)

        logpsi[start:end] = model(x_tensor).cpu().numpy().astype(np.float32)

    return logpsi

# ── Overlap computation (log-space, for numerical stability) ──────────────

def compute_overlap_matrix(models, sector_list, all_basis_by_sector, device, batch_size=4096, global_Nmax=8):
    """Compute the K×K inter-sector overlap matrix.

    Entry (i, j) = <Ψᵢ|Ψ⵽> evaluated within the ket's sector (sector N⵽).
    Each model's logpsi is normalized using its own training sector Z:
        logpsi_norm_i(s) = logpsi_raw_i(s) - 0.5 * logsumexp(2*logpsi_raw_i over sector i)
    This constant shift is applied regardless of which sector the model is evaluated on.

    Args:
        models: list of K trained PyTorch models
        sector_list: list of K target Ntot values (also training sector indices)
        all_basis_by_sector: dict {Ntot: (basis_states, basis_array)}
        device: torch device
        batch_size: forward pass batch size
        global_Nmax: global occupancy cutoff (channels = Nmax + 1)

    Returns:
        overlap: np.ndarray shape (K, K) float64
    """
    K = len(models)

    # Evaluate all models on all sectors
    logpsi_raw = {}  # logpsi_raw[i][j] = log|psi_i| evaluated in sector j
    for i in range(K):
        logpsi_raw[i] = {}
        for j, N_sect in enumerate(sector_list):
            basis_states = all_basis_by_sector[N_sect][0]
            logpsi_raw[i][j] = evaluate_logpsi(models[i], basis_states, device, batch_size, global_Nmax=global_Nmax)

    # Compute each model's normalization constant from its training sector:
    #   Z[i] = logsumexp(2 * logpsi_raw[i] over training sector i)
    # Use explicit shift to avoid overflow in 2*logpsi before logsumexp shifts it
    Z = []
    for i in range(K):
        lp = logpsi_raw[i][i]
        m = 0.0 #np.max(2.0 * lp)
        Z.append(float(np.log(np.sum(np.exp(2.0 * lp - m))) + m))

    # Compute overlap(i,j) = |<psi_i_j|psi_j_j>| / (||psi_i_j|| * ||psi_j_j||)
    # where each vector is normalized using its own training sector Z.
    # psi_i_j(s) = exp(logpsi_raw[i][j](s) - 0.5 * Z[i])
    overlap = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            # Logpsi of model i normalized by its own training sector Z[i]
            logpsi_i = logpsi_raw[i][j] - 0.5 * Z[i]
            logpsi_j = logpsi_raw[j][j] - 0.5 * Z[j]

            # Shift each by its own max for numerical stability of exp()
            shift_i = np.max(logpsi_i)
            shift_j = np.max(logpsi_j)

            # Evaluate wavefunctions with shift trick
            psi_i = np.exp(logpsi_i - shift_i)
            psi_j = np.exp(logpsi_j - shift_j)

            # L2 norms within evaluation sector j
            norm_i = np.sqrt(np.sum(psi_i ** 2))
            norm_j = np.sqrt(np.sum(psi_j ** 2))

            # Dot product with shift
            dot = float(np.sum(psi_i * psi_j))

            # Normalized overlap
            overlap[i, j] = float(np.abs(dot) / max(norm_i * norm_j, np.finfo(np.float64).eps))

    return overlap

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute 3x3 cross-sector overlap matrix for NQS models."
    )
    parser.add_argument("--arch", type=str, default="FCNet",
                        choices=["FCNet", "JastrowNet", "Conv1DNet", "RBMNet"])
    parser.add_argument("--L", type=int, default=8, help="Lattice size")
    parser.add_argument("--Nmax", type=int, default=8, help="Max occupancy per site")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10000, help="Training epochs per model")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--t", type=float, default=1.0, help="Hopping amplitude")
    parser.add_argument("--U", type=float, default=2.0, help="Interaction strength")
    parser.add_argument("--mu", type=float, default=0.0, help="Chemical potential")
    parser.add_argument("--batch_size", type=int, default=4096, help="Forward pass batch size")
    args = parser.parse_args()

    device = pick_best_device()
    channels = args.Nmax + 1

    # Unit filling: sectors N-1, N, N+1
    N = args.L  # unit filling => total particles = L
    sectors = [N - 1, N, N + 1]
    sector_labels = [str(n) for n in sectors]

    print(f"  Cross-sector overlap: arch={args.arch}, L={args.L}, Nmax={args.Nmax}")
    print(f"  Training sectors: {sectors}")
    print()

    # Generate basis for each sector
    all_basis_by_sector = {}
    for Ntot in sectors:
        basis_states = generate_basis(L=args.L, Ntot=Ntot, Nmax=args.Nmax)
        basis_array = np.array(basis_states, dtype=np.int8)
        all_basis_by_sector[Ntot] = (basis_states, basis_array)
        print(f"  Sector N={Ntot}: {len(basis_states)} states")
    print()

    # Train models on each sector
    builder = ARCH_BUILDERS[args.arch]
    models = []

    for idx, Ntot in enumerate(sectors):
        print(f"--- Model {idx}: training on sector N={Ntot} ---")
        x, y, ed_energy, basis, psi0 = make_ed_dataset(
            L=args.L, Ntot=Ntot, Nmax=args.Nmax,
            t=args.t, U=args.U, mu=args.mu,
            target="logabs",
        )
        model = builder(args.L, channels, args.hidden_dim)
        model = train_model_on_ed(model, x, y, target="logabs",
                                   lr=args.lr, epochs=args.epochs, device=device)
        models.append(model)
        print(f"  ED ground energy: {ed_energy:.6f}")
        print()

    # Compute overlap matrix
    print("--- Computing 3x3 overlap matrix ---")
    overlap = compute_overlap_matrix(models, sectors, all_basis_by_sector, device, args.batch_size, global_Nmax=args.Nmax)

    # Print result
    print()
    print("=" * 70)
    print("  3x3 INTER-SECTOR OVERLAP MATRIX")
    print("=" * 70)
    print(f"  Arch: {args.arch}, L={args.L}, Nmax={args.Nmax}, sectors={sectors}")
    print()

    # Column header
    col_width = 14
    header = " " * (col_width + 14)
    for s in sector_labels:
        header += f"N={s}>{' ' * (col_width - len(s) - 2)}"
    print(header)

    # Rows
    for i, N_i in enumerate(sectors):
        row = f"  model {i} (N={N_i}):"
        for j in range(len(sectors)):
            row += f"  {overlap[i, j]:>14.8e}"
        print(row)

    print("=" * 70)
    print()

if __name__ == "__main__":
    main()
