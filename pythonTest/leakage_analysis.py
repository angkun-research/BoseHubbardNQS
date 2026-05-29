#!/usr/bin/env python3
"""
Leakage Analysis for Neural Quantum States.

Measures the probability distribution of a trained NQS across all total
particle number sectors, revealing how much amplitude leaks outside the
training sector.

Physics motivation:
  The model is trained on a fixed-particle-number sector (e.g., Ntot=8),
  but unconstrained architectures can produce nonzero amplitudes for states
  with different total particle numbers. This leakage indicates that the
  model wastes variational capacity on wrong symmetry sectors.

Usage:
    # Analyze a trained model (full basis, L=8 Nmax=8)
    python leakage_analysis.py --model data/spin_net.pt --architecture FCNet

    # Quick test with a smaller basis
    python leakage_analysis.py --model data/spin_net.pt --architecture FCNet --Nmax 4

    # No overlap computation (no ED ground state needed)
    python leakage_analysis.py --model data/spin_net.pt --architecture RBMNet --no_overlap

    # Train fresh and analyze
    python leakage_analysis.py --architecture Conv1DNet --hidden_dim 64

    # Compare architectures
    for arch in FCNet JastrowNet Conv1DNet RBMNet; do
        python leakage_analysis.py --model data/${arch}_net.pt --architecture $arch \\
            --output fig_leakage_${arch}.pdf
    done
"""

import argparse
import sys
import time as _time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp as _sp_logsumexp
import torch

from NeuralNetworkClass import FCNet, JastrowNet, Conv1DNet, RBMNet
from BoseHubbardED import basis_to_one_hot
from utils import pick_best_device

# ── Architecture builders ──────────────────────────────────────────────────

ARCH_BUILDERS = {
    "FCNet":      lambda L, Ch, hd: FCNet(n_sites=L, channels=Ch, hidden_dim=hd),
    "JastrowNet": lambda L, Ch, hd: JastrowNet(n_sites=L, channels=Ch),
    "Conv1DNet":  lambda L, Ch, hd: Conv1DNet(n_sites=L, channels=Ch, hidden_dim=hd),
    "RBMNet":     lambda L, Ch, hd: RBMNet(n_sites=L, channels=Ch, n_hidden=hd),
}

# ── State generation ───────────────────────────────────────────────────────

def generate_all_basis(L, Nmax):
    """Generate ALL occupation states (n_0, ..., n_{L-1}) with 0 <= n_i <= Nmax.

    States are ordered first by increasing total particle number, then by
    lexicographic order within each sector (stable sort on the product ordering).

    Args:
        L: number of lattice sites
        Nmax: local occupancy cutoff

    Returns:
        all_states:       np.ndarray  shape (total, L), dtype=int8
        states_by_sector: dict {Ntot (int): np.ndarray of indices into all_states}
        ntot_all:         np.ndarray shape (total,), dtype=int8
    """
    channels = Nmax + 1
    total = channels ** L

    # Base-(Nmax+1) counting — last site changes fastest (lexicographic product order)
    all_states = np.zeros((total, L), dtype=np.int8)
    for i in range(L):
        stride = channels ** (L - 1 - i)
        all_states[:, i] = (np.arange(total, dtype=np.int32) // stride) % channels

    # Total particle number per state
    ntot_all = all_states.sum(axis=1)  # shape (total,), dtype=int8

    # Stable sort by Ntot — preserves lexicographic order within each sector
    order = np.argsort(ntot_all, kind="stable")
    all_states = all_states[order]
    ntot_all = ntot_all[order]

    # Group by sector
    states_by_sector = defaultdict(list)
    for idx in range(total):
        states_by_sector[int(ntot_all[idx])].append(idx)
    states_by_sector = {k: np.array(v, dtype=np.int32) for k, v in states_by_sector.items()}

    return all_states, states_by_sector, ntot_all


# ── Model evaluation (chunked for memory) ───────────────────────────────────

def _one_hot_buffer(batch_states, n_sites, nmax, buf):
    """In-place one-hot encoding into pre-allocated buffer."""
    channels = nmax + 1
    for s in range(n_sites):
        buf[:, s, batch_states[:, s]] = 1.0
    return buf


@torch.no_grad()
def evaluate_logpsi_chunks(model, all_states, device, batch_size=4096):
    """Evaluate logpsi for all basis states in memory-managed chunks.

    Args:
        model:      trained PyTorch nn.Module
        all_states: np.ndarray  shape (total, L), dtype=int8
        device:     torch device
        batch_size: number of states per forward pass

    Returns:
        logpsi: np.ndarray  shape (total,), dtype=float32
    """
    model.eval()
    total = all_states.shape[0]
    logpsi_all = np.zeros(total, dtype=np.float32)

    model = model.to(device)
    n_sites = all_states.shape[1]
    nmax = int(all_states.max())

    # Pre-allocate one-hot buffer (max batch_size)
    buf = np.empty((batch_size, n_sites, nmax + 1), dtype=np.float32)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        bs = end - start
        batch = all_states[start:end]

        _one_hot_buffer(batch, n_sites, nmax, buf[:bs])
        x_tensor = torch.tensor(buf[:bs], dtype=torch.float32, device=device)

        logpsi_all[start:end] = model(x_tensor).cpu().numpy().astype(np.float32)

    return logpsi_all


# ── Core analysis ──────────────────────────────────────────────────────────

def compute_sector_probabilities(model, L, Nmax, device, batch_size=4096):
    """Compute P(N') = sum_{s: sum(n_i)=N'} P(s) for every total particle number sector.

    Uses full-basis normalization:
        P(s) = exp(2*logpsi(s) - Z),  Z = logsumexp_i(2*logpsi(i))  over all i.

    Args:
        model:      trained PyTorch nn.Module
        L, Nmax:    lattice parameters
        device:     torch device
        batch_size: chunk size for evaluation

    Returns:
        sector_probs:      dict {N' (int): probability (float)}
        states_by_sector:  dict {N' (int): np.ndarray of indices into all_states}
        logpsi_all:        np.ndarray of log-amplitudes for all states
        all_states:        np.ndarray of all occupation tuples
        ntot_all:          np.ndarray of total particle numbers
    """
    print("=" * 60)
    print("  SECTOR PROBABILITY ANALYSIS")
    print("=" * 60)

    all_states, states_by_sector, ntot_all = generate_all_basis(L, Nmax)
    total = all_states.shape[0]
    max_ntot = L * Nmax

    print(f"\n  L={L}, Nmax={Nmax}  ->  total states = {total:,}  ({(Nmax+1)}^{L})")
    print(f"  Sectors: N'=0 .. N'={max_ntot}")
    for n in range(max_ntot + 1):
        cnt = len(states_by_sector[n])
        if cnt > 0:
            print(f"    N'={n:2d}: {cnt:>10,} states")

    # Evaluate logpsi in chunks
    print(f"\n  Evaluating model ({total:,} states, batch={batch_size}) ...")
    t0 = _time.perf_counter()
    logpsi_all = evaluate_logpsi_chunks(model, all_states, device, batch_size)
    t1 = _time.perf_counter()
    print(f"  Evaluation time: {t1 - t0:.1f}s  ({total / (t1 - t0):,.0f} states/s)")

    # Full normalization constant Z = logsumexp(2*logpsi)
    Z_all = _sp_logsumexp(2.0 * logpsi_all)
    print(f"\n  Z = logsumexp(2*logpsi) over all states = {Z_all:.6f}")

    # Sector probabilities
    sector_probs = {}
    for n in range(max_ntot + 1):
        idx = states_by_sector[n]
        log_prob = _sp_logsumexp(2.0 * logpsi_all[idx]) - Z_all
        sector_probs[n] = float(np.exp(log_prob))

    # Print non-negligible sectors
    nonzero = {k: v for k, v in sector_probs.items() if v > 1e-12}
    print(f"\n  Non-negligible sectors (P > 1e-12):")
    for n in sorted(nonzero.keys(), key=lambda x: -nonzero[x]):
        print(f"    N'={n:2d}: {nonzero[n]:.8e}")

    total_prob = sum(sector_probs.values())
    print(f"\n  Sum P(N') = {total_prob:.8f}  (should be 1.0)")

    return sector_probs, states_by_sector, logpsi_all, all_states, ntot_all


def compute_leakage_fraction(model, L, Nmax, train_Ntot, device, batch_size=4096):
    """Return total probability leakage outside the training sector.

    leakage = 1.0 - P(N_train)

    Args:
        model, L, Nmax, train_Ntot, device, batch_size: see compute_sector_probabilities

    Returns:
        leakage:   float
        sector_probs: dict {N': probability}
    """
    sector_probs, *_ = compute_sector_probabilities(
        model, L, Nmax, device, batch_size
    )
    P_train = sector_probs.get(train_Ntot, 0.0)
    leakage = 1.0 - P_train
    print(f"\n  === LEAKAGE FRACTION: {leakage:.6f}  ({leakage*100:.4f}%) ===")
    return leakage, sector_probs


def compute_sectored_overlap(model, L, Nmax, train_Ntot, true_basis, true_state, device, batch_size=4096):
    """Compute overlap of the model's amplitude with the ED ground state, sector by sector.

    For each sector N':
        overlap(N') = | < psi_model|_N'  |  phi_ED|_N' > |
                      -------------------------
                      ||psi_model|_N'|| * ||phi_ED|_N'||

    For N' == train_Ntot: measures fidelity within the training sector.
    For N' != train_Ntot: phi_ED has no amplitude -> overlap = 0.0 (reported as None).

    Additionally reports the model's norm restricted to each sector, which
    should equal sqrt(P(N')).

    Args:
        model:       trained PyTorch nn.Module
        L, Nmax:     lattice parameters
        train_Ntot:  total particle number used in training
        true_basis:  list of occupation tuples (ED training sector only)
        true_state:  ED ground-state eigenvector (shape = |training sector|)
        device:      torch device
        batch_size:  chunk size

    Returns:
        sectored_overlap:   dict {N': float or None}
        sector_model_norms: dict {N': model L2 norm restricted to sector}
        full_overlap:       float  (total fidelity)
    """
    # Reuse sector probabilities (also gives logpsi_all)
    sector_probs, states_by_sector, logpsi_all, all_states, _ = compute_sector_probabilities(
        model, L, Nmax, device, batch_size
    )
    Z_all = _sp_logsumexp(2.0 * logpsi_all)

    # Build lookup: true_basis state tuple -> true amplitude
    true_amp_map = {}
    for state_tuple, amp in zip(true_basis, true_state):
        true_amp_map[tuple(state_tuple)] = amp

    # Training sector indices into all_states
    train_indices = states_by_sector[train_Ntot]
    train_states = all_states[train_indices]
    train_state_tuples = [tuple(row) for row in train_states]

    # Model's normalized amplitude in training sector (full normalization)
    train_logpsi = logpsi_all[train_indices]
    train_amp_model = np.exp(train_logpsi - 0.5 * Z_all)

    # Build true amplitude vector in the same ordering as train_amp_model
    train_amp_true = np.zeros(len(train_indices), dtype=np.float64)
    for i, s in enumerate(train_state_tuples):
        if s in true_amp_map:
            train_amp_true[i] = true_amp_map[s]

    true_norm = np.linalg.norm(train_amp_true)

    # Full fidelity
    if true_norm > 0:
        model_norm = np.linalg.norm(train_amp_model)
        full_overlap = float(np.abs(np.dot(train_amp_model, train_amp_true))
                           / (model_norm * true_norm))
    else:
        full_overlap = float("nan")

    # Sector-by-sector
    max_ntot = L * Nmax
    sectored_overlap = {}
    sector_model_norms = {}

    for n in range(max_ntot + 1):
        idx = states_by_sector[n]
        logpsi_n = logpsi_all[idx]

        # Model's L2 norm in this sector
        sector_model_norms[n] = float(np.sqrt(np.exp(_sp_logsumexp(2.0 * logpsi_n) - Z_all)))

        if n == train_Ntot:
            # Overlap within training sector
            if true_norm > 0:
                model_norm_n = np.linalg.norm(train_amp_model)
                sectored_overlap[n] = float(np.abs(np.dot(train_amp_model, train_amp_true))
                                            / (model_norm_n * true_norm))
            else:
                sectored_overlap[n] = None
        else:
            # ED ground state has no amplitude outside training sector
            sectored_overlap[n] = None

    # Summary
    print(f"\n  === OVERLAP ANALYSIS ===")
    if not np.isnan(full_overlap):
        print(f"  Full fidelity:               {full_overlap:.8e}")
    print(f"  P(Ntrain) = {train_Ntot}:       {sector_probs.get(train_Ntot, 0.0):.8e}")
    print(f"  Model norm in training sector: {sector_model_norms.get(train_Ntot, 0.0):.8e}")
    nonzero_ov = {k: v for k, v in sectored_overlap.items() if v is not None}
    for n in sorted(nonzero_ov.keys()):
        print(f"  overlap(N'={n:2d}):           {nonzero_ov[n]:.8e}")

    return sectored_overlap, sector_model_norms, full_overlap


# ── Helpers that consume pre-computed data (no model evaluation) ──

def _compute_leakage_from_probs(sector_probs, train_Ntot):
    """Return total probability leakage outside the training sector.

    No model evaluation — works on already-computed sector_probs.
    """
    P_train = sector_probs.get(train_Ntot, 0.0)
    return 1.0 - P_train


def _compute_sectored_overlap_from_probs(
    sector_probs, states_by_sector, logpsi_all, Z_all,
    train_Ntot, true_basis, true_state, all_states,
):
    """Compute sectored overlap and model norms from pre-computed data.

    No model evaluation — works on already-computed logpsi_all, Z_all, etc.
    """
    # Build lookup: true_basis state tuple -> true amplitude
    true_amp_map = {}
    for state_tuple, amp in zip(true_basis, true_state):
        true_amp_map[tuple(state_tuple)] = amp

    # Training sector indices into all_states
    train_indices = states_by_sector[train_Ntot]

    # Model's normalized amplitude in training sector (full normalization)
    train_logpsi = logpsi_all[train_indices]
    train_amp_model = np.exp(train_logpsi - 0.5 * Z_all)

    # Build true amplitude vector in the same ordering as train_amp_model
    train_amp_true = np.zeros(len(train_indices), dtype=np.float64)
    for i, idx in enumerate(train_indices):
        s_tuple = tuple(all_states[idx])
        if s_tuple in true_amp_map:
            train_amp_true[i] = true_amp_map[s_tuple]

    true_norm = np.linalg.norm(train_amp_true)

    # Full fidelity
    if true_norm > 0:
        model_norm = np.linalg.norm(train_amp_model)
        full_overlap = float(np.abs(np.dot(train_amp_model, train_amp_true))
                             / (model_norm * true_norm))
    else:
        full_overlap = float("nan")

    # Sector-by-sector
    max_ntot = len(states_by_sector) - 1  # N' ranges 0..max
    # Actually compute max_ntot from the keys we have
    all_sectors = sorted(states_by_sector.keys())
    max_ntot = max(all_sectors) if all_sectors else 0

    sectored_overlap = {}
    sector_model_norms = {}

    for n in range(max_ntot + 1):
        if n not in states_by_sector:
            continue
        idx = states_by_sector[n]
        logpsi_n = logpsi_all[idx]

        # Model's L2 norm in this sector
        sector_model_norms[n] = float(np.sqrt(np.exp(_sp_logsumexp(2.0 * logpsi_n) - Z_all)))

        if n == train_Ntot:
            if true_norm > 0:
                model_norm_n = np.linalg.norm(train_amp_model)
                sectored_overlap[n] = float(np.abs(np.dot(train_amp_model, train_amp_true))
                                            / (model_norm_n * true_norm))
            else:
                sectored_overlap[n] = None
        else:
            sectored_overlap[n] = None

    return sectored_overlap, sector_model_norms, full_overlap


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_leakage_distribution(sector_probs, train_Ntot, output_file="fig_leakage.pdf", label=""):
    """Plot P(N') as a bar chart, highlighting the training sector.

    Args:
        sector_probs: dict {N': probability}
        train_Ntot:   training total particle number
        output_file:  path to save the figure
        label:        subplot / file label (architectural info)
    """
    sectors = sorted(sector_probs.keys())
    probs = [sector_probs[n] for n in sectors]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["steelblue" if n != train_Ntot else "darkred" for n in sectors]
    ax.bar(sectors, probs, color=colors, edgecolor="white", linewidth=0.5, width=0.8)
    ax.axvline(
        x=train_Ntot, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
        label=f"Training sector (N'={train_Ntot})",
    )

    # Annotate training sector probability
    P_train = sector_probs.get(train_Ntot, 0.0)
    ax.annotate(
        f"P(Ntrain) = {P_train:.6f}",
        xy=(train_Ntot, P_train),
        xytext=(train_Ntot + 2, max(P_train * 0.5, 1e-15)),
        fontsize=9, color="darkred", fontweight="bold",
    )

    # Annotate leakage
    leakage = 1.0 - P_train
    ax.set_title(
        f"Leakage Distribution {label}  |  Leakage = {leakage:.4f}",
        fontsize=12,
    )
    ax.set_xlabel("Total particle number N'")
    ax.set_ylabel(r"$P(N')$")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    ax.set_xticks(range(sectors[0], sectors[-1] + 1, 2))

    plt.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {output_file}")


# ── Model loading ──────────────────────────────────────────────────────────

def load_model_from_checkpoint(arch_name, model_path, L, Nmax, hidden_dim=128):
    """Instantiate a model and load its state dict from a .pt checkpoint.

    Returns the trained model (eval mode, CPU).
    """
    channels = Nmax + 1
    builder = ARCH_BUILDERS[arch_name]
    model = builder(L, channels, hidden_dim)

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    model.eval()
    print(f"  Loaded model from {model_path} ({arch_name})")
    return model


# ── CLI / main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Measure NQS leakage across total particle number sectors.",
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .pt checkpoint (default: train fresh on ED ground state)")
    parser.add_argument("--architecture", type=str, default="FCNet",
                        choices=["FCNet", "JastrowNet", "Conv1DNet", "RBMNet"],
                        help="NN architecture (default: FCNet)")
    parser.add_argument("--L", type=int, default=4, help="Number of lattice sites")
    parser.add_argument("--Nmax", type=int, default=4, help="Local occupancy cutoff")
    parser.add_argument("--train_Ntot", type=int, default=8,
                        help="Total particle number used in training")
    parser.add_argument("--t", type=float, default=1.0, help="Hopping amplitude")
    parser.add_argument("--U", type=float, default=2.0, help="Interaction strength")
    parser.add_argument("--mu", type=float, default=0.0, help="Chemical potential")
    parser.add_argument("--pbc", action="store_true", default=True,
                        help="Periodic boundary conditions")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Chunk size for model evaluation")
    parser.add_argument("--output", type=str, default="fig_leakage.pdf",
                        help="Output plot filename")
    parser.add_argument("--no_overlap", action="store_true",
                        help="Skip overlap computation (no ED ground state needed)")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension (FCNet/Conv1DNet) or n_hidden (RBMNet)")
    args = parser.parse_args()

    # ── Load or train model ──────────────────────────────────────────
    device = pick_best_device()
    print(f"Using device: {device}\n")

    true_basis = None
    true_state = None
    ed_energy = None

    if args.model is not None:
        model = load_model_from_checkpoint(
            args.architecture, args.model, args.L, args.Nmax, args.hidden_dim,
        )
    else:
        print("No model file provided -- training a fresh model on ED ground state...")
        from TrainEDstates import make_ed_dataset, train_model_on_ed

        channels = args.Nmax + 1
        builder = ARCH_BUILDERS[args.architecture]
        model = builder(args.L, channels, args.hidden_dim)

        x, y, ed_energy, true_basis, true_state = make_ed_dataset(
            L=args.L,
            Ntot=args.train_Ntot,
            Nmax=args.Nmax,
            t=args.t,
            U=args.U,
            mu=args.mu,
            pbc=args.pbc,
            target="logabs",
        )
        model = train_model_on_ed(
            model, x, y, target="logabs", lr=1e-3, epochs=10000, device=device,
        )
        print(f"\n  ED ground-state energy: {ed_energy:.6f}")

    # ── Memory check ─────────────────────────────────────────────────
    total_states = (args.Nmax + 1) ** args.L
    # Rough estimate: all_states (int8*L) + ntot (int8) + logpsi (float32)
    est_mem_gb = total_states * (args.L + 1) * 1 + total_states * 4
    est_mem_gb /= 1024 ** 3
    if est_mem_gb > 2:
        print(f"\n  WARNING: Estimated memory ~{est_mem_gb:.1f} GB. "
              "Consider using --Nmax < 8 for a smaller basis.\n")

    # ── Sector probability analysis (SINGLE evaluation) ───────────────
    sector_probs, states_by_sector, logpsi_all, all_states, ntot_all = compute_sector_probabilities(
        model, args.L, args.Nmax, device, args.batch_size,
    )
    Z_all = _sp_logsumexp(2.0 * logpsi_all)

    # ── Leakage fraction (from pre-computed data) ─────────────────
    leakage = _compute_leakage_from_probs(sector_probs, args.train_Ntot)
    print(f"\n  === LEAKAGE FRACTION: {leakage:.6f}  ({leakage*100:.4f}%) ===")

    # ── Overlap analysis (optional) ───────────────────────────────
    if args.no_overlap:
        sectored_overlap = {}
        sector_model_norms = {}
        full_overlap = float("nan")
        print("\n  Overlap analysis skipped (--no_overlap).")
    else:
        if true_basis is None or true_state is None:
            # User provided --model but didn't train fresh; recompute ED state
            from BoseHubbardED import exact_ground_state
            ed_energy, true_state, true_basis, _ = exact_ground_state(
                L=args.L,
                Ntot=args.train_Ntot,
                Nmax=args.Nmax,
                t=args.t,
                U=args.U,
                mu=args.mu,
                pbc=args.pbc,
            )
            print(f"\n  Recomputed ED ground state: energy = {ed_energy:.6f}")
            print(f"  Hilbert space dimension: {len(true_basis)}")

        if true_basis is not None and true_state is not None:
            (
                sectored_overlap,
                sector_model_norms,
                full_overlap,
            ) = _compute_sectored_overlap_from_probs(
                sector_probs, states_by_sector, logpsi_all, Z_all,
                args.train_Ntot, true_basis, true_state, all_states,
            )
        else:
            sectored_overlap = {}
            sector_model_norms = {}
            full_overlap = float("nan")
            print("\n  Overlap analysis skipped (no ED ground state available).")

    # ── Plot ─────────────────────────────────────────────────────────
    arch_label = f" [{args.architecture}]" if args.model else ""
    plot_leakage_distribution(sector_probs, args.train_Ntot, args.output, label=arch_label)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Architecture:     {args.architecture}")
    print(f"  Physics:          L={args.L}, Nmax={args.Nmax}, train_Ntot={args.train_Ntot}")
    print(f"                    t={args.t}, U={args.U}, mu={args.mu}, pbc={args.pbc}")
    print(f"  Total states:     {total_states:,}")
    P_train = sector_probs.get(args.train_Ntot, 0.0)
    print(f"  P(Ntrain):        {P_train:.8e}")
    print(f"  Leakage:          {leakage:.8e} ({leakage*100:.4f}%)")
    if not np.isnan(full_overlap):
        print(f"  Full fidelity:    {full_overlap:.8e}")
    if ed_energy is not None:
        print(f"  ED energy:        {ed_energy:.8f}")
    n_nonzero = len([k for k, v in sector_probs.items() if v > 1e-12])
    print(f"  Non-zero sectors: {n_nonzero} / {len(sector_probs)}")
    top = sorted(sector_probs.items(), key=lambda x: -x[1])[:10]
    print(f"  Top sectors:")
    for n, p in top:
        marker = " <- train" if n == args.train_Ntot else ""
        print(f"    N'={n:2d}: {p:.8e}{marker}")
    print("=" * 60)


if __name__ == "__main__":
    main()
