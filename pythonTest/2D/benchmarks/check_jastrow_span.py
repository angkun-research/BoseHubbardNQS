#!/usr/bin/env python3
"""Analytic Jastrow function-space check, not a supervised training optimizer.

For 2x3/3x3, unit filling, cutoff four, construct the raw onsite
and symmetric displacement-shared pair-table design in float64. Center rows over
the full sector to remove additive log-amplitude constants. The least-squares
residual bounds normalized-log-MSE from below; its coefficients need not minimize
that normalized loss. A positive residual does not prove that the joint physical
overlap/energy gate is impossible, nor is it a bound on achievable overlap.
Supervised training remains Adam-only. Boundaries default to periodic on both
axes; explicit axis flags override the legacy both-axis setting.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.special import logsumexp
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exact_diagonalization import (
    DEFAULT_MAX_DIMENSION, DEFAULT_MAX_SPARSE_BYTES, count_basis_states, exact_ground_state,
)
from models import BOND_CONVENTION, HAMILTONIAN_CONVENTION, JastrowNet
from lattice import add_boundary_arguments, lattice_shape, resolve_boundaries, resolve_cli_boundaries


MAX_DESIGN_BYTES = 256 * 1024**2
FORWARD_BATCH_SIZE = 256


def pair_geometry(Lx, Ly, pbc=True, *, pbc_x=None, pbc_y=None):
    """Independent coordinate pairs, using i=y*Lx+x and minimum axis distances."""
    Lx, Ly = lattice_shape(Lx, Ly)
    pbc_x, pbc_y = resolve_boundaries(pbc, pbc_x, pbc_y)
    pairs = []
    for i, j in combinations(range(Lx * Ly), 2):
        dx, dy = abs(i % Lx - j % Lx), abs(i // Lx - j // Lx)
        if pbc_x:
            dx = min(dx, Lx - dx)
        if pbc_y:
            dy = min(dy, Ly - dy)
        pairs.append((i, j, (dx, dy)))
    classes = sorted({distance for _, _, distance in pairs})
    return pairs, classes


def design_matrix(basis, pairs, classes, channels):
    """Columns match onsite then flattened raw pair tables, including redundancies.

    Each site contributes one onsite indicator. Each unordered pair contributes
    1/2 to table[d,n_i,n_j] and 1/2 to table[d,n_j,n_i]; equal occupations add one
    to the same column. These are the symmetric tables used by JastrowNet.
    """
    indices = {distance: index for index, distance in enumerate(classes)}
    X = np.zeros((len(basis), channels + len(classes) * channels**2), dtype=np.float64)
    for row, state in enumerate(basis):
        for occupation in state:
            X[row, occupation] += 1
        for i, j, distance in pairs:
            offset = channels + indices[distance] * channels**2
            X[row, offset + state[i] * channels + state[j]] += 0.5
            X[row, offset + state[j] * channels + state[i]] += 0.5
    return X


@torch.no_grad()
def forward_check(model, basis, X, coefficients):
    """Check all sector states in bounded batches, without using model pair buffers."""
    channels = model.channels
    model.onsite.copy_(torch.from_numpy(coefficients[:channels]))
    model.pair.copy_(torch.from_numpy(coefficients[channels:].reshape(model.pair.shape)))
    actual = []
    for start in range(0, len(basis), FORWARD_BATCH_SIZE):
        states = torch.tensor(basis[start:start + FORWARD_BATCH_SIZE], dtype=torch.long)
        actual.append(model(F.one_hot(states, channels).double()).numpy())
    actual = np.concatenate(actual)
    expected = X @ coefficients
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    return float(np.max(np.abs(actual - expected)))


def check_geometry(Lx, Ly, seed, pbc=True, *, pbc_x=None, pbc_y=None):
    started = time.perf_counter()
    sites, channels = Lx * Ly, 5
    pbc_x, pbc_y = resolve_boundaries(pbc, pbc_x, pbc_y)
    pairs, classes = pair_geometry(Lx, Ly, pbc, pbc_x=pbc_x, pbc_y=pbc_y)
    dimension = count_basis_states(sites, sites, 4)
    parameters = channels + len(classes) * channels**2
    # Budget X, centered X, thin SVD factors/copies/workspace, and diagnostic
    # vectors. ED retains its separate default dimension and sparse-memory guards.
    estimate = 8 * (6 * dimension * parameters + 8 * parameters**2 + 16 * dimension + 16 * parameters)
    if dimension > DEFAULT_MAX_DIMENSION or estimate > MAX_DESIGN_BYTES:
        raise ValueError(f"diagnostic preflight refused dimension={dimension}, estimated design bytes={estimate}")
    ground_energy, ground, basis, H = exact_ground_state(
        Lx, Ly, Ntot=sites, Nmax=4, t=1.0, U=16.74, mu=0.0, pbc=True,
        max_dimension=DEFAULT_MAX_DIMENSION, max_sparse_bytes=DEFAULT_MAX_SPARSE_BYTES,
        pbc_x=pbc_x, pbc_y=pbc_y,
    )
    if not np.isfinite(ground).all() or not np.all(ground > 0):
        raise ValueError("expected a finite strictly positive ED ground vector; no target floor is applied")
    ground = ground / np.linalg.norm(ground)
    target = np.log(ground)
    X = design_matrix(basis, pairs, classes, channels)
    model = JastrowNet(Lx, Ly, Nmax=4, pbc=pbc, pbc_x=pbc_x, pbc_y=pbc_y).double().eval()
    rng = np.random.default_rng(seed)
    discrepancies = [forward_check(model, basis, X, rng.normal(size=parameters)) for _ in range(2)]
    centered = X - X.mean(axis=0)
    target_centered = target - target.mean()
    left, singular, right = np.linalg.svd(centered, full_matrices=False)
    relative_tolerance = np.finfo(np.float64).eps * max(centered.shape)
    tolerance = relative_tolerance * singular[0]
    keep = singular > tolerance
    coefficients = right[keep].T @ ((left[:, keep].T @ target_centered) / singular[keep])
    residual = centered @ coefficients - target_centered
    lower_bound = float(np.mean(residual**2))
    logs = X @ coefficients
    normalized_logs = logs - 0.5 * logsumexp(2 * logs)
    normalized_loss = float(np.mean((normalized_logs - target)**2))
    # Any normalized residual splits into its centered part and its mean.
    np.testing.assert_allclose(normalized_loss, lower_bound + np.mean(normalized_logs - target)**2,
                               rtol=1e-10, atol=1e-12)
    example_forward_error = forward_check(model, basis, X, coefficients)
    psi = np.exp(normalized_logs)
    applied = H @ psi
    energy = float(psi @ applied)
    overlap = float(abs(ground @ psi))
    error_per_site = (energy - ground_energy) / sites
    return {
        "geometry": [Lx, Ly], "Ntot": sites, "Nmax": 4, "t": 1.0, "U": 16.74, "mu": 0.0,
        "pbc": pbc_x if pbc_x == pbc_y else None, "pbc_x": pbc_x, "pbc_y": pbc_y,
        "shape": list(X.shape), "displacement_classes": [list(distance) for distance in classes],
        "feature_order": "onsite[occupation], then raw pair[class,occupation_i,occupation_j] in C order",
        "estimated_design_workspace_bytes": estimate,
        "effective_centered_rank": int(keep.sum()), "rank_relative_tolerance": float(relative_tolerance),
        "rank_absolute_tolerance": float(tolerance),
        "rank_rule": "singular_value > float64_epsilon * max(design_shape) * largest_singular_value",
        "singular_values": singular.tolist(),
        "rank_at_tolerance_divided_by_10": int(np.count_nonzero(singular > tolerance / 10)),
        "rank_at_tolerance_times_10": int(np.count_nonzero(singular > tolerance * 10)),
        "random_forward_max_abs_errors": discrepancies, "centered_ls_forward_max_abs_error": example_forward_error,
        "minimum_centered_log_mse_lower_bound": lower_bound,
        "centered_residual_rms": float(np.sqrt(lower_bound)), "centered_residual_max_abs": float(np.max(abs(residual))),
        "centered_normal_equation_max_abs": float(np.max(abs(centered.T @ residual))),
        "raw_ed_reference": {"energy": float(ground_energy), "minimum_amplitude": float(ground.min()),
                             "energy_residual_norm": float(np.linalg.norm(H @ ground - ground_energy * ground))},
        "centered_ls_example": {
            "normalized_log_mse": normalized_loss, "normalized_sector_norm": float(np.linalg.norm(psi)),
            "amplitude_overlap_abs_unsquared": overlap, "energy": energy, "energy_per_site": energy / sites,
            "energy_error_to_ed": energy - ground_energy, "energy_error_per_site": error_per_site,
            "energy_variance": float(np.linalg.norm(applied - energy * psi)**2),
            "joint_physical_gate_passed_by_this_example": bool(overlap >= 0.9999 and error_per_site <= 0.001),
        },
        "wall_seconds_including_ed_design_svd_and_checks": time.perf_counter() - started,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometries", nargs="+", choices=["2x3", "3x3"], default=["2x3", "3x3"])
    parser.add_argument("--seed", type=int, default=123, help="random coefficient forward checks only")
    parser.add_argument("--output", type=Path, default=Path("results/jastrow-span.json"))
    add_boundary_arguments(parser)
    args = parser.parse_args(argv)
    resolve_cli_boundaries(args)
    if not 0 <= args.seed < 2**63 or len(set(args.geometries)) != len(args.geometries):
        parser.error("seed must be in [0,2**63); geometries must be distinct")
    output = (ROOT / "benchmarks" / args.output).resolve()
    if not output.is_relative_to(ROOT / "benchmarks" / "results") or output.exists():
        parser.error("output must be a new file under benchmarks/results/")
    torch.set_num_threads(1)
    sources = [ROOT / name for name in ("lattice.py", "exact_diagonalization.py", "models.py")]
    sources.append(Path(__file__).resolve())
    report = {
        "status": "completed", "kind": "analytic_function_space_check_not_Adam_training",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
        "source_sha256_at_start": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "settings": {"geometries": args.geometries, "seed": args.seed, "dtype": "float64", "device": "cpu",
                     "pbc": args.pbc, "pbc_x": args.pbc_x, "pbc_y": args.pbc_y,
                     "threads": 1, "max_ed_dimension": DEFAULT_MAX_DIMENSION,
                     "max_ed_sparse_bytes": DEFAULT_MAX_SPARSE_BYTES, "max_design_workspace_bytes": MAX_DESIGN_BYTES,
                     "forward_batch_size": FORWARD_BATCH_SIZE, "no_target_floor": True},
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "torch": str(torch.__version__),
                        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
        "hamiltonian_convention": HAMILTONIAN_CONVENTION, "bond_convention": BOND_CONVENTION,
        "centering": "Xc=X-mean_rows(X), yc=log(raw_ED)-mean(log(raw_ED)); solve min ||Xc*theta-yc||_2",
        "bound": "Normalized log residual MSE equals centered residual MSE plus squared mean residual; the centered minimum is a lower bound.",
        "interpretation": "Checks log-target representability modulo a constant only. This residual is NOT a bound on achievable overlap. The centered LS example need not minimize normalized-log-MSE, overlap error, or energy. A positive residual does not prove that Fabs>=0.99 or the historical joint gate is impossible. This is not an alternative supervised optimizer; supervised fits use Adam.",
        "resource_scope": "Fixed two tiny geometries; separate conservative ED and design estimates, not a hard process-memory bound. No dense Hamiltonian or checkpoints are allocated/saved.",
        "results": [],
    }
    for geometry in args.geometries:
        result = check_geometry(*(int(size) for size in geometry.split("x")), seed=args.seed, pbc=args.pbc,
                                pbc_x=args.pbc_x, pbc_y=args.pbc_y)
        report["results"].append(result)
        print(f"{geometry}: shape={result['shape']} rank={result['effective_centered_rank']} "
              f"centered_MSE_bound={result['minimum_centered_log_mse_lower_bound']:.12g} "
              f"example_normalized_MSE={result['centered_ls_example']['normalized_log_mse']:.12g}")
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(serialized)
    print(f"diagnostic record: {output}")
    return report


if __name__ == "__main__":
    main()
