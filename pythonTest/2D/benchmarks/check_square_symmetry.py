"""Audit the 3x3, N=9 torus ground state and its invariant orbit basis.

Uses the saved independently constructed ED state, not trained predictions.
Run from the repository root with single-thread BLAS:
  OPENBLAS_NUM_THREADS=1 .venv/bin/python -B pythonTest/2D/benchmarks/check_square_symmetry.py
The default writes a new reproducible audit, preserving the earlier audit files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exact_diagonalization import build_bose_hubbard_hamiltonian

REFERENCE = ROOT / "benchmarks/results/2026-11-01-periodic-focus/checker-3x3-torus-independent-ed-state.npz"
RESULTS = ROOT / "benchmarks/results/2026-11-02-square-symmetry"


def site_permutation(rotation, mirror, dx=0, dy=0):
    """Old site -> new site for T(dx,dy) R**rotation M**mirror, origin (0,0)."""
    result = []
    for site in range(9):
        x, y = site % 3, site // 3
        if mirror:
            y = -y
        for _ in range(rotation):
            x, y = -y, x
        result.append(((x + dx) % 3) + 3 * ((y + dy) % 3))
    return np.asarray(result)


def audit():
    saved = np.load(REFERENCE, allow_pickle=False)
    basis, psi = saved["basis"], saved["ground_state"]
    H, built_basis = build_bose_hubbard_hamiltonian(3, 3, 9, 4)
    assert np.array_equal(basis, built_basis)
    assert np.all(psi > 0) and abs(np.linalg.norm(psi) - 1) < 1e-12
    dimension = len(basis)
    strides = 5 ** np.arange(8, -1, -1)
    codes = basis @ strides
    identity = np.arange(dimension)

    def state_permutation(p):
        transformed = np.empty_like(basis)
        transformed[:, p] = basis
        keys = transformed @ strides
        indices = np.searchsorted(codes, keys)
        assert np.array_equal(codes[indices], keys)
        return indices

    indices = {}
    records = []
    representatives = identity.copy()
    fixed_sum = 0
    site_maps = set()
    for mirror in (False, True):
        for rotation in range(4):
            for dx in range(3):
                for dy in range(3):
                    p = site_permutation(rotation, mirror, dx, dy)
                    site_maps.add(tuple(p))
                    ix = state_permutation(p)
                    indices[rotation, mirror, dx, dy] = ix
                    representatives = np.minimum(representatives, ix)
                    fixed = int(np.count_nonzero(ix == identity))
                    fixed_sum += fixed
                    records.append({"rotation": rotation, "mirror": mirror,
                                    "dx": dx, "dy": dy, "fixed_configurations": fixed,
                                    "state_residual": float(np.linalg.norm(psi[ix] - psi))})
    assert len(site_maps) == 72
    generators = {}
    for name, key in {"Tx": (0, False, 1, 0), "Ty": (0, False, 0, 1),
                      "R": (1, False, 0, 0), "M": (0, True, 0, 0)}.items():
        ix = indices[key]
        difference = H[ix, :][:, ix] - H
        generators[name] = {"H_max_abs_difference": float(np.max(np.abs(difference.data), initial=0)),
                            "state_residual": float(np.linalg.norm(psi[ix] - psi))}

    characters = {"A1": [1]*8, "A2": [1]*4 + [-1]*4,
                  "B1": [1,-1,1,-1,1,-1,1,-1], "B2": [1,-1,1,-1,-1,1,-1,1],
                  "E": [2,0,-2,0,0,0,0,0]}
    point_weights = {}
    for name, chi in characters.items():
        vectors = [psi[indices[r, m, 0, 0]] for m in (False, True) for r in range(4)]
        projection = chi[0] / 8 * np.einsum("g,gn->n", chi, vectors)
        point_weights[name] = float(projection @ projection)
    momentum_weights = []
    for mx in range(3):
        for my in range(3):
            projection = sum(np.exp(-2j * np.pi * (mx*dx + my*dy) / 3)
                             * psi[indices[0, False, dx, dy]]
                             for dx in range(3) for dy in range(3)) / 9
            momentum_weights.append({"mx": mx, "my": my,
                                     "weight": float(np.vdot(projection, projection).real)})

    reps, labels, sizes = np.unique(representatives, return_inverse=True, return_counts=True)
    S = coo_matrix((1 / np.sqrt(sizes[labels]), (identity, labels)), shape=(dimension, len(reps))).tocsr()
    coefficients = S.T @ psi
    reduced = (S.T @ H @ S).tocsr()
    energy = float(psi @ (H @ psi))
    examples = {}
    for name, doublon in (("nearest_neighbor", 1), ("diagonal", 4)):
        config = np.ones(9, dtype=int)
        config[0], config[doublon] = 0, 2
        ix = int(np.searchsorted(codes, config @ strides))
        examples[name] = {"configuration": config.tolist(), "orbit_size": int(sizes[labels[ix]]),
                          "amplitude": float(psi[ix]), "orbit_coefficient": float(coefficients[labels[ix]])}
    histogram = {str(size): int(np.count_nonzero(sizes == size)) for size in np.unique(sizes)}
    sources = [Path(__file__), REFERENCE, ROOT / "exact_diagonalization.py", ROOT / "lattice.py"]
    result = {"status": "PASS", "settings": {"Lx": 3, "Ly": 3, "Ntot": 9, "Nmax": 4,
              "t": 1., "U": 16.74, "mu": 0., "pbc_x": True, "pbc_y": True},
              "dimension": dimension, "energy": energy, "ground_residual": float(np.linalg.norm(H @ psi - energy*psi)),
              "minimum_amplitude": float(psi.min()), "group_order": 72, "generators": generators,
              "all_operations": records, "maximum_state_symmetry_residual": max(r["state_residual"] for r in records),
              "point_group_projector_weights": point_weights, "momentum_projector_weights": momentum_weights,
              "orbit_dimension": len(reps), "burnside_fixed_sum": fixed_sum,
              "orbit_size_histogram": histogram, "reduced_H_nnz": reduced.nnz,
              "reconstruction_residual": float(np.linalg.norm(S @ coefficients - psi)),
              "reduced_eigen_residual": float(np.linalg.norm(reduced @ coefficients - energy*coefficients)),
              "hole_doublon_examples": examples,
              "amplitude_ratio_nearest_over_diagonal": examples["nearest_neighbor"]["amplitude"] / examples["diagonal"]["amplitude"],
              "sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    assert dimension == 19855 and len(reps) == 338 and fixed_sum == 72*338
    assert result["maximum_state_symmetry_residual"] < 1e-12
    assert all(row["H_max_abs_difference"] == 0 for row in generators.values())
    assert result["reduced_eigen_residual"] < 1e-11
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "reproducible-ed-audit.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("output must stay under pythonTest/2D/")
    result = audit()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "dimension", "group_order", "orbit_dimension", "energy", "maximum_state_symmetry_residual")}))


if __name__ == "__main__":
    main()
