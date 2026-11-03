"""Independent fixed 3x3 torus ED reference; no production geometry/ED imports.

From repo root, single-thread CPU:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B \
    pythonTest/2D/benchmarks/check_torus_ed.py
Only the N=9, cutoff=4 sector is enumerated. No full-Fock Hamiltonian, training,
or neighboring-sector diagonalizations. Default dimension/memory guards remain.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'benchmarks/results/2026-11-01-periodic-focus'


def main():
    started = time.perf_counter()
    output = RESULTS / 'checker-3x3-torus-independent-ed.json'
    state_path = RESULTS / 'checker-3x3-torus-independent-ed-state.npz'
    if output.exists() or state_path.exists():
        raise RuntimeError('refuse existing reference artifacts')
    # Independent coefficient counting before any sector/Hamiltonian allocation.
    counts = np.array([1], dtype=np.int64)
    for _ in range(9):
        counts = np.convolve(counts, np.ones(5, dtype=np.int64))
    dimension = int(counts[9])
    estimate = 64 * dimension * 37 + dimension * (256 + 40 * 9 + 32 * 8)
    if dimension > 20000 or estimate > 512 * 1024**2:
        raise ValueError('default ED dimension/sparse planning guard exceeded')
    basis = []
    for prefix in itertools.product(range(5), repeat=8):
        final = 9 - sum(prefix)
        if 0 <= final <= 4:
            basis.append(prefix + (final,))
    assert len(basis) == dimension == 19855
    index = {n: k for k, n in enumerate(basis)}
    basis = np.asarray(basis, dtype=np.int64)
    # Coordinate adjacency, including BOTH physical wraps, no snake ordering.
    bonds = []
    for i, j in itertools.combinations(range(9), 2):
        xi, yi, xj, yj = i % 3, i // 3, j % 3, j // 3
        dx, dy = abs(xi - xj), abs(yi - yj)
        if (yi == yj and min(dx, 3 - dx) == 1) or (xi == xj and min(dy, 3 - dy) == 1):
            bonds.append((i, j))
    assert len(bonds) == 18
    assert sum(i // 3 == j // 3 for i, j in bonds) == 9
    row, col, values = [], [], []
    for k, n in enumerate(basis):
        row.append(k); col.append(k)
        values.append(16.74 / 2 * int(np.sum(n * (n - 1))))
        for i, j in bonds:
            for dest, source in ((i, j), (j, i)):
                if n[source] > 0 and n[dest] < 4:
                    other = n.copy(); other[dest] += 1; other[source] -= 1
                    row.append(index[tuple(other)]); col.append(k)
                    values.append(-math.sqrt((n[dest] + 1) * n[source]))
    H = coo_matrix((values, (row, col)), shape=(dimension, dimension)).tocsr()
    assert (H - H.T).nnz == 0
    rng = np.random.default_rng(20261101)
    v0 = rng.normal(size=dimension); v0 /= np.linalg.norm(v0)
    energies, vectors = eigsh(H, k=2, which='SA', tol=1e-11, ncv=32,
                             maxiter=5000, v0=v0)
    order = np.argsort(energies); energies, vectors = energies[order], vectors[:, order]
    residuals = np.linalg.norm(H @ vectors - vectors * energies, axis=0)
    assert residuals.max() < 1e-7 and energies[1] > energies[0]
    psi = vectors[:, 0]
    if psi.sum() < 0:
        psi = -psi
    psi /= np.linalg.norm(psi)
    assert psi.min() > 0
    probability = psi**2
    density = probability @ basis
    factorial = probability @ (basis * (basis - 1))
    rho = np.diag(density)
    # Explicit ket-action matrix elements, independent of VMC row estimators.
    for i in range(9):
        for j in range(9):
            if i == j:
                continue
            for k, n in enumerate(basis):
                if n[j] > 0 and n[i] < 4:
                    other = n.copy(); other[i] += 1; other[j] -= 1
                    rho[i, j] += psi[index[tuple(other)]] * psi[k] * math.sqrt((n[i] + 1) * n[j])
    rebuilt = -sum(rho[i, j] + rho[j, i] for i, j in bonds) + 16.74 / 2 * factorial.sum()
    assert abs(rebuilt - energies[0]) < 1e-10
    assert np.max(abs(density - 1)) < 1e-10
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(state_path, basis=basis, ground_state=psi)
    result = dict(kind='independent_occupation_ladder_torus_ED', Lx=3, Ly=3,
        pbc_x=True, pbc_y=True, Ntot=9, Nmax=4, t=1., U=16.74, mu=0.,
        dimension=dimension, bonds=bonds, energy=float(energies[0]),
        energy_from_observables=float(rebuilt), density=density.tolist(),
        onsite_factorial_moment=factorial.tolist(), one_body_matrix=rho.tolist(),
        lowest_two_energies=energies.tolist(), neutral_gap=float(energies[1] - energies[0]),
        residual_norm=float(residuals[0]), eigenpair_residual_norms=residuals.tolist(),
        rho_trace=float(np.trace(rho)), rho_hermiticity_max_abs=float(abs(rho-rho.T).max()),
        NN_Gx=float(np.mean([rho[i, j] for i, j in bonds if i // 3 == j // 3])),
        NN_Gy=float(np.mean([rho[i, j] for i, j in bonds if i // 3 != j // 3])),
        density_uniformity_max_abs=float(abs(density-1).max()),
        guard=dict(max_dimension=20000, max_sparse_bytes=512*1024**2, estimated_bytes=estimate),
        eigensolver=dict(k=2, which='SA', ncv=32, tol=1e-11, seed=20261101,
                        starting_vector='Gaussian, not restricted to a spatial symmetry sector'),
        conventions='U/2; unique unordered bonds; physical i=y*3+x; canonical sector; float64',
        interpretation='Neutral fixed-N gap, not charge gap or bulk scaling. Uniform density alone is a weak test; factorial moment and full rho/NN coherence are nontrivial.',
        state_artifact=str(state_path.relative_to(ROOT)),
        state_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        wall_seconds=time.perf_counter()-started)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k: result[k] for k in ('energy', 'neutral_gap', 'eigenpair_residual_norms',
                      'NN_Gx', 'NN_Gy', 'density_uniformity_max_abs', 'wall_seconds')}, indent=2))
    print(output)


if __name__ == '__main__':
    main()
