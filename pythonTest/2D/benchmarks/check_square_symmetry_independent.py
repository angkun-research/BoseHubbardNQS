"""Independent 3x3 torus Burnside/ket-operator check, single-thread CPU only.

No production lattice/ED or parent symmetry helper imports, no eigensolve or fit.
From repo root: OPENBLAS_NUM_THREADS=1 .venv/bin/python -B \
    pythonTest/2D/benchmarks/check_square_symmetry_independent.py
Writes a new audit and refuses replacement; reuses only the validated ED cache.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'benchmarks/results/2026-11-02-square-symmetry'


def main():
    output = RESULTS / 'checker-independent-symmetry-audit.json'
    if output.exists():
        raise RuntimeError('refuse replacing independent audit')
    reference_path = ROOT / 'benchmarks/results/2026-11-01-periodic-focus/checker-3x3-torus-independent-ed.json'
    reference = json.loads(reference_path.read_text())
    state_path = ROOT / reference['state_artifact']
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(state_path) == reference['state_sha256']
    with np.load(state_path, allow_pickle=False) as saved:
        basis, psi = saved['basis'], saved['ground_state']
    assert basis.shape == (19855, 9) and np.all(basis.sum(1) == 9)
    assert np.all(psi > 0) and abs(psi @ psi - 1) < 1e-12
    coords = np.array([(i % 3, i // 3) for i in range(9)])
    # All signed coordinate-axis permutations; no R/M convention reused.
    matrices = []
    for axes in ((0, 1), (1, 0)):
        for signs in itertools.product((-1, 1), repeat=2):
            matrix = np.zeros((2, 2), dtype=int)
            for row in range(2):
                matrix[row, axes[row]] = signs[row]
            matrices.append(matrix)
    permutations = []
    cycle_records = Counter()
    fixed_sum = 0
    for matrix in matrices:
        for shift in itertools.product(range(3), repeat=2):
            xy = (coords @ matrix.T + shift) % 3
            permutation = xy[:, 0] + 3 * xy[:, 1]
            permutations.append(permutation)
            seen = set()
            cycles = []
            for start in range(9):
                if start in seen:
                    continue
                length, site = 0, start
                while site not in seen:
                    seen.add(site)
                    length += 1
                    site = int(permutation[site])
                cycles.append(length)
            # Coefficient of z^9 in product_cycles(1+z^ell+...+z^(4ell)).
            polynomial = [1] + [0] * 9
            for length in cycles:
                updated = [0] * 10
                for degree, count in enumerate(polynomial):
                    for occupation in range(5):
                        power = degree + occupation * length
                        if power <= 9:
                            updated[power] += count
                polynomial = updated
            fixed = polynomial[9]
            cycle_records[tuple(sorted(cycles)), fixed] += 1
            fixed_sum += fixed
    keys = {tuple(p) for p in permutations}
    assert len(keys) == 72 and fixed_sum == 24336 and fixed_sum // 72 == 338
    assert all(tuple(p[q]) in keys for p in permutations for q in permutations)
    assert all(tuple(np.argsort(p)) in keys for p in permutations)
    # Independent physical graph and bosonic ket-ladder Hamiltonian.
    bonds = [(i, j) for i, j in itertools.combinations(range(9), 2)
             if (i // 3 == j // 3 and min(abs(i % 3-j % 3), 3-abs(i % 3-j % 3)) == 1)
             or (i % 3 == j % 3 and min(abs(i // 3-j // 3), 3-abs(i // 3-j // 3)) == 1)]
    assert len(bonds) == 18
    bond_set = set(bonds)
    assert all({tuple(sorted((int(p[i]), int(p[j])))) for i, j in bonds} == bond_set for p in permutations)
    strides = 5 ** np.arange(8, -1, -1, dtype=np.int64)
    codes = basis @ strides
    assert np.all(np.diff(codes) > 0)
    rows, columns = [np.arange(len(basis))], [np.arange(len(basis))]
    values = [8.37 * (basis * (basis-1)).sum(1)]
    for i, j in bonds:
        for dst, src in ((i, j), (j, i)):
            keep = np.flatnonzero((basis[:, src] > 0) & (basis[:, dst] < 4))
            target_code = codes[keep] + strides[dst] - strides[src]
            target = np.searchsorted(codes, target_code)
            assert np.array_equal(codes[target], target_code)
            rows.append(target)
            columns.append(keep)
            values.append(-np.sqrt((basis[keep, dst]+1)*basis[keep, src]))
    H = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                   shape=(19855, 19855)).tocsr()
    energy = float(psi @ (H @ psi))
    assert abs(energy-reference['energy']) < 1e-12
    assert np.linalg.norm(H @ psi-energy*psi) < 1e-10
    representatives = codes.copy()
    state_error = 0.
    generator_errors = []
    fixed_enumeration_sum = 0
    for p in permutations:
        transformed = basis[:, np.argsort(p)]
        transformed_codes = transformed @ strides
        indices = np.searchsorted(codes, transformed_codes)
        assert np.array_equal(codes[indices], transformed_codes)
        fixed_enumeration_sum += int(np.count_nonzero(indices == np.arange(len(basis))))
        representatives = np.minimum(representatives, transformed_codes)
        state_error = max(state_error, float(np.linalg.norm(psi[indices]-psi)))
        if tuple(p) in {tuple((coords[:, 0]+1) % 3 + 3*coords[:, 1]),
                        tuple(coords[:, 0] + 3*((coords[:, 1]+1) % 3)),
                        tuple((-coords[:, 1]) % 3 + 3*(coords[:, 0] % 3)),
                        tuple(coords[:, 0] + 3*((-coords[:, 1]) % 3))}:
            difference = H[indices, :][:, indices] - H
            generator_errors.append(float(np.max(abs(difference.data), initial=0)))
    assert fixed_enumeration_sum == fixed_sum and state_error < 1e-12
    assert len(generator_errors) == 4 and max(generator_errors) == 0
    _, labels, sizes = np.unique(representatives, return_inverse=True, return_counts=True)
    assert len(sizes) == 338
    B = coo_matrix((1/np.sqrt(sizes[labels]), (np.arange(19855), labels)), shape=(19855, 338)).tocsr()
    reduced = (B.T @ H @ B).tocsr()
    coefficients = B.T @ psi
    reconstruction_error = float(np.linalg.norm(B @ coefficients - psi))
    reduced_residual = float(np.linalg.norm(reduced @ coefficients-energy*coefficients))
    assert reconstruction_error < 1e-12 and reduced_residual < 1e-11 and reduced.nnz == 5455
    result = {'status': 'PASS_INDEPENDENT_BURNSIDE_AND_PHYSICAL_SYMMETRY',
              'recorded_at': datetime.now(timezone.utc).isoformat(), 'group_order': 72,
              'Burnside_fixed_sum': fixed_sum, 'invariant_dimension': 338,
              'cycle_fixed_counts': [{'cycle_lengths': list(cycles), 'operation_count': count,
                                     'fixed_sector_configurations': fixed}
                                    for (cycles, fixed), count in sorted(cycle_records.items())],
              'orbit_size_histogram': {str(k): int(v) for k, v in Counter(sizes).items()},
              'Hamiltonian_generator_max_abs_errors': generator_errors,
              'ED_energy': energy, 'ED_symmetry_max_residual': state_error,
              'reconstruction_error': reconstruction_error, 'reduced_H_nnz': reduced.nnz,
              'reduced_H_eigen_residual': reduced_residual,
              'operator_contract': 'U_g b_i^dag b_j U_g^-1=b_p(i)^dag b_p(j); n_i and n_i(n_i-1) permute. Both wrap axes,18unique bonds,U/2,uniformcutoff. H real andfixedN; no flux.',
              'interpretation': 'Spatial k=0,A1 with338multiplicity directions; fixedU1charge9 is nontrivial phase exp(i9phi), notcharge0 orSU2singlet. AntiunitaryK iscomplexconjugation,K^2=+1; not anextraunitarysitepermutation. Symmetrydoesnotfix all338coefficients oroffsector extension.',
              'method': 'Signed-axis affine maps, groupclosure/inverses and cycle-polynomial Burnside before occupation enumeration; independent ket operators and savedEDcache, no parenthelper/productionED imports,no new eigensolve or training.',
              'input_sha256': {str(reference_path.relative_to(ROOT)): digest(reference_path),
                               str(state_path.relative_to(ROOT)): digest(state_path)},
              'helper_sha256': digest(Path(__file__))}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('status','group_order','Burnside_fixed_sum','invariant_dimension','cycle_fixed_counts')}))


if __name__ == '__main__':
    main()
