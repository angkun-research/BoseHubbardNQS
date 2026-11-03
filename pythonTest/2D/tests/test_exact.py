"""CPU physics and resource checks. Run from the repository root:

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s pythonTest/2D/tests -v
"""

import importlib.util
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import issparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import exact_diagonalization as ed
from lattice import bonds_2d


class GeometryTests(unittest.TestCase):
    def test_rectangular_bonds(self):
        open_edges = {(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)}
        self.assertEqual(set(bonds_2d(3, 2, False)), open_edges)
        self.assertEqual(set(bonds_2d(3, 2)), open_edges | {(0, 2), (3, 5)})
        self.assertEqual(bonds_2d(2), [(0, 1), (0, 2), (1, 3), (2, 3)])

    def test_degenerate_axes(self):
        self.assertEqual(bonds_2d(1, 1), [])
        self.assertEqual(bonds_2d(2, 1), [(0, 1)])
        self.assertEqual(bonds_2d(1, 2), [(0, 1)])
        self.assertEqual(bonds_2d(1, 3), [(0, 1), (0, 2), (1, 2)])
        self.assertEqual(bonds_2d(1, 3, False), [(0, 1), (1, 2)])
        for args in ((0, 2), (2, -1), (2.5, 2), (True, 2)):
            with self.assertRaises(ValueError):
                bonds_2d(*args)


class BasisTests(unittest.TestCase):
    def test_exact_count_and_basis(self):
        for sites, cutoff in ((1, 0), (3, 2), (4, 3)):
            states = list(itertools.product(range(cutoff + 1), repeat=sites))
            for total in range(sites * cutoff + 1):
                expected = [state for state in states if sum(state) == total]
                self.assertEqual(ed.count_basis_states(sites, total, cutoff), len(expected))
                self.assertEqual(ed.generate_basis(sites, total, cutoff), expected)
        self.assertEqual(ed.count_basis_states(40, 20, 1), 137846528820)

    def test_guards_before_enumeration(self):
        with patch.object(ed, "generate_basis", side_effect=AssertionError("enumerated")):
            with self.assertRaisesRegex(ValueError, "dimension"):
                ed.build_bose_hubbard_hamiltonian(4, Nmax=4)
            with self.assertRaisesRegex(ValueError, "sparse memory"):
                ed.build_bose_hubbard_hamiltonian(2, max_sparse_bytes=1)
        with self.assertRaisesRegex(ValueError, "dimension"):
            ed.generate_basis(10, 10, max_dimension=2)
        for total in (-1, 9):
            with self.assertRaises(ValueError):
                ed.count_basis_states(2, total, 4)

    def test_one_hot(self):
        basis = [(0, 2), (1, 1), (2, 0)]
        features = ed.basis_to_one_hot(basis, 2, 2)
        self.assertEqual(features.shape, (3, 2, 3))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_array_equal(features.argmax(-1), basis)
        np.testing.assert_array_equal(features.sum(-1), np.ones((3, 2)))
        self.assertEqual(ed.basis_to_one_hot([], 2, 2).shape, (0, 2, 3))

    def test_counting_workspace_preflight(self):
        # Previously attempted a billion-element DP list before dimension checks.
        with self.assertRaisesRegex(ValueError, "counting work"):
            ed.count_basis_states(2, 10**9, 10**9)
        with patch.object(ed, "bonds_2d", side_effect=AssertionError("built geometry")):
            with self.assertRaisesRegex(ValueError, "counting work"):
                ed.build_bose_hubbard_hamiltonian(2, 1, Ntot=10**9, Nmax=10**9)
            with self.assertRaisesRegex(ValueError, "sparse memory"):
                ed.build_bose_hubbard_hamiltonian(10**9, 1, Ntot=0)
        # Single-site sectors need no DP workspace, even with a large cutoff.
        self.assertEqual(ed.count_basis_states(1, 10**9, 2 * 10**9), 1)

    def test_thin_lattice_without_recursion(self):
        basis = ed.generate_basis(1000, 1, 1)
        self.assertEqual(len(basis), 1000)
        self.assertEqual(basis, sorted(basis))
        self.assertEqual(basis[0], (0,) * 999 + (1,))
        self.assertEqual(basis[-1], (1,) + (0,) * 999)
        self.assertTrue(all(sum(state) == 1 and len(state) == 1000 for state in basis))


class HamiltonianTests(unittest.TestCase):
    def test_physical_parameter_domain(self):
        for name in ("t", "U", "mu"):
            for value in (1 + 0j, 1j, np.complex128(1), np.nan, np.inf,
                          -np.inf, "1", None, True, [1], 10**1000):
                with self.subTest(name=name, value=repr(value)):
                    with self.assertRaisesRegex(ValueError, name):
                        ed.build_bose_hubbard_hamiltonian(2, 1, Ntot=1, **{name: value})
        for name in ("t", "U"):
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                ed.build_bose_hubbard_hamiltonian(2, 1, Ntot=1, **{name: -1})
        H, _ = ed.build_bose_hubbard_hamiltonian(
            2, 1, Ntot=1, t=np.float64(0), U=np.int64(0), mu=-0.5
        )
        np.testing.assert_array_equal(H.toarray(), 0.5 * np.eye(2))

    def test_diagonal_and_zero_ground_states(self):
        cases = [dict(Lx=2, Ly=1, Ntot=1, t=0),
                 dict(Lx=2, Ntot=2, t=0, U=0),
                 dict(Lx=2, Ntot=4, t=0, U=16.74, mu=0.3)]
        with patch.object(ed, "eigsh", side_effect=AssertionError("called ARPACK")):
            for settings in cases:
                with self.subTest(settings=settings):
                    energy, state, basis, H = ed.exact_ground_state(**settings)
                    diagonal = H.diagonal()
                    self.assertEqual(energy, min(diagonal))
                    self.assertEqual(np.argmax(state), np.argmin(diagonal))
                    self.assertEqual(np.linalg.norm(state), 1)
                    np.testing.assert_allclose(H @ state, energy * state, atol=1e-14)

    def test_independent_ladder_operator_reference(self):
        # Build the full tensor-product operator, then project onto N=3.
        cutoff, sites, total = 2, 4, 3
        annihilation = np.diag(np.sqrt(np.arange(1, cutoff + 1)), 1)
        identity = np.eye(cutoff + 1)
        operators = []
        for site in range(sites):
            operator = np.ones((1, 1))
            for index in range(sites):
                operator = np.kron(operator, annihilation if site == index else identity)
            operators.append(operator)
        t, U, mu = 0.7, 5.3, 0.2
        full = np.zeros(((cutoff + 1)**sites,) * 2)
        for operator in operators:
            number = operator.T @ operator
            full += U / 2 * (number @ number - number) - mu * number
        # Explicit 2x2 square edges, independent of the geometry helper.
        for i, j in ((0, 1), (0, 2), (1, 3), (2, 3)):
            full -= t * (operators[i].T @ operators[j] + operators[j].T @ operators[i])
        H, basis = ed.build_bose_hubbard_hamiltonian(
            2, Ntot=total, Nmax=cutoff, t=t, U=U, mu=mu
        )
        indices = [np.ravel_multi_index(state, (cutoff + 1,) * sites) for state in basis]
        self.assertTrue(issparse(H))
        np.testing.assert_allclose(H.toarray(), full[np.ix_(indices, indices)], atol=1e-13)
        np.testing.assert_allclose(H.toarray(), H.toarray().T, atol=0)
        self.assertTrue(all(sum(state) == total for state in basis))
        index = {state: i for i, state in enumerate(basis)}
        self.assertAlmostEqual(H[index[(0, 2, 1, 0)], index[(1, 1, 1, 0)]], -t * np.sqrt(2))
        self.assertEqual(H[index[(0, 1, 1, 1)], index[(1, 1, 1, 0)]], 0)

    def test_atomic_limit_and_chemical_shift(self):
        energy, state, basis, H = ed.exact_ground_state(2, Ntot=4, t=0, U=16.74, mu=0.3)
        self.assertAlmostEqual(energy, -1.2)
        for index, occupation in enumerate(basis):
            expected = 16.74 * sum(n * (n - 1) // 2 for n in occupation) - 1.2
            self.assertAlmostEqual(H[index, index], expected)
        H0, _ = ed.build_bose_hubbard_hamiltonian(2, Ntot=3, mu=0)
        Hmu, _ = ed.build_bose_hubbard_hamiltonian(2, Ntot=3, mu=0.4)
        np.testing.assert_allclose((Hmu - H0).toarray(), -1.2 * np.eye(H0.shape[0]), atol=1e-14)

    def test_single_state_sectors(self):
        for total, cutoff, expected in ((0, 4, 0), (8, 2, 4 * 16.74 - 0.5 * 8), (0, 0, 0)):
            energy, state, basis, H = ed.exact_ground_state(2, Ntot=total, Nmax=cutoff, mu=0.5)
            self.assertAlmostEqual(energy, expected)
            np.testing.assert_array_equal(state, [1])
            self.assertEqual(H.shape, (1, 1))
        energy, _, _, _ = ed.exact_ground_state(1, 1, Ntot=2, mu=0.5)
        self.assertAlmostEqual(energy, 16.74 - 1)

    def test_two_site_analytic_ground_state(self):
        U, t = 3.7, 0.8
        energy, state, basis, H = ed.exact_ground_state(2, 1, Ntot=2, Nmax=2, U=U, t=t)
        self.assertAlmostEqual(energy, (U - np.sqrt(U**2 + 16 * t**2)) / 2)
        np.testing.assert_allclose(H @ state, energy * state, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(state), 1)
        self.assertTrue(np.all(state > 0))

    def test_rectangular_ground_states(self):
        # One particle has separable open-chain energies. Periodic 3x2 with
        # unique bonds is degree three, giving a uniform ground state at -3t.
        for periodic, expected in ((False, -1 - np.sqrt(2)), (True, -3)):
            energy, state, _, H = ed.exact_ground_state(3, 2, Ntot=1, pbc=periodic)
            self.assertAlmostEqual(energy, expected)
            np.testing.assert_allclose(H @ state, energy * state, atol=1e-12)
        energy, state, _, H = ed.exact_ground_state(3, 2, Ntot=3, Nmax=2)
        self.assertAlmostEqual(energy, np.linalg.eigvalsh(H.toarray())[0])
        np.testing.assert_allclose(H @ state, energy * state, atol=1e-12)

    def test_one_dimensional_parity(self):
        spec = importlib.util.spec_from_file_location("legacy_ed", ROOT.parent / "BoseHubbardED.py")
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        for length, periodic in ((2, False), (3, True), (4, False)):
            with self.subTest(length=length, periodic=periodic):
                old_H, old_basis = legacy.build_bose_hubbard_hamiltonian(
                    length, length, 2, 0.9, 16.74 / 2, mu=0.1, pbc=periodic
                )
                new_H, new_basis = ed.build_bose_hubbard_hamiltonian(
                    length, 1, Ntot=length, Nmax=2, t=0.9, U=16.74, mu=0.1, pbc=periodic
                )
                self.assertEqual(new_basis, old_basis)
                np.testing.assert_allclose(new_H.toarray(), old_H, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
