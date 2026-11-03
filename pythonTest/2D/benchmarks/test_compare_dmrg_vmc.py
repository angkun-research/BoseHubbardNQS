"""Small comparisons; no model loading, sampling or DMRG runs."""

import copy
import unittest

import numpy as np

from compare_dmrg_vmc import BOND_CONVENTION, HAMILTONIAN_CONVENTION, PARAMETERS, compare_report, statistic


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.settings = dict(Lx=2, Ly=1, Ntot=1, Nmax=2, t=1., U=16.74, mu=0., pbc_x=False, pbc_y=True)
        self.report = {
            "kind": "frozen_model_observables", "status": "completed", "flattening": "i=y*Lx+x",
            "checkpoint_metadata": {**self.settings, "flattening": "i=y*Lx+x",
                                    "hamiltonian_convention": HAMILTONIAN_CONVENTION,
                                    "bond_convention": BOND_CONVENTION},
            "checkpoint": "test.pt", "checkpoint_sha256": "test", "sampling": {},
            "one_body_pairs": [[1, 0], [0, 1]],
            "statistics": {
                "energy": {"mean": -1., "all_samples_mean": -2., "block_standard_error": .2, "between_chain_standard_error": .4},
                "density": {"mean": [.4, .6], "block_standard_error": [.1, .3], "between_chain_standard_error": [.2, .1]},
                "one_body": {"mean": [.2, .5], "block_standard_error": [.3, .1], "between_chain_standard_error": [.1, .4]},
            },
        }
        self.density = np.array([.5, .5])
        self.rho = np.array([[.5, .1 + .02j], [.1 - .02j, .5]])

    def compare(self, report=None):
        return compare_report(self.report if report is None else report, self.settings, self.density, self.rho, -1.2)

    def test_index_order_signed_differences_and_error_units(self):
        result = self.compare()
        self.assertAlmostEqual(result["energy_error_per_site"], .1)
        self.assertEqual(result["conditional_energy_sem_per_site"], .2)
        self.assertEqual(result["energy_per_site"], -.5)
        self.assertEqual(result["all_samples_energy_per_site"], -1.)
        self.assertEqual([row["conditional_sem"] for row in result["density"]], [.2, .3])
        self.assertEqual([(row["i"], row["j"]) for row in result["rho"]], [(1, 0), (0, 1)])
        self.assertAlmostEqual(result["rho"][0]["difference_imag"], .02)
        self.assertAlmostEqual(result["rho"][1]["difference_real"], .4)
        self.assertEqual([row["conditional_sem_real"] for row in result["rho"]], [.3, .4])

    def test_every_parameter_and_both_index_declarations_must_match(self):
        for key in PARAMETERS:
            report = copy.deepcopy(self.report)
            report["checkpoint_metadata"][key] = not self.settings[key] if isinstance(self.settings[key], bool) else self.settings[key] + 1
            with self.subTest(parameter=key), self.assertRaisesRegex(ValueError, key):
                self.compare(report)
        for key in ("report", "checkpoint_metadata"):
            report = copy.deepcopy(self.report)
            (report if key == "report" else report[key])["flattening"] = "i=x*Ly+y"
            with self.assertRaisesRegex(ValueError, "physical index"):
                self.compare(report)

    def test_convention_mismatches_and_missing_declarations_rejected(self):
        for key, wrong in (("hamiltonian_convention", "U*sum(n*(n-1))"),
                           ("bond_convention", "parallel_bonds_on_short_periodic_axes")):
            for value in (wrong, None):
                report = copy.deepcopy(self.report)
                if value is None:
                    del report["checkpoint_metadata"][key]
                else:
                    report["checkpoint_metadata"][key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    self.compare(report)

    def test_missing_errors_and_reject_invalid_statistics_or_pairs(self):
        _, error = statistic({"mean": [1, 2]}, (2,))
        self.assertIsNone(error)
        _, error = statistic({"mean": 1., "between_chain_standard_error": .3}, ())
        self.assertEqual(error, .3)
        for stats in ({"mean": [1, 2], "block_standard_error": .1},
                      {"mean": [1, float("nan")]}, {"mean": [1, 2], "between_chain_standard_error": [-1, 0]}):
            with self.assertRaises(ValueError):
                statistic(stats, (2,))
        for pairs in ([[0, 1], [0, 1]], [[0, 2]], [[True, 0]]):
            report = copy.deepcopy(self.report)
            report["one_body_pairs"] = pairs
            with self.assertRaises(ValueError):
                self.compare(report)


if __name__ == "__main__":
    unittest.main()
