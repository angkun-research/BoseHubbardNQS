"""Composite covariance checks on synthetic, perfectly correlated measurements."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from compare_dmrg_sizes import ROOT, BOND_CONVENTION, HAMILTONIAN_CONVENTION, central_samples, frozen_summary
from observables import block_mean_statistics


class CompositeTests(unittest.TestCase):
    def setUp(self):
        self.settings = dict(Lx=4, Ly=4, U=8., t=1., mu=0.)
        edges = set()
        for y in range(4):
            for x in range(4):
                i = 4 * y + x
                if x < 3:
                    edges.add((i, i + 1))
                edges.add(tuple(sorted((i, 4 * ((y + 1) % 4) + x))))
        self.bonds = sorted(edges)
        self.pairs = np.array([(a, b) for i, j in self.bonds for a, b in ((i, j), (j, i))])
        self.sites = [4 * y + x for y in range(4) for x in (1, 2)]
        self.z = np.arange(8, dtype=float).reshape(4, 2)
        # Synthetic onsite/hopping measurements, not an occupation ensemble:
        # onsite=4z exactly cancels four incident hopping contributions z.
        self.density = np.ones((4, 2, 16))
        self.factorial = np.repeat(self.z[..., None], 16, axis=-1)
        self.rho = np.repeat(self.z[..., None], len(self.pairs), axis=-1)

    def values(self, pairs=None, rho=None):
        return central_samples(self.density, self.factorial, self.rho if rho is None else rho,
                               self.pairs if pairs is None else pairs, self.settings, self.bonds, self.sites)

    def test_composite_errors_keep_cancellation_and_unique_direction_means(self):
        values = self.values()
        np.testing.assert_array_equal(values["central_mean_local_energy_density"], 0)
        stats = block_mean_statistics(torch.from_numpy(values["central_mean_local_energy_density"]), 2)
        self.assertEqual(stats["block_standard_error"].item(), 0)
        self.assertEqual(stats["between_chain_standard_error"].item(), 0)
        self.assertGreater(np.std(self.factorial[..., 0]), 0)
        np.testing.assert_array_equal(values["central_NN_Gx"], self.z)
        np.testing.assert_array_equal(values["central_NN_Gy"], self.z)
        np.testing.assert_array_equal(values["central_mean_density"], 1)
        # Eight end sites have degree3, so E/site=(8*z)/16.
        np.testing.assert_array_equal(values["whole_lattice_energy_per_site"], self.z / 2)

    def test_directional_averaging_and_pair_order(self):
        rho = self.rho.copy()
        for k, (i, j) in enumerate(self.pairs):
            rho[..., k] += 2 if i // 4 == j // 4 else 4
        expected = self.values(rho=rho)
        np.testing.assert_array_equal(expected["central_NN_Gx"], self.z + 2)
        np.testing.assert_array_equal(expected["central_NN_Gy"], self.z + 4)
        actual = self.values(self.pairs[::-1], rho[..., ::-1].copy())
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_missing_one_directed_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing directed"):
            self.values(self.pairs[1:], self.rho[..., 1:])

    def test_raw_archive_roundtrip_validation_and_block_override(self):
        settings = {**self.settings, "Ntot": 16, "Nmax": 4, "pbc_x": False, "pbc_y": True}
        reference = {"settings": settings, "central_physical_sites": self.sites,
                     "central_mean_density": 1., "central_mean_local_energy_density": 0.,
                     "whole_lattice_energy_per_site": 0.,
                     "central_direction_resolved_NN_G": {d: {"G": 0.} for d in ("x", "y")}}
        # Physical occupation constraints hold. Synthetic rho values isolate the
        # archive/statistics plumbing; this fixture is not an equilibrium chain.
        energy = -2 * len(self.bonds) * self.z
        with tempfile.TemporaryDirectory(dir=ROOT / "benchmarks/results", prefix="test-strip-") as directory:
            raw_path, path = Path(directory) / "raw.npz", Path(directory) / "report.json"
            np.savez_compressed(raw_path, occupations=np.ones((4, 2, 16), dtype=np.int64),
                                density=self.density, onsite_factorial_moment=np.zeros_like(self.factorial),
                                physical_pairs=self.pairs, one_body_estimators=self.rho, local_energy=energy)
            report = {"status": "completed", "kind": "frozen_model_observables", "flattening": "i=y*Lx+x",
                      "axes": ["time", "walker", "physical_site"], "pair_axes": ["time", "walker", "selected_physical_pair"],
                      "checkpoint_metadata": {**settings, "flattening": "i=y*Lx+x",
                          "hamiltonian_convention": HAMILTONIAN_CONVENTION, "bond_convention": BOND_CONVENTION},
                      "checkpoint": "synthetic.pt", "checkpoint_sha256": "synthetic", "sampling": {},
                      "raw_samples_path": str(raw_path.relative_to(ROOT)), "one_body_pairs": self.pairs.tolist(),
                      "statistics": {"energy": {"time_steps": 4, "walkers": 2, "block_length": 2,
                                                   "all_samples_mean": energy.mean()}}}
            path.write_text(json.dumps(report))
            result = frozen_summary(path, reference, self.bonds, block_length=3)
            stats = result["statistics"]["central_NN_Gx"]
            self.assertEqual(stats["discarded_tail_steps"], 1)
            self.assertIsNone(stats["block_standard_error"])
            self.assertEqual(stats["mean"], self.z[:3].mean())
            self.assertEqual(stats["all_samples_mean"], self.z.mean())
            self.assertAlmostEqual(stats["conditional_sem"], .5)
            self.assertIn(str(raw_path.relative_to(ROOT)), result["input_sha256"])
            for section, key, value in (("checkpoint_metadata", "Ntot", 15),
                                        ("checkpoint_metadata", "pbc_x", True),
                                        ("checkpoint_metadata", "hamiltonian_convention", "wrong"),
                                        ("checkpoint_metadata", "bond_convention", "wrong"),
                                        (None, "axes", ["walker", "time", "physical_site"]),
                                        (None, "one_body_pairs", self.pairs[::-1].tolist())):
                invalid = copy.deepcopy(report)
                (invalid if section is None else invalid[section])[key] = value
                path.write_text(json.dumps(invalid))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    frozen_summary(path, reference, self.bonds)


if __name__ == "__main__":
    unittest.main()
