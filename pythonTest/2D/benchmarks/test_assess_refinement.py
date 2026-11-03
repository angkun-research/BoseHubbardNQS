"""Point/precision policy and spatial/temporal covariance preservation."""

import unittest

import numpy as np

from assess_refinement import assess_point, curve_samples, split_half_drift, statistics


class RefinementTests(unittest.TestCase):
    def test_point_failure_within_two_sem_is_never_a_pass(self):
        check = assess_point(1.06, 1., .04, .05, .025)
        self.assertFalse(check["point_target_met"])
        self.assertFalse(check["precision_target_met"])
        self.assertTrue(check["within_two_conditional_sem_of_reference"])
        self.assertEqual(check["status"], "inconclusive_point_target_missed")
        self.assertFalse(check["meets_point_and_precision_targets"])
        precise_failure = assess_point(1.06, 1., .01, .05, .025)
        self.assertEqual(precise_failure["status"], "point_target_missed")
        self.assertFalse(precise_failure["meets_point_and_precision_targets"])

    def test_precision_required_even_if_point_passes_and_missing_is_not_zero(self):
        for sem in (.1, None):
            check = assess_point(1., 1., sem, .05, .025)
            self.assertTrue(check["point_target_met"])
            self.assertFalse(check["meets_point_and_precision_targets"])
        check = assess_point(1.01, 1., .02, .05, .025)
        self.assertTrue(check["meets_point_and_precision_targets"])
        self.assertTrue(assess_point(0., 0., 0., 0., 0.)["meets_point_and_precision_targets"])
        for sem in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                assess_point(1., 1., sem, .05, .025)

    def test_per_x_averaging_keeps_negative_y_covariance_and_directed_line_order(self):
        z = np.array([[1, -1], [1, -1], [-1, 1], [-1, 1]])
        values = np.ones((4, 2, 4, 4))
        values[:, :, :2, :] += z[..., None, None]
        values[:, :, 2:, :] -= z[..., None, None]
        density = values.reshape(4, 2, 16)
        factorial = density * (density - 1)
        pairs = np.array(list(np.ndindex(16, 16)))
        rho = np.zeros((4, 2, len(pairs)))
        anchor = 10
        for j in (8, 9, 10, 11):
            rho[..., anchor * 16 + j] = 20 + j + z
            if j != anchor:
                rho[..., j * 16 + anchor] = 999
        rho[..., anchor * 16 + anchor] = density[..., anchor]
        curves = curve_samples(density, factorial, rho, pairs, dict(Lx=4, Ly=4), anchor)
        self.assertEqual(curves["line_pairs"], [(10, 8), (10, 9), (10, 10), (10, 11)])
        np.testing.assert_array_equal(curves["anchor_line"][..., 0], 28 + z)
        for name in ("density_per_x", "factorial_per_x"):
            stats = statistics(curves[name], 2)
            np.testing.assert_array_equal(stats["mean"], np.ones(4))
            np.testing.assert_array_equal(stats["conditional_sem"], np.zeros(4))
        self.assertGreater(statistics(density[..., 0], 2)["conditional_sem"], 0)
        keep = np.arange(len(pairs)) != anchor * 16 + 8
        with self.assertRaisesRegex(ValueError, "missing directed"):
            curve_samples(density, factorial, rho[..., keep], pairs[keep], dict(Lx=4, Ly=4), anchor)
        with self.assertRaisesRegex(ValueError, "central"):
            curve_samples(density, factorial, rho, pairs, dict(Lx=4, Ly=4), 0)

    def test_split_half_paired_chain_drift_retains_covariance(self):
        offsets = np.array([0., 100., 200.])
        values = np.stack([offsets, offsets, offsets + 999, offsets + 2, offsets + 2])
        drift = split_half_drift(values)
        self.assertEqual(drift["last_minus_first"], 2)
        self.assertEqual(drift["paired_between_chain_conditional_sem"], 0)
        self.assertEqual(drift["discarded_middle_steps"], 1)
        self.assertEqual(split_half_drift(values[:1])["status"], "insufficient_time_steps")


if __name__ == "__main__":
    unittest.main()
