"""Exact table derivatives and probability-weighted energy-gradient checks."""

import unittest

import numpy as np
import torch
from torch.nn import functional as F

from check_exact_jastrow_sr import table_features, flat_parameters, exact_state, weighted_sr
from exact_diagonalization import exact_ground_state
from models import JastrowNet


class ExactJastrowTests(unittest.TestCase):
    def test_both_features_match_random_forwards_and_scalar_autograd(self):
        states = torch.tensor([[2, 0, 1, 1], [0, 2, 1, 1], [1, 1, 0, 2], [1, 1, 2, 0]])
        for mode in ("displacement", "reflection_orbit"):
            torch.manual_seed(89)
            model = JastrowNet(4, 1, 2, pbc_x=False, pbc_y=True, pair_sharing=mode).double()
            with torch.no_grad():
                for p in model.parameters():
                    p.normal_(std=.2)
            features = table_features(model, states.numpy())
            logs = model(F.one_hot(states, 3).double())
            np.testing.assert_allclose(features @ flat_parameters(model), logs.detach().numpy(), atol=1e-13)
            for k in range(len(states)):
                gradients = torch.autograd.grad(logs[k], tuple(model.parameters()), retain_graph=True)
                actual = torch.cat([g.flatten() for g in gradients]).numpy()
                np.testing.assert_array_equal(features[k], actual)
            if mode == "reflection_orbit":
                cls = model.pair_orbits.tolist().index([0, 1, 0])
                offset = model.onsite.numel() + cls * 9
                self.assertEqual(features[0, offset + 2 * 3], 1.)
                self.assertEqual(features[0, offset + 2], 0.)

    def test_exact_probability_force_is_half_normalized_energy_gradient(self):
        ed_energy, ed, basis, H = exact_ground_state(3, 2, 2, 2, 1., 3., .2, pbc_x=False, pbc_y=True)
        for mode in ("displacement", "reflection_orbit"):
            torch.manual_seed(91)
            model = JastrowNet(3, 2, 2, pbc_x=False, pbc_y=True, pair_sharing=mode).double()
            with torch.no_grad():
                for p in model.parameters():
                    p.normal_(std=.1)
            features = table_features(model, basis)
            probability, local, metrics = exact_state(flat_parameters(model), features, H, ed)
            direction, force, centered = weighted_sr(features, probability, local, metrics["energy"], .02)
            logs = model(F.one_hot(torch.tensor(basis), 3).double())
            psi = (logs - .5 * torch.logsumexp(2 * logs, 0)).exp()
            energy = psi @ (torch.tensor(H.toarray()) @ psi)
            gradients = torch.autograd.grad(energy, tuple(model.parameters()))
            actual = torch.cat([g.flatten() for g in gradients]).numpy()
            np.testing.assert_allclose(actual, 2 * force, atol=1e-12, rtol=1e-12)
            covariance = centered.T @ (probability[:, None] * centered) + .02 * np.eye(features.shape[1])
            np.testing.assert_allclose(covariance @ direction, force, atol=1e-12, rtol=1e-12)
            self.assertAlmostEqual(probability.sum(), 1.)
            self.assertGreaterEqual(metrics["energy"], ed_energy - 1e-12)


if __name__ == "__main__":
    unittest.main()
