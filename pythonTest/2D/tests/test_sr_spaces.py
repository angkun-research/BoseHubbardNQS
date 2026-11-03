"""Independent primal/dual ridge checks and allocation guards for SR+SGD."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vmc
from models import FCNet


class LinearState(nn.Module):
    def __init__(self):
        super().__init__()
        self.Lx, self.Ly, self.n_sites, self.Nmax, self.pbc = 2, 1, 2, 2, True
        self.theta = nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float64))

    def forward(self, x):
        occupations = (x * torch.arange(3, dtype=x.dtype)).sum(-1)
        return occupations @ self.theta


class SRSpaceTests(unittest.TestCase):
    def test_primal_dual_and_numpy_in_both_size_regimes(self):
        generator = np.random.default_rng(123)
        for M, P in ((4, 11), (11, 4), (5, 5), (1, 7)):
            for rank_deficient in (False, True):
                J = generator.normal(size=(M, P))
                if rank_deficient:
                    J[:, -1] = J[:, 0]
                E = generator.normal(size=M)
                Jc = J - J.mean(0)
                tau = 0.07
                expected = np.linalg.solve(Jc.T @ Jc / M + tau * np.eye(P), Jc.T @ (E - E.mean()) / M)
                for solver in ("parameter", "sample", "auto"):
                    actual, diagnostics = vmc.sr_gradient(torch.tensor(J), E, tau, solver=solver, return_diagnostics=True)
                    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-13, rtol=2e-12)
                    if solver == "auto":
                        self.assertEqual(diagnostics["solve_dimension"], min(M, P))
                        self.assertEqual(diagnostics["solver"], "parameter" if P <= M else "sample")
        J = torch.full((4, 12), 3.0)
        for solver in ("parameter", "sample", "auto"):
            direction = vmc.sr_gradient(J, [1, 2, 3, 4], solver=solver)
            torch.testing.assert_close(direction, torch.zeros(12, dtype=torch.float64), atol=0, rtol=0)

    def test_actual_sgd_update_and_centering_invariance(self):
        states = torch.tensor([[0, 2], [1, 1], [2, 0], [1, 1]])
        energies = torch.tensor([1.2, -0.3, 0.7, 2.1], dtype=torch.float64)
        J = states.double()
        centered = J - J.mean(0)
        expected = torch.linalg.solve(centered.T @ centered / 4 + 0.13 * torch.eye(2, dtype=torch.float64),
                                      centered.T @ (energies - energies.mean()) / 4)
        for solver in ("sample", "parameter", "auto"):
            model = LinearState()
            before = model.theta.detach().clone()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.07)
            direction, diagnostics = vmc.sr_update(model, states, energies, optimizer, 0.13,
                                                  solver=solver, return_diagnostics=True)
            torch.testing.assert_close(direction, expected, atol=1e-14, rtol=1e-12)
            torch.testing.assert_close(model.theta, before - 0.07 * expected)
            shifted = vmc.sr_gradient(J + torch.tensor([8., -4.]), energies + 11, 0.13, solver=solver)
            torch.testing.assert_close(shifted, expected)

    def test_tau_and_budget_fail_before_forward_or_workspace_allocation(self):
        model = LinearState()
        states = torch.tensor([[1, 1], [2, 0]])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with patch.object(model, "forward", side_effect=AssertionError("forward called")):
            for tau in (0, -1, float("nan"), float("inf"), True, 1j, "0.1"):
                with self.subTest(tau=tau), self.assertRaises(ValueError):
                    vmc.sr_update(model, states, [1, 2], optimizer, tau)
            with self.assertRaisesRegex(ValueError, "SGD"):
                vmc.sr_update(model, states, [1, 2], torch.optim.Adam(model.parameters()))
            with patch.object(torch, "empty", side_effect=AssertionError("Jacobian allocated")):
                with self.assertRaisesRegex(ValueError, "workspace"):
                    vmc.compute_log_psi_jacobian(model, states, max_sr_bytes=1)
        with patch.object(torch.linalg, "solve", side_effect=AssertionError("solve called")):
            for solver in ("sample", "parameter"):
                with self.assertRaisesRegex(ValueError, "workspace"):
                    vmc.sr_gradient(torch.ones(2, 3), [1, 2], max_sr_bytes=1, solver=solver)
            with self.assertRaisesRegex(ValueError, "solver"):
                vmc.sr_gradient(torch.ones(2, 3), [1, 2], solver="invalid")

    def test_parameter_cap_is_not_the_old_2000_limit_and_mp_still_guarded(self):
        model = FCNet(4, 4, hidden_dim=32)
        budget = vmc.check_sr_budget(model, 8)
        self.assertGreater(budget["parameters"], 2000)
        self.assertEqual(budget["solver"], "sample")
        self.assertEqual(budget["solve_dimension"], 8)
        with self.assertRaisesRegex(ValueError, "workspace"):
            vmc.check_sr_budget(model, 1_000_000)
        with self.assertRaisesRegex(ValueError, "parameter count"):
            vmc.check_sr_budget(model, 8, max_sr_parameters=2000)
        dual = vmc.check_sr_budget(model, 8, solver="sample")
        with self.assertRaisesRegex(ValueError, "workspace"):
            vmc.check_sr_budget(model, 8, solver="parameter")
        primal = vmc.check_sr_budget(model, 8, solver="parameter", max_sr_bytes=2 * 1024**3)
        self.assertGreater(primal["estimated_bytes"], dual["estimated_bytes"])


if __name__ == "__main__":
    unittest.main()
