"""Independent small-system checks for fixed-number VMC and its updates."""

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
from exact_diagonalization import build_bose_hubbard_hamiltonian, generate_basis
from lattice import bonds_2d
from models import GENERAL_ARCHITECTURES, build_model


class LinearLogState(nn.Module):
    """Log psi = theta dot occupations, convenient for exact update math."""
    def __init__(self, Lx=2, Ly=1, Nmax=2, pbc=True):
        super().__init__()
        self.Lx, self.Ly, self.Nmax, self.pbc = Lx, Ly, Nmax, pbc
        self.n_sites = Lx * Ly
        self.theta = nn.Parameter(torch.linspace(-0.2, 0.3, self.n_sites, dtype=torch.float64))
        self.max_seen = 0
        self.grad_modes = []

    def forward(self, x):
        self.max_seen = max(self.max_seen, len(x))
        self.grad_modes.append(torch.is_grad_enabled())
        occupations = (x * torch.arange(self.Nmax + 1, device=x.device)).sum(-1)
        return occupations @ self.theta


class FactorialState(LinearLogState):
    def forward(self, x):
        occupations = (x * torch.arange(self.Nmax + 1, device=x.device)).sum(-1)
        return -0.5 * torch.lgamma(occupations + 1).sum(-1) + self.theta.sum() * 0


class SamplingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_initialization_and_sampler_invariants(self):
        model = LinearLogState(3, 2)
        initial = vmc.generate_initial_states(6, 5, 2, 32)
        samples, walkers, diagnostics = vmc.sample(model, initial, 7, 3, 5)
        self.assertEqual(samples.shape, (224, 6))
        self.assertEqual(samples.dtype, torch.long)
        self.assertFalse(samples.requires_grad)
        self.assertFalse(any(model.grad_modes))
        self.assertLessEqual(model.max_seen, 5)
        self.assertTrue(torch.all(samples.sum(1) == 5))
        self.assertTrue(torch.all((samples >= 0) & (samples <= 2)))
        self.assertTrue(torch.equal(walkers, samples[-32:]))
        self.assertEqual(diagnostics["attempts"], 224)
        self.assertLessEqual(diagnostics["moved"], diagnostics["accepted"])
        self.assertLessEqual(diagnostics["moved"], diagnostics["valid_proposals"])
        with self.assertRaisesRegex(ValueError, "integer"):
            vmc.sample(model, initial.double(), 1)
        with self.assertRaisesRegex(ValueError, "fixed-number"):
            vmc.sample(model, [[0] * 6, [1] * 6], 1)

    def test_exhaustive_proposal_balance(self):
        basis = torch.tensor(generate_basis(4, 2, 2))
        bonds = torch.tensor(bonds_2d(2, 2))
        draws = 2 * len(bonds)
        states = basis.repeat_interleave(draws, dim=0)
        choices = torch.arange(len(bonds)).repeat_interleave(2).repeat(len(basis))
        directions = torch.tensor([0, 1]).repeat(len(bonds) * len(basis))
        with patch.object(torch, "randint", side_effect=[choices, directions]):
            proposals, valid = vmc.propose_hops(states, bonds, 2)
        index = {tuple(state): i for i, state in enumerate(basis.tolist())}
        q = np.zeros((len(basis), len(basis)))
        for source, proposal in enumerate(proposals.tolist()):
            q[source // draws, index[tuple(proposal)]] += 1 / draws
        np.testing.assert_allclose(q.sum(1), 1)
        np.testing.assert_array_equal(q, q.T)
        self.assertTrue(np.any(np.diag(q) > 0))
        torch.testing.assert_close(valid, (proposals != states).any(1))
        # Independently add Metropolis acceptance and verify detailed balance
        # for a nonuniform target probability on the whole sector.
        probabilities = np.exp(np.arange(len(basis)) / 3)
        transitions = q * np.minimum(1, probabilities[None, :] / probabilities[:, None])
        flow = probabilities[:, None] * transitions
        np.testing.assert_allclose(flow, flow.T, atol=1e-14)

    def test_no_moves_and_zero_steps(self):
        for Lx, Ly, Nmax, total in ((1, 1, 4, 2), (2, 2, 2, 0), (2, 2, 2, 8), (2, 2, 0, 0)):
            model = LinearLogState(Lx, Ly, Nmax)
            initial = vmc.generate_initial_states(model.n_sites, total, Nmax, 3)
            samples, walkers, diagnostics = vmc.sample(model, initial, 4)
            self.assertEqual(diagnostics["accepted"], 12)
            self.assertEqual(diagnostics["moved"], 0)
            self.assertEqual(diagnostics["valid_proposals"], 0)
            torch.testing.assert_close(walkers, initial)
            expected = (0.5 * 4.2 * initial.double() * (initial - 1) - 0.3 * initial).sum(1)
            torch.testing.assert_close(vmc.local_energy(model, samples, U=4.2, mu=0.3), expected.repeat(4))
            empty, _, info = vmc.sample(model, initial, 0, burnin_steps=1)
            self.assertEqual(empty.shape, (0, model.n_sites))
            self.assertEqual(info["attempts"], 0)

    def test_cache_recomputed_after_parameter_change(self):
        model = LinearLogState(Nmax=1)
        initial = torch.tensor([[1, 0]])
        with torch.no_grad():
            model.theta.copy_(torch.tensor([2.0, 0.0]))
        def flip(states, bonds, cutoff):
            return states.flip(1), torch.ones(len(states), dtype=torch.bool)
        with patch.object(vmc, "propose_hops", side_effect=flip), patch.object(torch, "rand", return_value=torch.tensor([0.5])):
            _, walkers, info = vmc.sample(model, initial, 1)
            self.assertEqual(info["moved"], 0)
            with torch.no_grad():
                model.theta.copy_(torch.tensor([-2.0, 0.0]))
            _, walkers, info = vmc.sample(model, walkers, 1)
            self.assertEqual(info["moved"], 1)
            torch.testing.assert_close(walkers, torch.tensor([[0, 1]]))


class EnergyAndUpdateTests(unittest.TestCase):
    def test_nonfinite_connected_logs_are_not_masked(self):
        # -inf gives a finite zero hopping ratio; checking energies alone would
        # silently accept it. Base logs for [0,2] remain finite in every case.
        class InvalidConnectedState(LinearLogState):
            def forward(self, x):
                finite = super().forward(x)
                is_connected = x[:, 0, 1].bool()
                return torch.where(is_connected, torch.full_like(finite, invalid), finite)

        for invalid in (-float("inf"), float("inf"), float("nan")):
            for size in (1, 3):
                with self.subTest(invalid=invalid, size=size):
                    with self.assertRaisesRegex(RuntimeError, "non-finite"):
                        vmc.local_energy(InvalidConnectedState(), [[0, 2]], evaluation_batch_size=size)

    def test_shared_physical_validation(self):
        for name in ("t", "U", "mu"):
            for value in (10**1000, 1 + 0j, float("inf"), float("nan"), True):
                with self.subTest(name=name, value=repr(value)):
                    with self.assertRaisesRegex(ValueError, name):
                        vmc.physical_parameters(**{name: value})
        self.assertEqual(vmc.physical_parameters(t=0, U=0, mu=-1), (0., 0., -1.))

    def test_local_energy_against_exact_operator(self):
        torch.manual_seed(2)
        for periodic in (False, True):
            model = build_model("FCNet", 3, 2, Nmax=2, pbc=periodic, hidden_dim=5).double()
            H, basis = build_bose_hubbard_hamiltonian(3, 2, Ntot=2, Nmax=2, pbc=periodic, t=0.6, U=3.1, mu=0.4)
            states = torch.tensor(basis)
            with torch.no_grad():
                amplitudes = model(vmc.occupancy_tensor_to_onehot(states, 2, torch.float64)).exp().numpy()
            expected = (H @ amplitudes) / amplitudes
            actual = vmc.local_energy(model, states, t=0.6, U=3.1, mu=0.4, evaluation_batch_size=3)
            np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)
            self.assertFalse(actual.requires_grad)
        bounded = LinearLogState(3, 2)
        vmc.local_energy(bounded, states, evaluation_batch_size=2)
        self.assertLessEqual(bounded.max_seen, 2)
        self.assertFalse(any(bounded.grad_modes))

    def test_noninteracting_factorial_ground_state(self):
        # With cutoff >= Ntot, psi(n) proportional to 1/sqrt(prod(n_i!))
        # is the uniform-orbital condensate on this degree-three periodic grid.
        model = FactorialState(3, 2, Nmax=3)
        states = torch.tensor(generate_basis(6, 3, 3))
        energies = vmc.local_energy(model, states, t=0.7, U=0, mu=0.2, evaluation_batch_size=4)
        torch.testing.assert_close(energies, torch.full_like(energies, -3 * 3 * 0.7 - 0.2 * 3))

    def test_native_gradient_and_chunk_equivalence(self):
        states = torch.tensor([[0, 2], [1, 1], [2, 0], [1, 1]])
        energies = torch.tensor([1.2, -0.3, 0.7, 2.1], dtype=torch.float64, requires_grad=True)
        # For log psi=theta dot n, the gradient is exactly this covariance.
        expected_gradient = 2 * ((energies.detach() - energies.detach().mean())[:, None] * states).mean(0)
        for size in (1, 3, 8):
            model = LinearLogState()
            before = model.theta.detach().clone()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.07)
            before_logs = states.double() @ before
            expected_loss = 2 * ((energies.detach() - energies.detach().mean()) * before_logs).mean()
            loss = vmc.native_vmc_update(model, states, energies, optimizer, size)
            self.assertIsInstance(loss, float)
            self.assertAlmostEqual(loss, expected_loss.item())
            torch.testing.assert_close(model.theta.grad, expected_gradient)
            torch.testing.assert_close(model.theta, before - 0.07 * expected_gradient)
            self.assertLessEqual(model.max_seen, size)
        self.assertIsNone(energies.grad)

    def test_centered_sr_against_numpy_and_optimizer(self):
        states = torch.tensor([[0, 2], [1, 1], [2, 0], [1, 1]])
        energies = np.array([1.2, -0.3, 0.7, 2.1])
        J = states.numpy().astype(float)
        centered = J - J.mean(0)
        tau = 0.13
        expected = np.linalg.solve(centered.T @ centered / 4 + tau * np.eye(2),
                                   centered.T @ (energies - energies.mean()) / 4)
        direction = vmc.sr_gradient(torch.tensor(J), torch.tensor(energies), tau)
        np.testing.assert_allclose(direction.numpy(), expected, atol=1e-14)
        shifted = vmc.sr_gradient(torch.tensor(J + [4, -7]), torch.tensor(energies + 12), tau)
        torch.testing.assert_close(direction, shifted)
        model = LinearLogState()
        before = model.theta.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.07)
        jacobian = vmc.compute_log_psi_jacobian(model, states)
        torch.testing.assert_close(jacobian, states.double())
        vmc.sr_update(model, states, energies, optimizer, tau)
        torch.testing.assert_close(model.theta, before - 0.07 * direction)

    def test_sr_guards_before_forward_and_all_architectures(self):
        model = LinearLogState()
        states = torch.tensor([[1, 1], [0, 2]])
        with patch.object(model, "forward", side_effect=AssertionError("forward called")):
            with self.assertRaisesRegex(ValueError, "parameter"):
                vmc.compute_log_psi_jacobian(model, states, max_sr_parameters=1)
            with self.assertRaisesRegex(ValueError, "workspace"):
                vmc.compute_log_psi_jacobian(model, states, max_sr_bytes=1)
        with patch.object(torch.linalg, "solve", side_effect=AssertionError("solve called")):
            with self.assertRaisesRegex(ValueError, "workspace"):
                vmc.sr_gradient(torch.ones(2, 2), [0.1, 0.2], max_sr_bytes=1)
        for arch in GENERAL_ARCHITECTURES:
            settings = {} if arch == "JastrowNet" else {"n_hidden" if arch == "RBMNet" else "hidden_dim": 4 if arch == "Conv2DTransformerNet" else 2}
            model = build_model(arch, 2, 1, Nmax=2, **settings).double().eval()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            direction = vmc.sr_update(model, states, [0.3, -0.2], optimizer)
            self.assertTrue(torch.isfinite(direction).all())
        for options in ({"t": -1}, {"U": -1}, {"mu": float("nan")}):
            with self.assertRaises(ValueError):
                vmc.local_energy(LinearLogState(), states, **options)


if __name__ == "__main__":
    unittest.main()
