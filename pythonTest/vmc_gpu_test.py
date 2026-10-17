#!/usr/bin/env python3
"""Bose-Hubbard tensor VMC runner and small regression suite (--self-test)."""

import argparse
import copy
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from NeuralNetworkClass import FCNet, JastrowNet, Conv1DNet, RBMNet
from utils import pick_best_device
from vmc_utils_gpu import (
    GlobalSampler_gpu, Obtain_Sampling_batch_gpu, generate_initial_state,
    local_energy_on_the_fly_gpu, occupancy_tensor_to_onehot,
    compute_log_psi_jacobian_vmap, sr_gradient, sr_update, sr_update_optimizer,
    native_vmc_update,
)


ARCH_BUILDERS = {
    "FCNet": lambda L, Ch, hd: FCNet(n_sites=L, channels=Ch, hidden_dim=hd),
    "JastrowNet": lambda L, Ch, hd: JastrowNet(n_sites=L, channels=Ch),
    "Conv1DNet": lambda L, Ch, hd: Conv1DNet(n_sites=L, channels=Ch, hidden_dim=hd),
    "RBMNet": lambda L, Ch, hd: RBMNet(n_sites=L, channels=Ch, n_hidden=hd),
}


class _TinyLogModel(torch.nn.Module):
    """Small nonlinear real log ansatz, also recording autograd context."""

    def __init__(self, L, channels, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(
            -0.3, 0.4, L * channels, device=device, dtype=torch.float64
        ).reshape(L, channels))
        self.flags = []

    def forward(self, x):
        self.flags.append(torch.is_grad_enabled())
        return torch.tanh((x * self.weight).sum((1, 2)))


class GPUVMCRegressionTests(unittest.TestCase):
    device = torch.device("cpu")

    def setUp(self):
        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)

    def test_local_energies_scalar_and_ed(self):
        from BoseHubbardED import build_bose_hubbard_hamiltonian
        from vmc_utils import local_energy_on_the_fly

        for L, N, cutoff in [(2, 2, 2), (3, 3, 2), (3, 0, 2), (3, 6, 2)]:
            for pbc in (False, True):
                with self.subTest(L=L, N=N, pbc=pbc):
                    H, basis = build_bose_hubbard_hamiltonian(L, N, cutoff, 0.7, 1.3, 0.2, pbc)
                    model = _TinyLogModel(L, cutoff + 1, self.device)
                    states = torch.tensor(basis, device=self.device)
                    logs = model(occupancy_tensor_to_onehot(states, cutoff, torch.float64)).detach()
                    amplitudes = logs.exp().cpu().numpy()
                    expected = torch.tensor(H @ amplitudes / amplitudes, device=self.device)
                    scalar = torch.stack([local_energy_on_the_fly(
                        s, model, L, cutoff, 0.7, 1.3, 0.2, pbc, self.device
                    ) for s in basis]).detach()
                    for chunk in (None, 2):
                        actual = local_energy_on_the_fly_gpu(
                            states, model, L, cutoff, 0.7, 1.3, 0.2, pbc,
                            evaluation_batch_size=chunk, logpsi=logs,
                        )
                        torch.testing.assert_close(actual, expected)
                        torch.testing.assert_close(actual, scalar)
                        self.assertFalse(actual.requires_grad)
                    one = local_energy_on_the_fly_gpu(states[0], model, L, cutoff, 0.7, 1.3)
                    self.assertEqual(one.ndim, 0)
        # Scalar _bonds_1d has no L=1 self bond; ED's PBC builder differs here.
        model = _TinyLogModel(1, 3, self.device)
        self.assertAlmostEqual(local_energy_on_the_fly_gpu(
            [2], model, 1, 2, 1.0, 1.3, 0.2, True
        ).item(), 2.2)

    def test_proposals_invariants_and_symmetry(self):
        from BoseHubbardED import generate_basis

        for L in (1, 2, 3):
            for pbc in (False, True):
                basis = generate_basis(L, L, 2)
                states = torch.tensor(basis, device=self.device)
                original = states.clone()
                counts = torch.zeros((len(basis), len(basis)), dtype=torch.long)
                index = {s: i for i, s in enumerate(basis)}
                n_bonds = L if pbc and L > 1 else L - 1
                # Enumerate every equally likely bond/direction, including invalid hops.
                for bond in range(max(n_bonds, 1)):
                    for direction in (0, 1):
                        draws = [torch.full((len(states),), value, device=self.device)
                                 for value in (bond, direction)]
                        with patch("vmc_utils_gpu.torch.randint", side_effect=draws):
                            proposed = GlobalSampler_gpu(states, L, 2, pbc)
                        self.assertEqual(proposed.dtype, torch.long)
                        torch.testing.assert_close(proposed.sum(1), states.sum(1))
                        self.assertTrue(((proposed >= 0) & (proposed <= 2)).all())
                        diff = (proposed - states).abs()
                        self.assertTrue(((diff.sum(1) == 0) | (diff.sum(1) == 2)).all())
                        for i, row in enumerate(proposed.cpu().tolist()):
                            counts[i, index[tuple(row)]] += 1
                torch.testing.assert_close(counts, counts.T)
                torch.testing.assert_close(states, original)

    def test_sampling_order_cache_burnin_and_gradients(self):
        model = _TinyLogModel(3, 3, self.device)
        with torch.no_grad():
            model.weight.zero_()  # All cyclic proposals accepted.
        initial = torch.tensor([[2, 1, 0], [0, 2, 1]], device=self.device)

        def rotate(states, **kwargs):
            self.assertFalse(torch.is_grad_enabled())
            return states.roll(1, 1)

        logs, energies, final, samples = Obtain_Sampling_batch_gpu(
            model, 3, 2, initial, 4, Sampler=rotate, device="ignored"
        )
        expected = torch.stack([initial.roll(i, 1) for i in range(1, 5)]).reshape(-1, 3)
        torch.testing.assert_close(samples, expected)
        torch.testing.assert_close(final, expected[-2:])
        self.assertEqual(logs.shape, (8,))
        self.assertEqual(energies.shape, (8,))
        self.assertEqual(samples.dtype, torch.long)
        self.assertEqual(samples.device, next(model.parameters()).device)
        self.assertTrue(logs.requires_grad)
        self.assertFalse(energies.requires_grad)
        # Initial + four proposals + final logs + connected energies, no base reevaluation.
        self.assertEqual(model.flags, [False] * 5 + [True, False])
        logs.sum().backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all())
        for burnin, steps in ((True, 4), (False, 0)):
            empty_logs, empty_e, walkers, empty_s = Obtain_Sampling_batch_gpu(
                model, 3, 2, initial.tolist(), steps, burnin=burnin, Sampler=rotate
            )
            self.assertEqual(empty_logs.shape, (0,))
            self.assertEqual(empty_e.shape, (0,))
            self.assertEqual(empty_s.shape, (0, 3))
            torch.testing.assert_close(walkers, initial.roll(steps, 1))
        chunked = Obtain_Sampling_batch_gpu(
            model, 3, 2, initial, 4, Sampler=rotate, evaluation_batch_size=2
        )
        torch.testing.assert_close(chunked[0], logs)
        torch.testing.assert_close(chunked[1], energies)
        for bad_chunk in (0, -1):
            with self.assertRaises(ValueError):
                Obtain_Sampling_batch_gpu(model, 3, 2, initial, 0, evaluation_batch_size=bad_chunk)
        # Rejected proposals must leave the cached current log amplitude intact.
        rejecting = _TinyLogModel(2, 3, self.device)
        with torch.no_grad():
            rejecting.weight.zero_()
            rejecting.weight[0, 2], rejecting.weight[1, 2] = 1, -1
        start = torch.tensor([[2, 0]], device=self.device)
        with patch("vmc_utils_gpu.torch.rand", return_value=torch.tensor([0.5], device=self.device)):
            rejected = Obtain_Sampling_batch_gpu(rejecting, 2, 2, start, 3, Sampler=rotate)
        torch.testing.assert_close(rejected[2], start)
        torch.testing.assert_close(rejected[3], start.expand(3, -1))
        torch.testing.assert_close(rejected[0], torch.tanh(rejecting.weight.new_ones(3)))

    def test_sr_and_native_match_scalar_math(self):
        from vmc_utils import sr_gradient as scalar_sr_gradient

        states = torch.tensor([[2, 0], [1, 1], [0, 2]], device=self.device)
        model = _TinyLogModel(2, 3, self.device)
        x = occupancy_tensor_to_onehot(states, 2, torch.float64)
        e = torch.tensor([-1.2, 0.7, 2.3], device=self.device, dtype=torch.float64)
        J = compute_log_psi_jacobian_vmap(model, x, evaluation_batch_size=2)
        manual = torch.stack([torch.autograd.grad(model(row[None]).sum(), model.weight)[0].flatten()
                              for row in x])
        torch.testing.assert_close(J, manual)
        centered = e - e.mean()
        expected = torch.linalg.solve(J.T @ J + 0.1 * torch.eye(
            J.shape[1], device=self.device, dtype=J.dtype
        ), J.T @ centered)
        torch.testing.assert_close(sr_gradient(J, centered, 0.1), expected)
        torch.testing.assert_close(sr_gradient(J, centered, 0.1).float(), scalar_sr_gradient(J, centered, 0.1))
        # NQSpy's optional solve dtype must preserve output dtype and device.
        for dtype in (None, torch.float32):
            direction = sr_gradient(J, centered, 0.1, solve_dtype=dtype)
            self.assertEqual(direction.dtype, J.dtype)
            self.assertEqual(direction.device, J.device)
            torch.testing.assert_close(direction, expected, rtol=1e-5, atol=1e-6)
        before = model.weight.detach().clone()
        optimized = copy.deepcopy(model)
        sr_update(model, states, e, 2, 2, 0.03, 0.1, device="ignored")
        torch.testing.assert_close(model.weight, before - 0.03 * expected.reshape_as(before))
        optimizer = torch.optim.SGD(optimized.parameters(), lr=0.03)
        sr_update_optimizer(optimized, states, e, 2, 2, optimizer, 0.1)
        torch.testing.assert_close(optimized.weight, model.weight)
        reference = copy.deepcopy(model)
        loss = 2 * (centered * reference(x)).mean()
        gradient, = torch.autograd.grad(loss, reference.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        logs = model(x)
        native_vmc_update(model, states, e.requires_grad_(), 2, 2, optimizer, logpsis=logs)
        torch.testing.assert_close(model.weight, reference.weight - 0.02 * gradient)
        self.assertIsNone(e.grad)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="FCNet", choices=list(ARCH_BUILDERS))
    for name, default in (("L", 8), ("Nmax", 8), ("Ntot", 8), ("hidden_dim", 64),
                          ("epochs", 100), ("n_walkers", 64), ("n_steps", 32),
                          ("burnin_steps", 32), ("eval_every", 10), ("seed", 123)):
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("t", 1.0), ("U", 2.0), ("mu", 0.0), ("lr", 5e-3), ("tau", 1e-3)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--pbc", action="store_true", default=True)
    parser.add_argument("--no-pbc", dest="pbc", action="store_false")
    parser.add_argument("--update_method", choices=["sr", "native"], default="sr")
    parser.add_argument("--native_opt", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--native_lr", type=float)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda[:index], or mps")
    parser.add_argument("--evaluation_batch_size", "--evaluation-batch-size", type=int)
    parser.add_argument("--skip-ed", "--skip_ed", action="store_true", help="avoid dense ED allocation")
    parser.add_argument("--self-test", action="store_true", help="run regression tests and exit")
    args = parser.parse_args()
    device = pick_best_device() if args.device == "auto" else torch.device(args.device)
    if args.self_test:
        GPUVMCRegressionTests.device = device
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(GPUVMCRegressionTests)
        )
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if (args.L < 1 or args.Nmax < 0 or not 0 <= args.Ntot <= args.L * args.Nmax
            or min(args.n_walkers, args.n_steps, args.eval_every) <= 0
            or min(args.epochs, args.burnin_steps) < 0
            or (args.evaluation_batch_size is not None and args.evaluation_batch_size <= 0)):
        parser.error("invalid lattice, particle count, sampling count, or evaluation chunk size")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Using device: {device}; seed={args.seed}")
    model = ARCH_BUILDERS[args.arch](args.L, args.Nmax + 1, args.hidden_dim).to(device)
    optimizer = None
    if args.update_method == "native":
        lr = args.native_lr if args.native_lr is not None else args.lr
        optimizer = (torch.optim.Adam(model.parameters(), lr=lr) if args.native_opt == "adam"
                     else torch.optim.SGD(model.parameters(), lr=lr))
    ed_energy = None
    if not args.skip_ed:
        from BoseHubbardED import exact_ground_state

        if args.L == 1 or args.Ntot in (0, args.L * args.Nmax):
            # A single basis vector: eigsh(k=1) cannot handle dimension one.
            state = generate_initial_state(args.L, args.Ntot, args.Nmax)
            ed_energy = sum(args.U * n * (n - 1) - args.mu * n for n in state)
        else:
            ed_energy, _, basis, _ = exact_ground_state(
                args.L, args.Ntot, args.Nmax, args.t, args.U, args.mu, args.pbc
            )
            print(f"ED Hilbert-space dimension: {len(basis)}")
        print(f"ED ground-state energy: {ed_energy:.10f}")
    states = [generate_initial_state(args.L, args.Ntot, args.Nmax) for _ in range(args.n_walkers)]
    sampling = dict(psi=model, L=args.L, Nmax=args.Nmax, t=args.t, U=args.U,
                    mu=args.mu, pbc=args.pbc, evaluation_batch_size=args.evaluation_batch_size)
    for epoch in range(1, args.epochs + 1):
        _, _, states, _ = Obtain_Sampling_batch_gpu(
            initial_states=states, n_steps=args.burnin_steps, burnin=True, **sampling
        )
        logs, energies, states, samples = Obtain_Sampling_batch_gpu(
            initial_states=states, n_steps=args.n_steps,
            print_rate=(epoch % args.eval_every == 0), **sampling
        )
        mean, std = energies.mean().item(), energies.std(unbiased=False).item()
        if args.update_method == "sr":
            del logs  # SR builds its own Jacobian; release the final amplitude graph.
            sr_update(model, samples, energies, args.L, args.Nmax, args.lr, args.tau,
                      evaluation_batch_size=args.evaluation_batch_size)
            info = "sr"
        else:
            loss = native_vmc_update(model, samples, energies, args.L, args.Nmax,
                                     optimizer, logpsis=logs)
            info = f"native_loss={loss:.4e}"
        if epoch == 1 or epoch % args.eval_every == 0:
            delta = "" if ed_energy is None else f"  delta_to_ED={mean - ed_energy:.4e}"
            print(f"epoch={epoch:4d}  E_vmc={mean:.8f}  sigma={std:.4e}{delta}  {info}")


if __name__ == "__main__":
    main()
