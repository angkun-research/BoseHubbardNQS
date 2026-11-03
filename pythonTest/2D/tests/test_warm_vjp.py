"""Batched-VJP reference checks, strict inference warm-starts and schedules."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exact_diagonalization import generate_basis
from models import GENERAL_ARCHITECTURES, Conv2DNet, build_model, save_model
import train
import vmc


def independent_jacobian(model, states):
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    rows = []
    for state in states:
        x = F.one_hot(state[None], model.Nmax + 1).to(dtype=parameters[0].dtype)
        value = model(x).reshape(())
        gradients = torch.autograd.grad(value, parameters, allow_unused=True)
        rows.append(torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                               for p, g in zip(parameters, gradients)]).double())
    return torch.stack(rows)


class BatchedVJPTests(unittest.TestCase):
    def test_all_five_jacobians_and_sgd_updates_with_unused_parameters(self):
        torch.manual_seed(17)
        states = torch.tensor(generate_basis(6, 2, 2)[:5])
        energies = torch.tensor([0.3, -1., 0.8, 0.1, 1.3], dtype=torch.float64)
        for arch in GENERAL_ARCHITECTURES:
            for px, py in ((False, True), (True, False)):
                settings = {} if arch == "JastrowNet" else {"n_hidden" if arch == "RBMNet" else "hidden_dim": 4}
                model = build_model(arch, 3, 2, Nmax=2, pbc_x=px, pbc_y=py, **settings).double().eval()
                model.register_parameter("unused_parameter", nn.Parameter(torch.tensor([0.2, -0.1], dtype=torch.float64)))
                expected = independent_jacobian(model, states)
                reference = vmc.compute_log_psi_jacobian(model, states)
                torch.testing.assert_close(reference, expected, rtol=0, atol=0)
                delta = vmc.sr_gradient(expected, energies, tau=0.05)
                before = torch.cat([p.detach().flatten() for p in model.parameters()])
                for chunk in (2, 8):
                    with self.subTest(arch=arch, axes=(px, py), chunk=chunk):
                        actual = vmc.compute_log_psi_jacobian(model, states, jacobian_batch_size=chunk)
                        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
                        candidate = copy.deepcopy(model)
                        optimizer = torch.optim.SGD(candidate.parameters(), lr=0.02)
                        direction, diagnostics = vmc.sr_update(candidate, states, energies, optimizer, 0.05,
                                                              jacobian_batch_size=chunk, return_diagnostics=True)
                        torch.testing.assert_close(direction, delta, rtol=1e-9, atol=1e-10)
                        after = torch.cat([p.detach().flatten() for p in candidate.parameters()])
                        torch.testing.assert_close(after, before - 0.02 * delta, rtol=1e-9, atol=1e-10)
                        self.assertEqual(diagnostics["jacobian_batch_size"], chunk)
                        self.assertGreater(diagnostics["batched_vjp_workspace_bytes"], 0)
                        torch.testing.assert_close(candidate.unused_parameter, model.unused_parameter, rtol=0, atol=0)

    def test_guards_before_forward_and_nonfinite_logs(self):
        model = Conv2DNet(2, 2, Nmax=2, hidden_dim=2).double()
        states = torch.tensor(generate_basis(4, 2, 2)[:5])
        reference_budget = vmc.check_sr_budget(model, len(states))
        with patch.object(model, "forward", side_effect=AssertionError("forward called")):
            for chunk in (0, True, 1.5, 65):
                with self.assertRaises(ValueError):
                    vmc.compute_log_psi_jacobian(model, states, jacobian_batch_size=chunk)
            with patch.object(torch, "empty", side_effect=AssertionError("Jacobian allocated")):
                with self.assertRaisesRegex(ValueError, "workspace"):
                    vmc.compute_log_psi_jacobian(model, states, jacobian_batch_size=4,
                                                  max_sr_bytes=reference_budget["estimated_bytes"])
        with torch.no_grad():
            model.conv.weight.fill_(float("nan"))
        for chunk in (1, 4):
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                vmc.compute_log_psi_jacobian(model, states, jacobian_batch_size=chunk)


class WarmStartTests(unittest.TestCase):
    def test_initial_burnin_completes_before_any_optimizer_step(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="test-equilibrate-") as directory:
            events = []
            real_sample, real_step = train.sample, torch.optim.SGD.step
            def record_sample(model, walkers, n_steps, burnin_steps=0, *args, **kwargs):
                if n_steps == 0:
                    self.assertEqual(events, [])
                    self.assertEqual(burnin_steps, 7)
                    before = {k: v.clone() for k, v in model.state_dict().items()}
                    result = real_sample(model, walkers, n_steps, burnin_steps, *args, **kwargs)
                    for k, v in model.state_dict().items():
                        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
                    self.assertEqual(len(result[0]), 0)
                    events.append("initial_burnin_completed")
                    return result
                self.assertEqual(events[0], "initial_burnin_completed")
                events.append("measurement")
                return real_sample(model, walkers, n_steps, burnin_steps, *args, **kwargs)
            def record_step(optimizer, *args, **kwargs):
                self.assertEqual(events[0], "initial_burnin_completed")
                events.append("optimizer_step")
                return real_step(optimizer, *args, **kwargs)
            argv = ["--Lx", "2", "--Nmax", "2", "--hidden-dim", "2", "--epochs", "2",
                    "--walkers", "2", "--steps", "1", "--burnin", "1", "--initial-burnin", "7",
                    "--thin", "3", "--device", "cpu", "--optimizer", "sgd",
                    "--output", str(Path(directory) / "model.pt")]
            with patch.object(train, "sample", side_effect=record_sample), patch.object(torch.optim.SGD, "step", record_step), redirect_stdout(io.StringIO()):
                report = train.main(argv)
            self.assertEqual(events.count("initial_burnin_completed"), 1)
            self.assertEqual(events.count("optimizer_step"), 2)
            self.assertEqual(report["initial_equilibration"]["total_walker_proposals"], 14)
            self.assertEqual(report["initial_equilibration"]["retained_samples"], 0)

    def test_zero_initial_burnin_preserves_legacy_sampling_and_seeded_weights(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="test-zero-burn-") as directory:
            argv = ["--Lx", "2", "--Nmax", "2", "--hidden-dim", "2", "--epochs", "1",
                    "--walkers", "2", "--steps", "2", "--burnin", "1", "--device", "cpu"]
            reports, weights = [], []
            for extra, name in (([], "default.pt"), (["--initial-burnin", "0"], "zero.pt")):
                output = Path(directory) / name
                with patch.object(train, "sample", wraps=train.sample) as sampler, redirect_stdout(io.StringIO()):
                    reports.append(train.main([*argv, *extra, "--output", str(output)]))
                self.assertEqual(sampler.call_count, 2)  # epoch and saved-checkpoint measurement only
                self.assertTrue(all(call.args[2] > 0 for call in sampler.call_args_list))
                weights.append(train.load_model(output)[0].state_dict())
            self.assertEqual(reports[0]["history"], reports[1]["history"])
            self.assertEqual(reports[0]["final_measurement"], reports[1]["final_measurement"])
            self.assertFalse(reports[0]["initial_equilibration"]["executed"])
            for name in weights[0]:
                torch.testing.assert_close(weights[0][name], weights[1][name], rtol=0, atol=0)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(["--initial-burnin", "-1"])

    def test_warm_start_first_measurement_schedules_and_source_identity(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="test-warm-") as directory:
            source, output = Path(directory) / "source.pt", Path(directory) / "new.pt"
            initial = Conv2DNet(2, 2, Nmax=2, hidden_dim=4, pbc_x=False, pbc_y=True).double()
            save_model(source, initial, Ntot=4)
            checksum = hashlib.sha256(source.read_bytes()).hexdigest()
            first = []
            real_measurement = train._measurement
            def record_weights(model, walkers, args):
                if not first:
                    first.append({name: tensor.detach().clone() for name, tensor in model.state_dict().items()})
                return real_measurement(model, walkers, args)
            argv = ["--arch", "Conv2DNet", "--Lx", "2", "--Ly", "2", "--Ntot", "4", "--Nmax", "2",
                    "--hidden-dim", "4", "--no-pbc-x", "--pbc-y", "--warm-start", str(source),
                    "--update-method", "sr", "--optimizer", "sgd", "--epochs", "3", "--walkers", "2",
                    "--steps", "1", "--burnin", "1", "--device", "cpu", "--dtype", "float64",
                    "--lr", "0.02", "--lr-end", "0.005", "--tau", "0.1", "--tau-end", "0.025",
                    "--jacobian-batch-size", "2", "--output", str(output)]
            with patch.object(train, "_measurement", side_effect=record_weights), redirect_stdout(io.StringIO()):
                report = train.main(argv)
            for name, tensor in initial.state_dict().items():
                torch.testing.assert_close(first[0][name], tensor, rtol=0, atol=0)
            self.assertEqual(report["warm_start"]["checkpoint_sha256"], checksum)
            self.assertFalse(report["warm_start"]["exact_resume"])
            self.assertIn("fresh", report["warm_start"]["optimizer"])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), checksum)
            for row, lr, tau in zip(report["history"], (0.02, 0.01, 0.005), (0.1, 0.05, 0.025)):
                self.assertAlmostEqual(row["lr"], lr)
                self.assertAlmostEqual(row["tau"], tau)
                self.assertEqual(row["sr"]["jacobian_batch_size"], 2)
            self.assertEqual(report["command"][-2:], ["--output", str(output)])
            self.assertEqual(report["source_sha256_at_start"]["vmc.py"], hashlib.sha256((ROOT / "vmc.py").read_bytes()).hexdigest())

    def test_warm_start_checks_geometry_physics_architecture_and_new_outputs(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="test-warm-mismatch-") as directory:
            source, output = Path(directory) / "source.pt", Path(directory) / "new.pt"
            initial = Conv2DNet(2, 2, Nmax=2, hidden_dim=4, pbc_x=False, pbc_y=True).double()
            save_model(source, initial, Ntot=4)
            argv = ["--arch", "Conv2DNet", "--Lx", "2", "--Ly", "2", "--Ntot", "4", "--Nmax", "2",
                    "--hidden-dim", "4", "--no-pbc-x", "--pbc-y", "--warm-start", str(source),
                    "--update-method", "sr", "--optimizer", "sgd", "--device", "cpu", "--output", str(output)]
            for name, value in (("arch", "FCNet"), ("Lx", 3), ("Ly", 3), ("Ntot", 3), ("Nmax", 3),
                                ("pbc_x", True), ("pbc_y", False), ("t", 0.8), ("U", 15.), ("mu", 0.2), ("hidden_dim", 8)):
                args = train.parse_args(argv)
                setattr(args, name, value)
                settings = {"hidden_dim": args.hidden_dim}
                if args.arch == "FCNet":
                    settings["hidden_layers"] = args.hidden_layers
                requested = build_model(args.arch, args.Lx, args.Ly, args.Nmax, pbc_x=args.pbc_x, pbc_y=args.pbc_y, **settings)
                with self.subTest(field=name), self.assertRaisesRegex(ValueError, "mismatch"):
                    train._load_warm_start(args, requested)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(["--warm-start", str(source), "--device", "cpu"])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args([*argv, "--output", str(source)])
            for options in (["--lr-end", "0"], ["--tau-end", "nan"], ["--jacobian-batch-size", "65"]):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    train.parse_args(options)

    def test_default_constant_schedule_and_endpoint_semantics(self):
        for epoch in range(1, 5):
            self.assertEqual(train._schedule_value(0.001, None, epoch, 4), 0.001)
            self.assertEqual(train._schedule_value(0.02, 0.02, epoch, 4), 0.02)
        self.assertEqual(train._schedule_value(0.02, 0.005, 1, 1), 0.02)
        self.assertEqual(train._schedule_value(0.02, 0.005, 1, 3), 0.02)
        self.assertAlmostEqual(train._schedule_value(0.02, 0.005, 2, 3), 0.01)
        self.assertEqual(train._schedule_value(0.02, 0.005, 3, 3), 0.005)


if __name__ == "__main__":
    unittest.main()
