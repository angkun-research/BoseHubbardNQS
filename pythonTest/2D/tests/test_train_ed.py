"""Independent checks for full-sector supervised log-amplitude fitting."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_ed
from models import ARCH_BUILDERS, build_model, load_model


class TableModel(nn.Module):
    def __init__(self, logs, n_sites=2, Nmax=2):
        super().__init__()
        self.n_sites, self.Nmax = n_sites, Nmax
        self.logs = nn.Parameter(torch.as_tensor(logs, dtype=torch.float64).clone())

    def forward(self, x):
        return self.logs[:len(x)]


class SupervisedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_cli_omitted_Ly_is_square_and_explicit_rectangle_is_preserved(self):
        default = train_ed.parse_args(["--device", "cpu"])
        self.assertEqual((default.Lx, default.Ly, default.Ntot), (2, 2, 4))
        square = train_ed.parse_args(["--Lx", "3", "--device", "cpu"])
        self.assertEqual((square.Lx, square.Ly, square.Ntot), (3, 3, 9))
        rectangle = train_ed.parse_args(["--Lx", "2", "--Ly", "3", "--device", "cpu"])
        self.assertEqual((rectangle.Lx, rectangle.Ly, rectangle.Ntot), (2, 3, 6))

    def test_loss_and_gradient_against_direct_reference(self):
        values = np.array([0.7, -1.1, 0.2])
        targets = np.log(np.array([0.2, 0.3, 0.5]))
        pred = torch.tensor(values, requires_grad=True)
        target = torch.tensor(targets)
        residual = values - 0.5 * np.log(np.exp(2 * values).sum())
        residual -= targets - 0.5 * np.log(np.exp(2 * targets).sum())
        probabilities = np.exp(2 * values) / np.exp(2 * values).sum()
        expected_gradient = 2 / len(values) * (residual - probabilities * residual.sum())
        loss = train_ed.normalized_log_mse(pred, target)
        gradient, = torch.autograd.grad(loss, pred)
        self.assertAlmostEqual(loss.item(), np.mean(residual**2), places=14)
        np.testing.assert_allclose(gradient.numpy(), expected_gradient, atol=1e-14)
        # Normalizing each of two minibatches would give a different objective.
        separate = (2 * train_ed.normalized_log_mse(pred[:2], target[:2])
                    + train_ed.normalized_log_mse(pred[2:], target[2:])) / 3
        self.assertGreater(abs(separate.item() - loss.item()), 0.01)

    def test_global_shift_invariance_of_loss_and_gradients(self):
        target = torch.tensor([-1.2, -2.3, -0.7], dtype=torch.float64)
        baseline = None
        for offset in (0, 31, -50):
            pred = torch.tensor([0.2, -0.3, 1.2], dtype=torch.float64, requires_grad=True)
            loss = train_ed.normalized_log_mse(pred + offset, target - offset)
            gradient, = torch.autograd.grad(loss, pred)
            if baseline is None:
                baseline = loss.detach(), gradient
            torch.testing.assert_close(loss, baseline[0], atol=1e-13, rtol=1e-13)
            torch.testing.assert_close(gradient, baseline[1], atol=1e-13, rtol=1e-13)
            self.assertAlmostEqual(gradient.sum().item(), 0, places=13)

    def test_infidelity_direct_loss_gradient_and_shift_invariance(self):
        values = np.array([0.7, -1.1, 0.2])
        raw = np.array([0.2, 0.3, 0.5])
        target = raw / np.linalg.norm(raw)
        psi = np.exp(values) / np.linalg.norm(np.exp(values))
        overlap = psi @ target
        expected_gradient = -2 * overlap * (psi * target - overlap * psi**2)
        for offset in (0, 31, -50):
            pred = torch.tensor(values + offset, requires_grad=True)
            loss = train_ed.normalized_infidelity(pred, -7 * torch.tensor(raw))
            gradient, = torch.autograd.grad(loss, pred)
            self.assertAlmostEqual(loss.item(), 1 - overlap**2, places=13)
            np.testing.assert_allclose(gradient.numpy(), expected_gradient, atol=1e-13)
            self.assertAlmostEqual(gradient.sum().item(), 0, places=13)
        for invalid in ([0, 0, 0], [1, float("nan"), 0], [1, 0]):
            with self.assertRaises(ValueError):
                train_ed.normalized_infidelity(torch.tensor(values), invalid)

    def test_infidelity_uses_raw_ed_and_selection_is_not_log_mse(self):
        raw = np.array([1.0, 0.1, 1e-15])
        raw /= np.linalg.norm(raw)
        basis = [(0, 2), (1, 1), (2, 0)]
        H = csr_matrix(np.eye(3) - np.outer(raw, raw))
        with patch.object(train_ed, "exact_ground_state", return_value=(0.0, raw, basis, H)):
            data = train_ed.make_ed_dataset(2, 1, Ntot=2, Nmax=2, eps=0.2)
        model = TableModel(data["target_log"])
        initial = train_ed.compare_predictions(model, data, loss="infidelity")
        self.assertAlmostEqual(initial["normalized_log_mse"], 0)
        self.assertGreater(initial["training_loss"], 0)
        self.assertAlmostEqual(initial["training_loss"], 1 - initial["amplitude_overlap_squared"])
        self.assertAlmostEqual(initial["log_mse_gradient_norm"], 0)
        self.assertGreater(initial["selected_objective_gradient_norm"], 0)
        with redirect_stdout(io.StringIO()):
            model, report, best = train_ed.train_model_on_ed(model, data, epochs=20, lr=0.03, eval_every=10, loss="infidelity")
        self.assertEqual(report["loss_name"], "infidelity")
        self.assertIsNone(report["best_normalized_log_mse"])
        self.assertGreater(report["best_epoch"], 0)
        self.assertGreater(report["normalized_log_mse_at_selected_checkpoint"], 0)
        self.assertIn("not selected by log-MSE", report["selection"])
        model.load_state_dict(best)
        selected = train_ed.compare_predictions(model, data, loss="infidelity")
        self.assertAlmostEqual(selected["training_loss"], report["best_training_loss"])
        self.assertLess(selected["training_loss"], initial["training_loss"])
        # Flooring is irrelevant to infidelity, including its parameter gradient.
        altered = train_ed.compare_predictions(model, {**data, "target_log": data["target_log"] + torch.tensor([2., -1., 3.])}, loss="infidelity")
        self.assertEqual(selected["training_loss"], altered["training_loss"])
        self.assertEqual(selected["selected_objective_gradient_norm"], altered["selected_objective_gradient_norm"])

    def test_default_log_loss_compatibility(self):
        self.assertEqual(train_ed.parse_args(["--device", "cpu"]).loss, "log_mse")
        data = train_ed.make_ed_dataset(2, 1, Ntot=2, Nmax=2, U=3.0)
        states = []
        for options in ({}, {"loss": "log_mse"}):
            model = TableModel([0.5, -0.4, 0.1])
            with redirect_stdout(io.StringIO()):
                model, report, best = train_ed.train_model_on_ed(model, data, epochs=5, eval_every=5, **options)
            states.append(model.logs.detach().clone())
            self.assertEqual(report["best_normalized_log_mse"], report["best_training_loss"])
            self.assertEqual(report["best_training_loss"], report["normalized_log_mse_at_selected_checkpoint"])
            final = report["final_metrics"]
            self.assertEqual(final["training_loss"], final["normalized_log_mse"])
            self.assertEqual(final["gradient_norm"], final["log_mse_gradient_norm"])
            self.assertEqual(final["gradient_norm"], final["selected_objective_gradient_norm"])
        torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            train_ed.train_model_on_ed(model, data, loss="unknown")

    def test_larger_fidelity_is_unsquared_and_separate_from_historical_gate(self):
        data = train_ed.make_ed_dataset(3, 3, Ntot=1, Nmax=1)
        raw = data["raw_ed_state"]
        direction = np.zeros(9)
        direction[:2] = [-1 / np.sqrt(2), 1 / np.sqrt(2)]
        # Uniform raw ED state is orthogonal to this signed direction.
        psi = 0.994 * raw + np.sqrt(1 - 0.994**2) * direction
        self.assertTrue(np.all(psi > 0))
        metrics = train_ed.compare_predictions(TableModel(np.log(psi), n_sites=9, Nmax=1), data)
        self.assertAlmostEqual(metrics["amplitude_overlap_abs_unsquared"], 0.994)
        self.assertLess(metrics["amplitude_overlap_squared"], 0.99)
        self.assertTrue(metrics["larger_system_fidelity_applied"])
        self.assertTrue(metrics["larger_system_fidelity_converged"])
        self.assertFalse(metrics["fit_gate_passed"])
        rectangle = train_ed.make_ed_dataset(2, 5, Ntot=1, Nmax=1)
        rectangle_model = TableModel(np.log(np.abs(rectangle["raw_ed_state"])), n_sites=10, Nmax=1)
        rectangle_metrics = train_ed.compare_predictions(rectangle_model, rectangle)
        self.assertTrue(rectangle_metrics["larger_system_fidelity_applied"])
        self.assertTrue(rectangle_metrics["larger_system_fidelity_converged"])
        small = train_ed.make_ed_dataset(2, 1, Ntot=2, Nmax=2)
        small_metrics = train_ed.compare_predictions(TableModel([0., 0., 0.]), small)
        self.assertFalse(small_metrics["larger_system_fidelity_applied"])
        self.assertIsNone(small_metrics["larger_system_fidelity_converged"])

    def test_clamp_is_reported_and_physical_ed_stays_raw(self):
        raw = np.array([1.0, 0.1, 1e-15])
        raw /= np.linalg.norm(raw)
        basis = [(0, 2), (1, 1), (2, 0)]
        H = csr_matrix(np.eye(3) - np.outer(raw, raw))
        with patch.object(train_ed, "exact_ground_state", return_value=(0.0, raw, basis, H)):
            data = train_ed.make_ed_dataset(2, 1, Ntot=2, Nmax=2, eps=0.2)
        self.assertEqual(data["metadata"]["clamped_target_count"], 2)
        self.assertEqual(data["metadata"]["eps"], 0.2)
        np.testing.assert_array_equal(data["raw_ed_state"], raw)
        model = TableModel(data["target_log"])
        metrics = train_ed.compare_predictions(model, data)
        self.assertAlmostEqual(metrics["normalized_log_mse"], 0, places=14)
        self.assertLess(metrics["amplitude_overlap_abs_unsquared"], 0.9999)
        self.assertGreater(metrics["energy"], 0.001)
        self.assertFalse(metrics["fit_gate_passed"])
        self.assertAlmostEqual(data["metadata"]["raw_ed_reference"]["energy"], 0, places=14)
        self.assertLess(data["metadata"]["raw_ed_reference"]["energy_variance"], 1e-28)
        # A small energy error alone does not pass the overlap condition.
        weak_H = {**data, "H": H * 0.001}
        weak_metrics = train_ed.compare_predictions(model, weak_H)
        self.assertLess(weak_metrics["energy_error_per_site"], 0.001)
        self.assertFalse(weak_metrics["fit_gate_passed"])
        # Conversely, high overlap alone does not pass the energy condition.
        with patch.object(train_ed, "exact_ground_state", return_value=(0.0, raw, basis, H * 200)):
            high_overlap_data = train_ed.make_ed_dataset(2, 1, Ntot=2, Nmax=2, eps=0.005)
        high_metrics = train_ed.compare_predictions(TableModel(high_overlap_data["target_log"]), high_overlap_data)
        self.assertGreater(high_metrics["amplitude_overlap_abs_unsquared"], 0.9999)
        self.assertGreater(high_metrics["energy_error_per_site"], 0.001)
        self.assertFalse(high_metrics["fit_gate_passed"])

    def test_dataset_guards_signs_zeros_and_defaults(self):
        with patch.object(train_ed, "exact_ground_state", side_effect=AssertionError("ED called")):
            for kwargs in ({"t": 0}, {"t": -1}, {"U": -1}, {"eps": 0}, {"eps": 1}, {"eps": float("nan")}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    train_ed.make_ed_dataset(2, **kwargs)
        for raw in (np.array([1.0, 0.0]), np.array([1.0, -0.2]), np.array([1.0, np.nan])):
            with patch.object(train_ed, "exact_ground_state", return_value=(0, raw, [(0, 1), (1, 0)], csr_matrix((2, 2)))):
                with self.assertRaises(ValueError):
                    train_ed.make_ed_dataset(2, 1, Ntot=1, Nmax=1)
        square = train_ed.make_ed_dataset(2, Ntot=1, Nmax=1)
        self.assertEqual((square["metadata"]["Lx"], square["metadata"]["Ly"]), (2, 2))
        self.assertEqual(square["x"].dtype, torch.float64)
        self.assertEqual(square["metadata"]["clamped_target_count"], 0)
        with self.assertRaisesRegex(ValueError, "dimension"):
            train_ed.make_ed_dataset(2, 3, max_dimension=1)
        with self.assertRaisesRegex(ValueError, "sparse memory"):
            train_ed.make_ed_dataset(2, 1, max_sparse_bytes=1)

    def test_reduced_full_sector_fit_loss(self):
        data = train_ed.make_ed_dataset(2, 1, Ntot=2, Nmax=2, U=3.0)
        model = TableModel([0.5, -0.4, 0.1])
        initial = train_ed.compare_predictions(model, data)
        with redirect_stdout(io.StringIO()):
            model, report, best = train_ed.train_model_on_ed(model, data, lr=0.03, epochs=120, eval_every=40)
        self.assertLess(report["final_metrics"]["normalized_log_mse"], initial["normalized_log_mse"] * 0.01)
        self.assertGreater(report["final_metrics"]["amplitude_overlap_abs_unsquared"], 0.9999)
        self.assertTrue(report["final_metrics"]["fit_gate_passed"])
        final = train_ed.compare_predictions(model, data)
        self.assertEqual(final["gradient_norm"], report["final_metrics"]["gradient_norm"])
        model.load_state_dict(best)
        self.assertAlmostEqual(train_ed.compare_predictions(model, data)["normalized_log_mse"], report["best_normalized_log_mse"])
        self.assertEqual([entry["epoch"] for entry in report["history"]], [0, 40, 80, 120])

    def test_all_five_model_full_batch_update(self):
        data = train_ed.make_ed_dataset(2, 1, Ntot=2, Nmax=2)
        for loss in train_ed.LOSS_NAMES:
            for arch in ARCH_BUILDERS:
                with self.subTest(arch=arch, loss=loss):
                    settings = {} if arch == "JastrowNet" else {"n_hidden" if arch == "RBMNet" else "hidden_dim": 4}
                    model = build_model(arch, 2, 1, Nmax=2, **settings).double()
                    with redirect_stdout(io.StringIO()):
                        _, report, _ = train_ed.train_model_on_ed(model, data, epochs=1, eval_every=1, loss=loss)
                    self.assertTrue(np.isfinite(report["final_metrics"]["gradient_norm"]))
                    self.assertTrue(np.isfinite(report["final_metrics"]["selected_objective_gradient_norm"]))
                    self.assertEqual(len(report["history"]), 2)
        with self.assertRaisesRegex(ValueError, "entire"):
            train_ed.compare_predictions(model, {**data, "x": data["x"][:1]})

    def test_best_loss_selection_keeps_initial_state_after_overshoot(self):
        data = train_ed.make_ed_dataset(2, 1, Ntot=2, Nmax=2, U=3.0)
        model = TableModel([0.0, 0.0, 0.0])
        initial = train_ed.compare_predictions(model, data)
        with redirect_stdout(io.StringIO()):
            model, report, best_state = train_ed.train_model_on_ed(model, data, lr=10, epochs=1, eval_every=1)
        self.assertEqual(report["best_epoch"], 0)
        self.assertGreater(report["final_metrics"]["normalized_log_mse"], initial["normalized_log_mse"])
        model.load_state_dict(best_state)
        self.assertAlmostEqual(train_ed.compare_predictions(model, data)["normalized_log_mse"], initial["normalized_log_mse"])

    def test_cli_saved_metrics_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="test-ed-fit-") as directory:
            path = Path(directory) / "best.pt"
            argv = ["--Lx", "2", "--Ly", "1", "--Ntot", "2", "--Nmax", "2", "--arch", "FCNet",
                    "--width", "4", "--epochs", "4", "--eval-every", "2", "--device", "cpu", "--output", str(path)]
            with redirect_stdout(io.StringIO()):
                report = train_ed.main(argv)
            restored, metadata = load_model(path)
            data = train_ed.make_ed_dataset(2, 1, Ntot=2, Nmax=2)
            actual = train_ed.compare_predictions(restored, data)
            for key, value in actual.items():
                if isinstance(value, str) or value is None:
                    self.assertEqual(value, report["best_checkpoint_metrics"][key])
                else:
                    self.assertAlmostEqual(value, report["best_checkpoint_metrics"][key], places=13)
            self.assertAlmostEqual(actual["normalized_log_mse"], report["best_normalized_log_mse"], places=13)
            self.assertEqual(next(restored.parameters()).dtype, torch.float64)
            saved = json.loads(Path(str(path) + ".json").read_text())
            self.assertEqual(saved["final_metrics"], report["final_metrics"])
            self.assertEqual(metadata["U"], 16.74)
            checkpoint_bytes = path.read_bytes()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train_ed.main(argv)
            self.assertEqual(path.read_bytes(), checkpoint_bytes)

    def test_bad_cli_settings_before_ed_and_stage_failure(self):
        for argv in (["--epochs", "0"], ["--lr", "nan"], ["--t", "0"], ["--device", "mps"],
                     ["--eps", "inf"], ["--output", "../escape.pt"], ["--Nmax", "-1"]):
            with patch.object(train_ed, "make_ed_dataset", side_effect=AssertionError("ED called")):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    train_ed.main(argv)
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="test-ed-stage-") as directory:
            path = Path(directory) / "best.pt"
            with patch.object(train_ed, "load_model", side_effect=RuntimeError("staged validation failed")):
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "validation"):
                    train_ed.main(["--Lx", "1", "--Ly", "1", "--Ntot", "0", "--epochs", "1",
                                   "--device", "cpu", "--output", str(path)])
            self.assertFalse(path.exists())
            self.assertFalse(Path(str(path) + ".json").exists())


if __name__ == "__main__":
    unittest.main()
