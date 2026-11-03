"""CLI preflight, short CPU runs, and final saved-model measurement checks."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train
from exact_diagonalization import build_bose_hubbard_hamiltonian
from models import build_model, load_model
from vmc import occupancy_tensor_to_onehot


class TrainingTests(unittest.TestCase):
    def test_float64_summary_and_nonfinite_rejection(self):
        model = build_model("FCNet", 2, 1, Nmax=2, hidden_dim=2)
        walkers = torch.tensor([[0, 2], [1, 1]])
        args = train.parse_args(["--Lx", "2", "--Ly", "1", "--device", "cpu"])
        energies = torch.tensor([0., 1e30])
        with patch.object(train, "sample", return_value=(walkers, walkers, {})):
            with patch.object(train, "local_energy", return_value=energies):
                _, _, returned, summary = train._measurement(model, walkers, args)
            self.assertIs(returned, energies)
            self.assertTrue(np.isfinite(summary["local_energy_variance"]))
            self.assertEqual(summary["energy"], energies.double().mean().item())
            self.assertEqual(summary["local_energy_variance"], energies.double().var(unbiased=False).item())
            for invalid in (torch.tensor([float("inf"), 0.]),
                            torch.tensor([1e308, -1e308], dtype=torch.float64)):
                with patch.object(train, "local_energy", return_value=invalid):
                    with self.assertRaisesRegex(RuntimeError, "summary"):
                        train._measurement(model, walkers, args)

    def test_final_failures_preserve_destinations(self):
        original_measurement = train._measurement
        for failure in ("measurement", "serialization"):
            for previous in (False, True):
                with self.subTest(failure=failure, previous=previous):
                    with tempfile.TemporaryDirectory(dir=ROOT, prefix="test-staging-") as directory:
                        output = Path(directory) / "model.pt"
                        sidecar = Path(str(output) + ".json")
                        if previous:
                            output.write_bytes(b"previous checkpoint")
                            sidecar.write_bytes(b'{"previous_run": true}\n')
                        calls = 0

                        def measurement(*args):
                            nonlocal calls
                            calls += 1
                            if calls == 2 and failure == "measurement":
                                raise RuntimeError("injected final measurement failure")
                            result = original_measurement(*args)
                            if calls == 2 and failure == "serialization":
                                result[-1]["energy"] = float("nan")
                            return result

                        error = RuntimeError if failure == "measurement" else ValueError
                        with patch.object(train, "_measurement", side_effect=measurement), \
                                patch.object(train, "load_model", wraps=train.load_model) as loader, \
                                redirect_stdout(io.StringIO()), self.assertRaises(error):
                            train.main([
                                "--Lx", "1", "--Ly", "1", "--Ntot", "1", "--Nmax", "1",
                                "--epochs", "1", "--walkers", "1", "--steps", "1", "--burnin", "0",
                                "--hidden-dim", "2", "--device", "cpu", "--output", str(output),
                            ])
                        staged = loader.call_args.args[0]
                        self.assertNotEqual(staged, output)
                        self.assertTrue(staged.is_relative_to(ROOT))
                        self.assertFalse(staged.exists())
                        self.assertEqual(list(Path(directory).glob(".vmc-stage-*")), [])
                        if previous:
                            self.assertEqual(output.read_bytes(), b"previous checkpoint")
                            self.assertEqual(sidecar.read_bytes(), b'{"previous_run": true}\n')
                        else:
                            self.assertFalse(output.exists())
                            self.assertFalse(sidecar.exists())

    def test_defaults_and_preflight(self):
        args = train.parse_args([])
        self.assertEqual((args.Lx, args.Ly, args.Ntot, args.Nmax), (4, 4, 16, 4))
        self.assertEqual((args.t, args.U, args.mu), (1, 16.74, 0))
        self.assertEqual((args.hidden_dim, args.epochs, args.n_walkers, args.n_steps, args.burnin_steps), (32, 100, 64, 32, 32))
        self.assertEqual((args.update_method, args.optimizer, args.lr), ("native", "adam", 0.001))
        self.assertFalse(args.ed_reference)
        self.assertTrue(args.pbc)
        self.assertEqual(train.parse_args(["--update-method", "sr"]).optimizer, "sgd")
        self.assertEqual(train.parse_args(["--update-method", "sr", "--optimizer", "sgd"]).optimizer, "sgd")
        invalid = (["--t", "-1"], ["--U", "-1"], ["--mu", "nan"], ["--lr", "inf"],
                   ["--tau", "nan"], ["--Ntot", "-1"], ["--Nmax", "0"], ["--walkers", "0"],
                   ["--output", "../escape.pt"], ["--device", "mps"], ["--steps", "0"])
        with patch.object(train, "build_model", side_effect=AssertionError("expensive work")):
            for options in invalid:
                with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    train.main(options)
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="test-cli-paths-") as directory:
            output = Path(directory) / "model.pt"
            Path(str(output) + ".json").symlink_to("/tmp/escape-vmc-sidecar.json")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(["--output", str(output)])

    def test_sr_adam_rejected_before_model_construction(self):
        stderr = io.StringIO()
        with patch.object(train, "build_model", side_effect=AssertionError("model constructed")), \
                redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            train.main(["--update-method", "sr", "--optimizer", "adam", "--device", "cpu"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("SR requires torch.optim.SGD", stderr.getvalue())
        self.assertIn("--optimizer sgd", stderr.getvalue())

    def test_final_measurement_uses_updated_saved_weights(self):
        H, basis = build_bose_hubbard_hamiltonian(2, 1, Ntot=2, Nmax=2, U=5.0)
        states = torch.tensor(basis)
        x = occupancy_tensor_to_onehot(states, 2, torch.float64)
        torch.manual_seed(123)
        initial = build_model("FCNet", 2, 1, Nmax=2, hidden_dim=4).double()
        with torch.no_grad():
            initial_logs = initial(x)
        def full_basis_samples(model, walkers, n_steps, burnin_steps, evaluation_batch_size, *, proposal_steps_per_sample=1):
            return states.clone(), walkers, {"attempts": 3, "accepted": 3, "moved": 2,
                                             "accepted_rate": 1, "moved_rate": 2 / 3}
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="test-final-measurement-") as directory:
            output = Path(directory) / "final.pt"
            with patch.object(train, "sample", side_effect=full_basis_samples), redirect_stdout(io.StringIO()):
                report = train.main([
                    "--Lx", "2", "--Ly", "1", "--Ntot", "2", "--Nmax", "2", "--U", "5",
                    "--hidden-dim", "4", "--epochs", "1", "--walkers", "3", "--steps", "1",
                    "--burnin", "0", "--device", "cpu", "--dtype", "float64", "--lr", "0.03",
                    "--evaluation-batch-size", "2", "--output", str(output),
                ])
            model, metadata = load_model(output)
            with torch.no_grad():
                final_logs = model(x)
            self.assertGreater((final_logs - initial_logs).abs().max().item(), 1e-5)
            amplitudes = final_logs.exp().numpy()
            expected_energy = np.mean((H @ amplitudes) / amplitudes)
            self.assertAlmostEqual(report["final_measurement"]["energy"], expected_energy, places=12)
            self.assertNotAlmostEqual(report["history"][0]["energy"], expected_energy, places=5)
            self.assertEqual(report["final_measurement"]["measurement_timing"], "after_final_update_from_saved_checkpoint")
            saved = json.loads(Path(str(output) + ".json").read_text())
            self.assertEqual(saved["final_measurement"], report["final_measurement"])
            self.assertIsNone(report["ed_energy"])
            self.assertEqual(metadata["U"], 5.0)
            self.assertIsNone(report["cuda_peak_allocated_bytes"])
            self.assertIsNone(report["cuda_peak_allocated_scope"])

    def test_cli_cpu_native_sr_and_vacuum_smokes(self):
        configurations = (
            ["--arch", "FCNet", "--ed-reference", "--no-pbc"],
            ["--arch", "TransformerNet", "--hidden-dim", "2", "--update-method", "sr", "--optimizer", "sgd"],
            ["--arch", "JastrowNet", "--Nmax", "0", "--Ntot", "0"],
        )
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           MPLCONFIGDIR="/tmp/bhnqs-2d-mpl")
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="test-cli-smoke-") as directory:
            for index, configuration in enumerate(configurations):
                with self.subTest(configuration=configuration):
                    output = Path(directory) / f"smoke-{index}.pt"
                    result = subprocess.run([
                        sys.executable, str(ROOT / "train.py"), "--Lx", "2", "--Ly", "1",
                        "--Nmax", "2", "--Ntot", "2", "--hidden-dim", "4", "--epochs", "1",
                        "--walkers", "3", "--steps", "2", "--burnin", "1", "--device", "cpu",
                        "--evaluation-batch-size", "2", "--output", str(output), *configuration,
                    ], capture_output=True, text=True, env=environment, timeout=45)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    report = json.loads(Path(str(output) + ".json").read_text())
                    self.assertEqual(report["final_measurement"]["n_samples"], 6)
                    self.assertEqual(len(report["history"]), 1)
                    self.assertTrue(np.isfinite(report["final_measurement"]["energy"]))
                    if index == 0:
                        self.assertIsNotNone(report["ed_energy"])
                        self.assertEqual(report["settings"]["optimizer"], "adam")
                    if index == 1:
                        self.assertIn("no extra 2", report["sr_convention"])
                        self.assertEqual(report["settings"]["optimizer"], "sgd")
                    if index == 2:
                        self.assertEqual(report["final_measurement"]["energy"], 0)
                        self.assertEqual(report["final_measurement"]["sampling"]["moved"], 0)


if __name__ == "__main__":
    unittest.main()
