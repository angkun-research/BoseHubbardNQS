"""Frozen estimator checks against independent ladder operators and time blocks."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate
from observables import block_mean_statistics, energy_from_observables, measure_observables, one_body_estimators, sample_frozen
from models import FCNet, save_model
from vmc import sample


class LookupState(nn.Module):
    """Positive arbitrary wavefunction on a tiny, explicitly enumerated sector."""
    def __init__(self, basis, amplitudes, Lx=2, Ly=1, Nmax=2):
        super().__init__()
        self.Lx, self.Ly, self.Nmax = Lx, Ly, Nmax
        self.n_sites, self.pbc_x, self.pbc_y, self.pbc = Lx * Ly, False, True, None
        self.offset = nn.Parameter(torch.zeros((), dtype=torch.float64))
        powers = (Nmax + 1)**torch.arange(self.n_sites - 1, -1, -1)
        self.register_buffer("powers", powers)
        self.register_buffer("codes", torch.tensor(basis) @ powers)
        self.register_buffer("logs", torch.tensor(np.log(amplitudes), dtype=torch.float64))
        self.max_seen = 0
        self.grad_modes = []

    def forward(self, x):
        self.max_seen = max(self.max_seen, len(x))
        self.grad_modes.append(torch.is_grad_enabled())
        codes = x.argmax(-1) @ self.powers
        indices = torch.searchsorted(self.codes, codes)
        return self.logs[indices] + self.offset


class ObservableTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.basis = [(0, 2), (1, 1), (2, 0)]
        self.psi = np.array([0.1, 0.5, 0.8])
        self.psi /= np.linalg.norm(self.psi)
        self.model = LookupState(self.basis, self.psi).eval()
        self.chains = torch.tensor(self.basis).reshape(3, 1, 2)

    def test_row_action_pointwise_and_probability_weighted_operator_reference(self):
        # Full tensor-product ladder matrices, independent of connected-state
        # generation. Projection indices encode |n0,n1> with site0 first.
        a = np.diag(np.sqrt([1., 2.]), 1)
        operators = [np.kron(a, np.eye(3)), np.kron(np.eye(3), a)]
        indices = [np.ravel_multi_index(state, (3, 3)) for state in self.basis]
        expected = np.empty((3, 2, 2))
        for i in range(2):
            for j in range(2):
                operator = (operators[i].T @ operators[j])[np.ix_(indices, indices)]
                expected[:, i, j] = operator @ self.psi / self.psi
        values = one_body_estimators(self.model, self.chains, evaluation_batch_size=2)
        np.testing.assert_allclose(values[:, 0].numpy(), expected, atol=1e-14)
        # At bra (2,0), rho[0,1] has a nonzero ROW action. The reversed
        # ket-action implementation would incorrectly return zero here.
        self.assertAlmostEqual(values[2, 0, 0, 1].item(), np.sqrt(2) * self.psi[1] / self.psi[2])
        self.assertEqual(values[2, 0, 1, 0].item(), 0)
        probability = self.psi**2
        rho = np.einsum("s,sij->ij", probability, values[:, 0].numpy())
        np.testing.assert_allclose(rho, rho.T, atol=1e-14)
        density = probability @ np.asarray(self.basis)
        np.testing.assert_allclose(rho.diagonal(), density, atol=1e-14)
        self.assertAlmostEqual(np.trace(rho), 2)
        self.assertLessEqual(self.model.max_seen, 2)
        self.assertFalse(any(self.model.grad_modes))
        self.assertIsNone(self.model.offset.grad)

    def test_energy_reconstruction_and_raw_hermiticity_before_symmetrizing(self):
        result = measure_observables(self.model, self.chains, t=0.7, U=5, mu=0.2, evaluation_batch_size=1)
        occupation = np.asarray(self.basis)
        H = np.diag(2.5 * (occupation * (occupation - 1)).sum(1) - 0.4)
        H[0, 1] = H[1, 0] = H[1, 2] = H[2, 1] = -0.7 * np.sqrt(2)
        expected = H @ self.psi / self.psi
        np.testing.assert_allclose(result["local_energy_samples"][:, 0], expected, atol=1e-13)
        np.testing.assert_allclose(result["reconstructed_energy_samples"][:, 0], expected, atol=1e-13)
        self.assertEqual(result["diagnostics"]["rho_diagonal_density_max_abs"], 0)
        self.assertEqual(result["diagnostics"]["rho_trace_minus_Ntot_max_abs"], 0)
        self.assertFalse(result["diagnostics"]["symmetrized"])
        self.assertGreater(result["diagnostics"]["raw_mean_rho_hermiticity_max_abs"], 0.1)
        # Uniform averaging over a basis is NOT a physical expectation. Keep
        # its raw asymmetry rather than masking it with symmetrization.
        raw_mean = result["one_body_matrix"]
        self.assertFalse(torch.allclose(raw_mean, raw_mean.T))
        expected_raw = (raw_mean - raw_mean.T).abs().max().item()
        self.assertEqual(result["diagnostics"]["raw_mean_rho_hermiticity_max_abs"], expected_raw)
        probability = torch.tensor(self.psi**2)
        physical_energy = (probability * result["reconstructed_energy_samples"][:, 0]).sum().item()
        self.assertAlmostEqual(physical_energy, self.psi @ H @ self.psi, places=13)

    def test_selected_pairs_agree_with_full_and_missing_bonds_are_explicit(self):
        full = one_body_estimators(self.model, self.chains, evaluation_batch_size=1)
        pairs = [(1, 0), (0, 1)]
        selected = measure_observables(self.model, self.chains, pairs=pairs, evaluation_batch_size=1)
        self.assertEqual(selected["one_body_pairs"], pairs)
        self.assertEqual(selected["one_body_samples"].shape, (3, 1, 2))
        torch.testing.assert_close(selected["one_body_samples"], full[..., [1, 0], [0, 1]])
        self.assertIsNone(selected["one_body_matrix"])
        self.assertIsNotNone(selected["reconstructed_energy_samples"])
        missing = measure_observables(self.model, self.chains, pairs=[(0, 1)])
        self.assertIsNone(missing["reconstructed_energy_samples"])
        self.assertEqual(missing["diagnostics"]["missing_directed_bonds_for_energy_reconstruction"], [(1, 0)])
        self.assertIsNone(missing["diagnostics"]["raw_mean_rho_hermiticity_max_abs"])
        with self.assertRaisesRegex(ValueError, "directed"):
            energy_from_observables(missing["density"], missing["onsite_factorial_moment"], missing["one_body_pair_means"],
                                    2, 1, pbc_x=False, pbc_y=True, pairs=[(0, 1)])

    def test_vacuum_full_cutoff_and_validation_guards(self):
        for basis, Nmax, expected in (([(0, 0)], 2, [0, 0]), ([(2, 2)], 2, [2, 2]), ([(0, 0)], 0, [0, 0])):
            model = LookupState(basis, [1.], Nmax=Nmax).eval()
            result = measure_observables(model, torch.tensor(basis).reshape(1, 1, 2))
            torch.testing.assert_close(result["one_body_matrix"], torch.diag(torch.tensor(expected, dtype=torch.float64)))
        with patch.object(self.model, "forward", side_effect=AssertionError("forward called")):
            with self.assertRaisesRegex(ValueError, "storage"):
                one_body_estimators(self.model, self.chains, max_observable_bytes=1)
        for invalid in ([(0, 2)], [(0, 0), (0, 0)], [(0.5, 1)]):
            with self.assertRaises(ValueError):
                one_body_estimators(self.model, self.chains, pairs=invalid)
        with self.assertRaisesRegex(ValueError, "integer"):
            one_body_estimators(self.model, self.chains.double())
        with self.assertRaisesRegex(ValueError, "time"):
            one_body_estimators(self.model, self.chains[:, 0])

    def test_time_blocks_chain_axes_and_explicit_insufficient_data(self):
        values = torch.tensor([[t, 100 + t] for t in range(9)], dtype=torch.float64)
        result = block_mean_statistics(values, 2)
        expected_blocks = np.array([[0.5, 100.5], [2.5, 102.5], [4.5, 104.5], [6.5, 106.5]])
        np.testing.assert_array_equal(result["block_means"].numpy(), expected_blocks)
        self.assertEqual(result["mean"].item(), 53.5)
        self.assertEqual(result["all_samples_mean"].item(), 54)
        self.assertEqual(result["discarded_tail_steps"], 1)
        self.assertAlmostEqual(result["block_standard_error"].item(), expected_blocks.std(ddof=1) / np.sqrt(8))
        self.assertEqual(result["between_chain_standard_error"].item(), 50)
        insufficient = block_mean_statistics(values[:3], 2)
        self.assertIsNone(insufficient["block_standard_error"])
        self.assertEqual(insufficient["status"], "insufficient_complete_time_blocks")
        no_blocks = block_mean_statistics(values[:1, :1], 2)
        self.assertIsNone(no_blocks["block_standard_error"])
        self.assertIsNone(no_blocks["between_chain_standard_error"])
        with self.assertRaises(ValueError):
            block_mean_statistics(values, 0)

    def test_frozen_sampler_preserves_time_walker_order_and_weights(self):
        initial = torch.tensor([[0, 2], [2, 0]])
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        torch.manual_seed(12)
        flat, expected_walkers, _ = sample(self.model, initial, 5, 3, 2)
        torch.manual_seed(12)
        chains, walkers, diagnostics = sample_frozen(self.model, initial, 5, 3, 2)
        self.assertEqual(chains.shape, (5, 2, 2))
        torch.testing.assert_close(chains.reshape(-1, 2), flat)
        torch.testing.assert_close(walkers, expected_walkers)
        measure_observables(self.model, chains, pairs=[(0, 0), (0, 1)], block_length=2)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        self.assertEqual(diagnostics["attempts"], 10)
        self.assertIsNone(self.model.offset.grad)

    def test_cli_selected_default_and_retained_raw_time_axes(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="test-frozen-") as directory:
            checkpoint = Path(directory) / "model.pt"
            output = Path(directory) / "evaluation.json"
            model = FCNet(3, 2, Nmax=2, hidden_dim=2, pbc_x=False, pbc_y=True).double()
            save_model(checkpoint, model, Ntot=2)
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            argv = ["--checkpoint", str(checkpoint), "--walkers", "2", "--steps", "4", "--burnin", "3",
                    "--block-length", "2", "--batch-size", "2", "--device", "cpu", "--output", str(output)]
            with redirect_stdout(io.StringIO()):
                report = evaluate.main(argv)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["evaluation_dtype"], "float64")
            self.assertEqual(report["settings"]["dtype"], "checkpoint")
            self.assertLess(len(report["one_body_pairs"]), 36)
            self.assertIn("reference_rows", report["pair_policy"])
            self.assertEqual(report["statistics"]["energy"]["complete_blocks_per_chain"], 2)
            self.assertLess(report["raw_checks_before_symmetrization"]["local_energy_reconstruction_max_abs"], 1e-12)
            self.assertEqual(report["raw_checks_before_symmetrization"]["rho_trace_minus_Ntot_max_abs"], 0)
            with np.load(ROOT / report["raw_samples_path"]) as raw:
                self.assertEqual(raw["occupations"].shape, (4, 2, 6))
                self.assertEqual(raw["one_body_estimators"].shape, (4, 2, len(report["one_body_pairs"])))
                np.testing.assert_array_equal(raw["physical_pairs"], report["one_body_pairs"])
                np.testing.assert_allclose(raw["local_energy"], raw["reconstructed_energy"], atol=1e-12)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluate.main(argv)

    def test_float32_checkpoint_evaluated_in_float64_without_rewriting_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="test-frozen-dtype-") as directory:
            checkpoint = Path(directory) / "model.pt"
            output = Path(directory) / "evaluation.json"
            model = FCNet(2, 1, Nmax=2, hidden_dim=3).float()
            save_model(checkpoint, model, Ntot=2)
            original_bytes = checkpoint.read_bytes()
            checksum = hashlib.sha256(original_bytes).hexdigest()
            real_sample = evaluate.sample_frozen
            seen = []

            def check_sampling_dtype(loaded, *args, **kwargs):
                self.assertEqual(next(loaded.parameters()).dtype, torch.float64)
                for name, value in loaded.state_dict().items():
                    torch.testing.assert_close(value, model.state_dict()[name].double(), rtol=0, atol=0)
                seen.append(True)
                return real_sample(loaded, *args, **kwargs)

            with patch.object(evaluate, "sample_frozen", side_effect=check_sampling_dtype), redirect_stdout(io.StringIO()):
                report = evaluate.main(["--checkpoint", str(checkpoint), "--walkers", "2", "--steps", "4",
                                        "--burnin", "2", "--block-length", "2", "--device", "cpu",
                                        "--dtype", "float64", "--output", str(output)])
            self.assertEqual(seen, [True])
            self.assertEqual(report["evaluation_dtype"], "float64")
            self.assertEqual(report["checkpoint_metadata"]["dtype"], "float32")
            self.assertEqual(report["checkpoint_sha256"], checksum)
            self.assertEqual(checkpoint.read_bytes(), original_bytes)
            self.assertEqual(json.loads(output.read_text())["evaluation_dtype"], "float64")

    def test_checkpoint_replacement_during_sampling_keeps_loaded_identity(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="test-frozen-identity-") as directory:
            checkpoint = Path(directory) / "model.pt"
            replacement = Path(directory) / "replacement.pt"
            output = Path(directory) / "evaluation.json"
            original_model = FCNet(2, 1, Nmax=2, hidden_dim=2).double()
            replacement_model = copy.deepcopy(original_model)
            with torch.no_grad():
                replacement_model.net[0].weight[0, 0] += 0.7
            save_model(checkpoint, original_model, Ntot=2)
            save_model(replacement, replacement_model, Ntot=2)
            loaded_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            replacement_hash = hashlib.sha256(replacement.read_bytes()).hexdigest()
            self.assertNotEqual(loaded_hash, replacement_hash)
            real_sample = evaluate.sample_frozen

            def replace_while_sampling(model, *args, **kwargs):
                torch.testing.assert_close(model.net[0].weight, original_model.net[0].weight, rtol=0, atol=0)
                checkpoint.write_bytes(replacement.read_bytes())
                return real_sample(model, *args, **kwargs)

            with patch.object(evaluate, "sample_frozen", side_effect=replace_while_sampling), redirect_stdout(io.StringIO()):
                report = evaluate.main(["--checkpoint", str(checkpoint), "--walkers", "2", "--steps", "4",
                                        "--burnin", "1", "--block-length", "2", "--device", "cpu", "--output", str(output)])
            self.assertEqual(report["checkpoint_sha256"], loaded_hash)
            self.assertEqual(json.loads(output.read_text())["checkpoint_sha256"], loaded_hash)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), replacement_hash)

    def test_checkpoint_replacement_during_load_fails_before_sampling(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="test-frozen-load-race-") as directory:
            checkpoint = Path(directory) / "model.pt"
            replacement = Path(directory) / "replacement.pt"
            output = Path(directory) / "evaluation.json"
            model = FCNet(2, 1, Nmax=2, hidden_dim=2).double()
            save_model(checkpoint, model, Ntot=2)
            with torch.no_grad():
                model.net[0].weight[0, 0] += 0.7
            save_model(replacement, model, Ntot=2)
            real_load = evaluate.load_model

            def replace_during_load(path, device):
                loaded = real_load(path, device=device)
                checkpoint.write_bytes(replacement.read_bytes())
                return loaded

            with patch.object(evaluate, "load_model", side_effect=replace_during_load):
                with patch.object(evaluate, "sample_frozen", side_effect=AssertionError("sampled ambiguous checkpoint")):
                    with self.assertRaisesRegex(RuntimeError, "changed during loading"):
                        evaluate.main(["--checkpoint", str(checkpoint), "--device", "cpu", "--output", str(output)])
            self.assertFalse(output.exists())
            self.assertFalse(Path(str(output) + ".samples.npz").exists())


if __name__ == "__main__":
    unittest.main()
