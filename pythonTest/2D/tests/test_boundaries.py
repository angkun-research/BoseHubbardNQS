"""Mixed-axis geometry, independent references, depth and checkpoint migration."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lattice import bonds_2d, resolve_boundaries
from exact_diagonalization import build_bose_hubbard_hamiltonian, exact_ground_state
from models import GENERAL_ARCHITECTURES, Conv2DNet, FCNet, JastrowNet, build_model, load_model, save_model, spatial_symmetries
from vmc import local_energy, sample
import train
import train_ed


def features(occupations, cutoff=1):
    return F.one_hot(torch.tensor(occupations), cutoff + 1).double()


def chain_adjacency(length, periodic):
    matrix = np.zeros((length, length))
    for i in range(length - 1):
        matrix[i, i + 1] = matrix[i + 1, i] = 1
    if periodic and length > 1:
        matrix[0, -1] = matrix[-1, 0] = 1
    return matrix


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)

    def test_bond_counts_orientations_and_legacy_calls(self):
        self.assertEqual(len(bonds_2d(8, 4, pbc_x=False, pbc_y=True)), 60)
        self.assertEqual(len(bonds_2d(12, 4, pbc_x=False, pbc_y=True)), 92)
        for px, py in ((False, True), (True, False)):
            edges = bonds_2d(3, 4, pbc_x=px, pbc_y=py)
            adjacency = np.kron(np.eye(4), chain_adjacency(3, px)) + np.kron(chain_adjacency(4, py), np.eye(3))
            expected = set(zip(*np.where(np.triu(adjacency, 1))))
            self.assertEqual(set(edges), expected)
            self.assertEqual(len(edges), len(set(edges)))
            self.assertTrue(all(i < j for i, j in edges))
        for pbc in (False, True):
            self.assertEqual(bonds_2d(3, 4, pbc), bonds_2d(3, 4, pbc_x=pbc, pbc_y=pbc))
        self.assertEqual(bonds_2d(1, 1, pbc_x=False, pbc_y=True), [])
        self.assertEqual(bonds_2d(1, 2, pbc_x=False, pbc_y=True), [(0, 1)])
        self.assertEqual(resolve_boundaries(False, pbc_y=True), (False, True))
        with self.assertRaises(ValueError):
            bonds_2d(2, pbc_x="periodic")

    def test_independent_single_particle_hamiltonian_and_local_energy(self):
        for px, py in ((False, True), (True, False)):
            with self.subTest(pbc_x=px, pbc_y=py):
                # Independent tensor-sum of 1D adjacency operators in physical
                # coordinate order, then permute into the enumerated Fock basis.
                physical = -0.7 * (np.kron(np.eye(4), chain_adjacency(3, px))
                                   + np.kron(chain_adjacency(4, py), np.eye(3))) - 0.2 * np.eye(12)
                H, basis = build_bose_hubbard_hamiltonian(3, 4, Ntot=1, Nmax=1, t=0.7, mu=0.2,
                                                         pbc_x=px, pbc_y=py)
                order = [state.index(1) for state in basis]
                reference = physical[np.ix_(order, order)]
                np.testing.assert_allclose(H.toarray(), reference, atol=0)
                model = FCNet(3, 4, Nmax=1, hidden_dim=3, hidden_layers=3, pbc_x=px, pbc_y=py).double()
                with torch.no_grad():
                    amplitudes = model(features(basis)).exp().numpy()
                measured = local_energy(model, torch.tensor(basis), t=0.7, mu=0.2, evaluation_batch_size=2)
                np.testing.assert_allclose(measured.numpy(), reference @ amplitudes / amplitudes, atol=1e-12)
                energy, state, _, _ = exact_ground_state(3, 4, Ntot=1, Nmax=1, t=0.7, mu=0.2, pbc_x=px, pbc_y=py)
                self.assertAlmostEqual(energy, np.linalg.eigvalsh(reference)[0])
                np.testing.assert_allclose(H @ state, energy * state, atol=1e-12)
                initial = torch.tensor(basis)
                samples, walkers, diagnostics = sample(model, initial, 4, evaluation_batch_size=3)
                self.assertTrue(torch.all(samples.sum(1) == 1))
                self.assertEqual(samples.dtype, torch.long)
                self.assertLessEqual(diagnostics["moved"], diagnostics["accepted"])
        # Independent U/2 and bosonic factors on a y wrap, absent when y is open.
        for px, py in ((False, True), (True, False)):
            H, basis = build_bose_hubbard_hamiltonian(3, 4, Ntot=2, Nmax=2, U=5, t=0.7, pbc_x=px, pbc_y=py)
            index = {state: i for i, state in enumerate(basis)}
            double = (2,) + (0,) * 11
            split = (1,) + (0,) * 8 + (1, 0, 0)
            self.assertEqual(H[index[double], index[double]], 5)
            self.assertAlmostEqual(H[index[split], index[double]], -0.7 * np.sqrt(2) if py else 0)

    def test_conv_axis_padding_and_legacy_uniform_outputs(self):
        x = features([[1] + [0] * 11])
        for px, py in ((False, True), (True, False)):
            for dy, dx in ((0, 1), (1, 0)):
                model = Conv2DNet(3, 4, Nmax=1, hidden_dim=1, pbc_x=px, pbc_y=py).double()
                with torch.no_grad():
                    model.conv.weight.zero_()
                    model.conv.bias.zero_()
                    model.conv.weight[0, 1, 1 + dy, 1 + dx] = 1
                outputs = []
                hook = model.conv.register_forward_hook(lambda module, args, output: outputs.append(output))
                model(x)
                hook.remove()
                expected = torch.zeros((4, 3), dtype=torch.float64)
                if dx and px:
                    expected[0, 2] = 1
                if dy and py:
                    expected[3, 0] = 1
                torch.testing.assert_close(outputs[0][0, 0], expected)
        for periodic in (False, True):
            model = Conv2DNet(3, 4, Nmax=1, pbc=periodic, hidden_dim=3).double()
            old_conv = nn.Conv2d(2, 3, 3, padding=1, padding_mode="circular" if periodic else "zeros").double()
            old_conv.load_state_dict(model.conv.state_dict())
            grid = x.transpose(1, 2).reshape(1, 2, 4, 3)
            expected = model.head(old_conv(grid)).squeeze(-1)
            torch.testing.assert_close(model(x), expected, atol=1e-12, rtol=1e-12)

    def test_jastrow_wraps_and_axis_symmetries(self):
        for px, py in ((False, True), (True, False)):
            model = JastrowNet(3, 4, Nmax=1, pbc_x=px, pbc_y=py).double()
            with torch.no_grad():
                for index, (dx, dy) in enumerate(model.displacements.tolist()):
                    model.pair[index, 1, 1] = dx + 10 * dy
            x_ends = [1, 0, 1] + [0] * 9
            y_ends = [1] + [0] * 8 + [1, 0, 0]
            self.assertEqual(model(features([x_ends])).item(), 1 if px else 2)
            self.assertEqual(model(features([y_ends])).item(), 10 if py else 30)
            self.assertEqual("x_periodic_translations" in spatial_symmetries(model), px)
            self.assertEqual("y_periodic_translations" in spatial_symmetries(model), py)
            x = features([[1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]])
            grid = x.reshape(1, 4, 3, 2)
            torch.testing.assert_close(model(x), model(grid.flip(1).reshape_as(x)))
            torch.testing.assert_close(model(x), model(grid.flip(2).reshape_as(x)))
            torch.testing.assert_close(model(x), model(grid.roll(1, 2 if px else 1).reshape_as(x)))

    def test_fc_depths_and_v2_all_model_round_trips(self):
        x = features([[0, 1, 0, 1, 0, 0]], 1)
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="test-v2-") as directory:
            for depth in (1, 2, 3, 4):
                model = FCNet(3, 2, 1, True, 4, hidden_layers=depth, pbc_x=False).double()
                self.assertEqual(sum(isinstance(layer, nn.Linear) for layer in model.net), depth + 1)
                model(x).sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
                if depth == 2:
                    self.assertEqual(list(model.state_dict()), [f"net.{i}.{suffix}" for i in (0, 2, 4) for suffix in ("weight", "bias")])
                path = save_model(Path(directory) / f"fc-depth{depth}.pt", model, Ntot=2)
                restored, metadata = load_model(path)
                self.assertEqual(metadata["format_version"], 2)
                self.assertEqual(metadata["architecture_settings"]["hidden_layers"], depth)
                self.assertIsNone(metadata["pbc"])
                self.assertEqual((metadata["pbc_x"], metadata["pbc_y"]), (False, True))
                torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)
            for arch in GENERAL_ARCHITECTURES:
                settings = {} if arch == "JastrowNet" else {"n_hidden" if arch == "RBMNet" else "hidden_dim": 4}
                for px, py in ((False, True), (True, False)):
                    model = build_model(arch, 3, 2, 1, False, pbc_x=px, pbc_y=py, **settings).double()
                    path = save_model(Path(directory) / f"{arch}-{px}.pt", model, Ntot=2)
                    restored, metadata = load_model(path)
                    torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)
                    self.assertEqual((restored.pbc_x, restored.pbc_y), (px, py))
        self.assertEqual(FCNet(2, depth=3).hidden_layers, 3)
        for options in ({"hidden_layers": 0}, {"hidden_layers": True}, {"depth": -1}, {"hidden_layers": 3, "depth": 4}):
            with self.assertRaises(ValueError):
                FCNet(2, **options)

    def test_format1_predictions_and_invalid_v2_metadata(self):
        x = features([[0, 1, 0, 1, 0, 0]])
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="test-v1-") as directory:
            path = Path(directory) / "legacy.pt"
            for arch in GENERAL_ARCHITECTURES:
                for periodic in (False, True):
                    settings = {} if arch == "JastrowNet" else {"n_hidden" if arch == "RBMNet" else "hidden_dim": 4}
                    model = build_model(arch, 3, 2, 1, periodic, **settings).double().eval()
                    expected = model(x)
                    if arch == "FCNet":
                        old = nn.Sequential(nn.Linear(12, 4), nn.Tanh(), nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 1)).double()
                        old.load_state_dict(model.net.state_dict())
                        expected = old(x.flatten(1)).squeeze(-1)
                    if arch == "Conv2DNet":
                        old = nn.Conv2d(2, 4, 3, padding=1, padding_mode="circular" if periodic else "zeros").double()
                        old.load_state_dict(model.conv.state_dict())
                        expected = model.head(old(x.transpose(1, 2).reshape(1, 2, 2, 3))).squeeze(-1)
                    save_model(path, model, Ntot=2)
                    checkpoint = torch.load(path, weights_only=True)
                    meta = checkpoint["metadata"]
                    meta["format_version"] = 1
                    meta.pop("pbc_x")
                    meta.pop("pbc_y")
                    meta["architecture_settings"].pop("hidden_layers", None)
                    torch.save(checkpoint, path)
                    restored, metadata = load_model(path)
                    self.assertEqual((restored.pbc_x, restored.pbc_y), (periodic, periodic))
                    torch.testing.assert_close(restored(x), expected, atol=1e-12, rtol=1e-12)
            model = FCNet(3, 2, 1, pbc_x=False)
            save_model(path, model, Ntot=2)
            original = torch.load(path, weights_only=True)
            for changes in ({"pbc": True}, {"pbc_x": None}, {"pbc_y": "periodic"}):
                altered = copy.deepcopy(original)
                altered["metadata"].update(changes)
                torch.save(altered, path)
                with self.assertRaises(ValueError):
                    load_model(path)

    def test_cli_precedence_depth_and_benchmark_defaults(self):
        parsers = [train.parse_args, train_ed.parse_args]
        for name in ("run_baseline", "run_ed_fits"):
            spec = importlib.util.spec_from_file_location(f"boundary_{name}", ROOT / "benchmarks" / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            parsers.append(module.parse_args)
        for index, parse in enumerate(parsers):
            default = parse(["--device", "cpu"])
            self.assertEqual((default.pbc_x, default.pbc_y), (True, True))
            self.assertEqual(default.hidden_layers, 2)
            for options, expected in ((["--no-pbc", "--pbc-y"], (False, True)),
                                      (["--pbc-y", "--no-pbc"], (False, True)),
                                      (["--pbc", "--no-pbc-y"], (True, False)),
                                      (["--no-pbc"], (False, False)), (["--pbc"], (True, True))):
                args = parse(["--device", "cpu", "--depth", "3", *options])
                self.assertEqual((args.pbc_x, args.pbc_y), expected)
                self.assertEqual(args.hidden_layers, 3)

    def test_independently_saved_format1_forward_gold_when_available(self):
        path = ROOT / "benchmarks/results/pre-cylinder-legacy-forward-gold.json"
        if not path.exists():
            self.skipTest("independent local checkpoint gold is not available")
        gold = json.loads(path.read_text())
        if not all((ROOT / record["checkpoint"]).exists() for record in gold["records"]):
            self.skipTest("ignored historical checkpoints are not available")
        x = features(gold["configurations"], gold["Nmax"])
        for record in gold["records"]:
            with self.subTest(arch=record["arch"]):
                checkpoint = ROOT / record["checkpoint"]
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), record["checkpoint_sha256"])
                model, metadata = load_model(checkpoint)
                self.assertEqual(metadata["format_version"], 1)
                self.assertEqual((model.pbc_x, model.pbc_y), (True, True))
                actual = model.double()(x)
                expected = torch.tensor(record["expected_log_amplitudes_float64"], dtype=torch.float64)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
