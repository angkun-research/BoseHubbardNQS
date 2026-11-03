"""Focused CPU checks for deterministic 2D log-amplitude models."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import (
    ARCH_BUILDERS, BOND_CONVENTION, HAMILTONIAN_CONVENTION,
    Conv2DNet, JastrowNet, TransformerNet, build_model, load_model,
    parameter_count, save_model,
)


def features(occupations, cutoff=2):
    return F.one_hot(torch.tensor(occupations), cutoff + 1).double()


def small_settings(arch):
    if arch == "JastrowNet":
        return {}
    return {"n_hidden" if arch == "RBMNet" else "hidden_dim": 8}


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_shapes_repeatability_and_gradients(self):
        for arch in ARCH_BUILDERS:
            for Lx, Ly in ((3, 2), (1, 1)):
                for periodic in (False, True):
                    with self.subTest(arch=arch, shape=(Lx, Ly), pbc=periodic):
                        model = build_model(arch, Lx, Ly, Nmax=2, pbc=periodic,
                                            **small_settings(arch)).double().train()
                        x = features([[index % 3 for index in range(Lx * Ly)]])
                        output = model(x)
                        self.assertEqual(output.shape, (1,))
                        self.assertEqual(output.dtype, torch.float64)
                        self.assertTrue(torch.isfinite(output).all())
                        torch.testing.assert_close(output, model(x), rtol=0, atol=0)
                        model.eval()
                        torch.testing.assert_close(output, model(x), rtol=0, atol=0)
                        model.train()
                        batch = model(x.expand(3, -1, -1))
                        torch.testing.assert_close(batch, output.expand(3))
                        output.sum().backward()
                        for parameter in model.parameters():
                            self.assertIsNotNone(parameter.grad)
                            self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters()), 0)
                        self.assertGreater(parameter_count(model), 0)

    def test_invalid_dimensions_and_shapes(self):
        for arch in ARCH_BUILDERS:
            for options in ({"Lx": 0}, {"Lx": 2, "Ly": -1}, {"Lx": 2, "Nmax": -1},
                            {"Lx": True}, {"Lx": 2, "pbc": "open"}):
                with self.subTest(arch=arch, options=options), self.assertRaises(ValueError):
                    build_model(arch, **options)
            model = build_model(arch, 2, Nmax=2, **small_settings(arch))
            self.assertEqual((model.Lx, model.Ly), (2, 2))
            with self.assertRaises(ValueError):
                model(torch.ones(1, 3, 3))
            if arch != "JastrowNet":
                with self.assertRaises(ValueError):
                    build_model(arch, 2, **{next(iter(small_settings(arch))): 0})
        with self.assertRaises(ValueError):
            build_model("unknown", 2)
        with self.assertRaises(ValueError):
            TransformerNet(2, feedforward_dim=0)

    def test_parameter_gradients_against_finite_difference(self):
        x = features([[0, 1, 2, 2, 1, 0], [2, 0, 1, 1, 0, 2]])
        for arch in ARCH_BUILDERS:
            with self.subTest(arch=arch):
                model = build_model(arch, 3, 2, Nmax=2, **small_settings(arch)).double()
                weights = torch.tensor([1.0, 0.37], dtype=torch.float64)
                objective = (model(x) * weights).sum()
                parameters = list(model.parameters())
                gradients = torch.autograd.grad(objective, parameters)
                directions = [torch.randn_like(parameter) for parameter in parameters]
                analytical = sum((gradient * direction).sum() for gradient, direction in zip(gradients, directions))
                originals = [parameter.detach().clone() for parameter in parameters]
                epsilon = 1e-6
                values = []
                with torch.no_grad():
                    for sign in (1, -1):
                        for parameter, original, direction in zip(parameters, originals, directions):
                            parameter.copy_(original + sign * epsilon * direction)
                        values.append((model(x) * weights).sum())
                numerical = (values[0] - values[1]) / (2 * epsilon)
                torch.testing.assert_close(analytical, numerical, rtol=1e-4, atol=1e-7)

    def test_jastrow_symmetries_and_capacity(self):
        x = features([[0, 1, 2, 2, 0, 1]])
        for periodic in (False, True):
            model = JastrowNet(3, 2, Nmax=2, pbc=periodic).double()
            with torch.no_grad():
                model.onsite.normal_()
                model.pair.normal_()
            grid = x.reshape(1, 2, 3, 3)
            reference = model(x)
            for axis in (1, 2):
                torch.testing.assert_close(reference, model(grid.flip(axis).reshape_as(x)))
                if periodic:
                    torch.testing.assert_close(reference, model(grid.roll(1, axis).reshape_as(x)))
            with torch.no_grad():
                # Antisymmetric occupation contributions cancel exactly.
                perturbation = torch.randn_like(model.pair)
                model.pair.add_(perturbation - perturbation.transpose(-1, -2))
            torch.testing.assert_close(reference, model(x))
        self.assertEqual(JastrowNet(4, 2).pair.shape[0], 5)
        self.assertEqual(JastrowNet(4, 2, pbc=False).pair.shape[0], 7)

    def test_jastrow_distances_and_axis_distinction(self):
        # Two occupied sites separated across the x edge share a table with
        # adjacent sites only under periodic boundaries.
        for periodic, expected in ((False, 7), (True, 2)):
            model = JastrowNet(4, 2, Nmax=1, pbc=periodic).double()
            with torch.no_grad():
                for index, (dx, dy) in enumerate(model.displacements.tolist()):
                    model.pair[index, 1, 1] = 2 if (dx, dy) == (1, 0) else 7
            self.assertEqual(model(features([[1, 0, 0, 1, 0, 0, 0, 0]], 1)).item(), expected)
        model = JastrowNet(2, 2, Nmax=1).double()
        with torch.no_grad():
            for index, displacement in enumerate(model.displacements.tolist()):
                model.pair[index, 1, 1] = 2 if displacement == [1, 0] else 5
        self.assertEqual(model(features([[1, 1, 0, 0]], 1)).item(), 2)
        self.assertEqual(model(features([[1, 0, 1, 0]], 1)).item(), 5)

    def test_rectangular_jastrow_group_and_pair_gradients(self):
        for Lx, Ly in ((4, 3), (3, 2), (1, 4)):
            model = JastrowNet(Lx, Ly, Nmax=2).double()
            with torch.no_grad():
                model.onsite.normal_()
                model.pair.normal_()
            x = features(torch.randint(3, (2, Lx * Ly)).tolist())
            grid = x.reshape(2, Ly, Lx, 3)
            expected = model(x)
            for dx in range(Lx):
                for dy in range(Ly):
                    shifted = grid.roll((dy, dx), (1, 2))
                    for reflected in (shifted, shifted.flip(1), shifted.flip(2), shifted.flip((1, 2))):
                        torch.testing.assert_close(expected, model(reflected.reshape_as(x)),
                                                   rtol=1e-12, atol=1e-12)
            expected.sum().backward()
            # Each unordered site pair contributes total derivative one to its
            # occupation table, including the transpose averaging.
            self.assertAlmostEqual(model.pair.grad.sum().item(), Lx * Ly * (Lx * Ly - 1))
            self.assertGreater(model.pair.grad.abs().sum().item(), 0)

    def test_convolution_encoding_and_boundary(self):
        # Read the x+1 occupation-one feature: wrap at the right edge only
        # for periodic grids. Also check that reshaping preserves y*Lx+x.
        x = features([[1, 0, 0, 0, 1, 0]], 1)
        for periodic, expected in ((False, [[0, 0, 0], [1, 0, 0]]),
                                   (True, [[0, 0, 1], [1, 0, 0]])):
            model = Conv2DNet(3, 2, Nmax=1, pbc=periodic, hidden_dim=1).double()
            with torch.no_grad():
                model.conv.weight.zero_()
                model.conv.bias.zero_()
                model.conv.weight[0, 1, 1, 2] = 1
            captured = []
            handle = model.conv.register_forward_hook(lambda module, args, output: captured.append(output))
            model(x)
            handle.remove()
            torch.testing.assert_close(captured[0][0, 0], torch.tensor(expected).double())

    def test_transformer_positions_and_occupation_encoding(self):
        model = TransformerNet(3, 2, Nmax=2, hidden_dim=8).double().train()
        x = features([[0, 1, 2, 0, 2, 1]])
        captured = []
        handle = model.encoder.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        output = model(x)
        handle.remove()
        for y in range(2):
            for xpos in range(3):
                site = y * 3 + xpos
                expected = model.occupation_embedding(x[:, site]) + model.x_embedding[xpos] + model.y_embedding[y]
                torch.testing.assert_close(captured[0][:, site], expected)
        # Learned position information lets the ansatz distinguish equal-count
        # configurations with different arrangements despite mean pooling.
        rearranged = model(x.flip(1))
        self.assertGreater((output - rearranged).abs().item(), 1e-10)

    def test_transformer_physical_evaluation_contexts(self):
        model = TransformerNet(4, 3, Nmax=2, hidden_dim=8).double().train()
        x = features(torch.randint(3, (3, 12)).tolist())
        reference = model(x).detach()
        with torch.no_grad():
            torch.testing.assert_close(reference, model(x), rtol=1e-12, atol=1e-12)
        model.eval()
        with torch.inference_mode():
            torch.testing.assert_close(reference, model(x), rtol=1e-12, atol=1e-12)
            chunked = torch.cat([model(block) for block in x.split(1)])
            torch.testing.assert_close(reference, chunked, rtol=1e-12, atol=1e-12)

    def test_checkpoint_physical_validation(self):
        model = build_model("FCNet", 2, 1, Nmax=2, hidden_dim=8)
        invalid = [("t", -1), ("U", -1), ("Ntot", -1), ("Ntot", 5), ("Ntot", 1.5)]
        invalid += [(name, value) for name in ("t", "U", "mu")
                    for value in (1 + 0j, 1j, float("nan"), float("inf"),
                                  -float("inf"), True, "1", None)]
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="test-physical-") as directory:
            path = Path(directory) / "model.pt"
            for name, value in invalid:
                with self.subTest(operation="save", name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        save_model(path, model, **{name: value})
                    self.assertFalse(path.exists())
            save_model(path, model, Ntot=2, t=0, U=0, mu=-0.5)
            checkpoint = torch.load(path, weights_only=True)
            _, metadata = load_model(path)
            self.assertEqual((metadata["t"], metadata["U"], metadata["mu"]), (0, 0, -0.5))
            for name, value in invalid:
                with self.subTest(operation="load", name=name, value=value):
                    altered = {**checkpoint, "metadata": {**checkpoint["metadata"], name: value}}
                    torch.save(altered, path)
                    with self.assertRaisesRegex(ValueError, name):
                        load_model(path)

    def test_inference_checkpoint_round_trips(self):
        x = features([[0, 1, 2, 2, 1, 0]])
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="test-checkpoints-") as directory:
            for arch in ARCH_BUILDERS:
                for dtype in (torch.float32, torch.float64):
                    with self.subTest(arch=arch, dtype=dtype):
                        model = build_model(arch, 3, 2, Nmax=2, pbc=False,
                                            **small_settings(arch)).to(dtype=dtype)
                        if arch == "JastrowNet":
                            with torch.no_grad():
                                model.pair.normal_()
                        path = save_model(Path(directory) / f"{arch}-{dtype}.pt", model, Ntot=6)
                        restored, metadata = load_model(path)
                        self.assertFalse(restored.training)
                        self.assertEqual(metadata["architecture_settings"], model.architecture_settings)
                        self.assertEqual(metadata["parameter_count"], parameter_count(model))
                        self.assertEqual(metadata["hamiltonian_convention"], HAMILTONIAN_CONVENTION)
                        self.assertEqual(metadata["bond_convention"], BOND_CONVENTION)
                        self.assertEqual((metadata["Lx"], metadata["Ly"], metadata["pbc"]), (3, 2, False))
                        self.assertEqual(metadata["U"], 16.74)
                        torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
                        checkpoint = torch.load(path, weights_only=True)
                        checkpoint["metadata"]["bond_convention"] = "parallel_bonds"
                        torch.save(checkpoint, path)
                        with self.assertRaisesRegex(ValueError, "convention"):
                            load_model(path)

    def test_output_path_confinement(self):
        model = build_model("FCNet", 2)
        with self.assertRaisesRegex(ValueError, "under pythonTest/2D"):
            save_model("../escape.pt", model)
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="test-paths-") as directory:
            link = Path(directory) / "outside"
            link.symlink_to("/tmp", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "under pythonTest/2D"):
                save_model(link / "escape.pt", model)


if __name__ == "__main__":
    unittest.main()
