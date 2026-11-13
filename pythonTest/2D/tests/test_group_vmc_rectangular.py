"""D4-tied rectangular filters and independent GCNN VMC compatibility checks."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import GroupEquivariantConv2DNet, load_model, save_model
from exact_diagonalization import exact_ground_state
from vmc import compute_log_psi_jacobian, local_energy, sr_gradient
sys.path.insert(0, str(ROOT/'benchmarks'))
from train_rectangular_group import rectangular_orbits
from train_square_group import orbit_log_mse
from train_ed import normalized_log_mse


class GroupVMCRectangularTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(654)

    def test_rectangular_d2_all_translations_and_transposed_quarter_turn(self):
        for lx, ly in ((3, 4), (3, 5), (4, 5)):
            model = GroupEquivariantConv2DNet(lx, ly, Nmax=2, hidden_dim=2).double()
            image = F.one_hot(torch.randint(3, (2, ly, lx)), 3).double()
            x = image.flatten(1, 2)
            with torch.no_grad():
                output = model(x)
                for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                    for tx in range(lx):
                        for ty in range(ly):
                            sites = [((sy*(y-ty)) % ly)*lx + (sx*(xx-tx)) % lx
                                     for y in range(ly) for xx in range(lx)]
                            torch.testing.assert_close(model(x[:, sites]), output, rtol=1e-11, atol=1e-12)
                transposed = GroupEquivariantConv2DNet(ly, lx, Nmax=2, hidden_dim=2).double()
                transposed.load_state_dict(model.state_dict())
                rotated = torch.rot90(image, dims=(1, 2)).flatten(1, 2)
                torch.testing.assert_close(transposed(rotated), output, rtol=1e-11, atol=1e-12)
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = save_model(Path(directory)/'rect.pt', model)
                loaded, metadata = load_model(path)
                self.assertIn('D2_rotations_reflections', metadata['spatial_symmetries'])
                self.assertNotIn('D4_rotations_reflections', metadata['spatial_symmetries'])
                torch.testing.assert_close(loaded(x), model(x), rtol=0, atol=0)

    def test_local_energy_matches_sparse_operator_and_vmap_matches_reference(self):
        _, _, basis, hamiltonian = exact_ground_state(2, 3, 2, 2)
        model = GroupEquivariantConv2DNet(2, 3, Nmax=2, hidden_dim=2).double()
        states = torch.tensor(basis)
        with torch.no_grad():
            psi = model(F.one_hot(states, 3).double()).exp().numpy()
        energies = local_energy(model, states, evaluation_batch_size=16)
        np.testing.assert_allclose(energies.numpy(), (hamiltonian @ psi)/psi, rtol=1e-11, atol=1e-11)
        subset = states[:8]
        reference = compute_log_psi_jacobian(model, subset, jacobian_method='vjp')
        vectorized = compute_log_psi_jacobian(model, subset, jacobian_method='vmap', jacobian_batch_size=4)
        torch.testing.assert_close(vectorized, reference, rtol=1e-10, atol=1e-12)
        primal = sr_gradient(reference, energies[:8], tau=1e-3, solver='parameter')
        dual = sr_gradient(vectorized, energies[:8], tau=1e-3, solver='sample')
        torch.testing.assert_close(primal, dual, rtol=1e-8, atol=1e-10)

    def test_rectangular_orbit_loss_and_gradients_match_full_sector(self):
        _, _, basis, _ = exact_ground_state(2, 3, 2, 2)
        reps, labels, counts = rectangular_orbits(basis, 2, 3, 2)
        x = F.one_hot(torch.tensor(basis), 3).double()
        model = GroupEquivariantConv2DNet(2,3,Nmax=2,hidden_dim=2).double()
        target = torch.randn(len(basis),dtype=torch.float64)
        target -= .5*torch.logsumexp(2*target,dim=0)
        means = np.bincount(labels,weights=target.numpy())/counts
        variance = np.bincount(labels,weights=(target.numpy()-means[labels])**2)/counts
        full = normalized_log_mse(model(x), target)
        reduced = orbit_log_mse(model(x[reps]),torch.tensor(counts).double(),
                               torch.tensor(means),torch.tensor(variance))
        full_grad = torch.autograd.grad(full,tuple(model.parameters()))
        reduced_grad = torch.autograd.grad(reduced,tuple(model.parameters()))
        torch.testing.assert_close(full,reduced,rtol=1e-12,atol=1e-12)
        for a,b in zip(full_grad,reduced_grad):
            torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-11)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
