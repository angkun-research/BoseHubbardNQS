"""Independent coordinate/orientation checks for the square regular D4 CNN."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import GroupEquivariantConv2DNet, GENERAL_ARCHITECTURES, build_model, load_model, save_model
from square_group import d4_product, d4_inverse, site_pullback, space_group_pullbacks, transform_features
import train_ed
sys.path.insert(0, str(ROOT/'benchmarks'))
from train_square_group import make_orbit_data, orbit_log_mse
from exact_diagonalization import generate_basis


# Oracle uses integer matrices, not the production multiplication/index helpers.
R = np.array([[0, -1], [1, 0]])
S = np.array([[1, 0], [0, -1]])
MATRICES = [np.linalg.matrix_power(R, r) @ np.linalg.matrix_power(S, m) for m in (0, 1) for r in range(4)]


def matrix_index(matrix):
    return next(k for k, candidate in enumerate(MATRICES) if np.array_equal(matrix, candidate))


def independent_sites(L, q, tx=0, ty=0):
    result = []
    for y in range(L):
        for x in range(L):
            sx, sy = MATRICES[q].T @ np.array([x-tx, y-ty])
            result.append(int((sy % L)*L + sx % L))
    return torch.tensor(result)


def independent_features(features, q, tx=0, ty=0):
    indices = [matrix_index(MATRICES[q].T @ matrix) for matrix in MATRICES]
    return features[:, indices].flatten(-2)[..., independent_sites(features.shape[-1], q, tx, ty)].reshape_as(features)


class GroupCNNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_group_action_and_all72_distinct_site_permutations(self):
        for q in range(8):
            self.assertEqual(d4_inverse(q), matrix_index(MATRICES[q].T))
            for a in range(8):
                self.assertEqual(d4_product(q, a), matrix_index(MATRICES[q] @ MATRICES[a]))
            for ty in range(3):
                for tx in range(3):
                    torch.testing.assert_close(site_pullback(3, q, tx, ty), independent_sites(3, q, tx, ty))
        self.assertNotEqual(d4_product(1, 4), d4_product(4, 1))
        self.assertEqual(len(torch.unique(space_group_pullbacks(3), dim=0)), 72)

    def test_lift_and_true_orientation_mixing_against_coordinate_sums(self):
        model = GroupEquivariantConv2DNet(3, Nmax=1, hidden_dim=2).double()
        x = F.one_hot(torch.randint(2, (1, 9)), 2).double()
        image = x.reshape(1, 3, 3, 2)[0]
        random_features = torch.randn(1, 8, 2, 3, 3, dtype=torch.float64)
        expected_lift = torch.empty_like(random_features)
        expected_group = torch.empty_like(random_features)
        with torch.no_grad():
            for h in range(8):
                for y in range(3):
                    for xx in range(3):
                        lift = model.lift_bias.clone()
                        group = model.group_bias.clone()
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                kx, ky = MATRICES[h].T @ np.array([dx, dy])
                                lift += model.lift_weight[:, :, ky+1, kx+1] @ image[(y+dy) % 3, (xx+dx) % 3]
                                for a in range(8):
                                    relative = matrix_index(MATRICES[h].T @ MATRICES[a])
                                    group += model.group_weight[:, :, relative, ky+1, kx+1] @ random_features[0, a, :, (y+dy) % 3, (xx+dx) % 3]
                        expected_lift[0, h, :, y, xx] = F.silu(lift)
                        expected_group[0, h, :, y, xx] = F.silu(group)
            torch.testing.assert_close(model.lift_features(x), expected_lift, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(model.group_features(random_features), expected_group, rtol=1e-12, atol=1e-12)

    def test_all72_internal_equivariance_and_scalar_invariance(self):
        model = GroupEquivariantConv2DNet(3, Nmax=2, hidden_dim=3).double()
        x = F.one_hot(torch.randint(3, (2, 9)), 3).double()
        arbitrary = torch.randn(2, 8, 3, 3, 3, dtype=torch.float64)
        with torch.no_grad():
            lift, group, output = model.lift_features(x), model.group_features(arbitrary), model(x)
            for q in range(8):
                for ty in range(3):
                    for tx in range(3):
                        transformed = x[:, independent_sites(3, q, tx, ty)]
                        torch.testing.assert_close(model.lift_features(transformed), independent_features(lift, q, tx, ty), rtol=1e-11, atol=1e-12)
                        torch.testing.assert_close(model.group_features(independent_features(arbitrary, q, tx, ty)), independent_features(group, q, tx, ty), rtol=1e-11, atol=1e-12)
                        torch.testing.assert_close(model(transformed), output, rtol=1e-11, atol=1e-12)
                        torch.testing.assert_close(transform_features(arbitrary, q, tx, ty), independent_features(arbitrary, q, tx, ty))
        # The same group law works beyond the 3x3 teaching lattice.
        larger = GroupEquivariantConv2DNet(5, Nmax=1, hidden_dim=2).double()
        y = F.one_hot(torch.randint(2, (1, 25)), 2).double()
        torch.testing.assert_close(larger(y), larger(y[:, independent_sites(5, 5, 2, 3)]), rtol=1e-11, atol=1e-12)

    def test_determinism_gradients_and_nontrivial_geometry_dependence(self):
        model = GroupEquivariantConv2DNet(3, hidden_dim=2).double().train()
        states = torch.tensor([[0,2,1,1,1,1,1,1,1], [0,1,1,1,2,1,1,1,1]])
        x = F.one_hot(states, 5).double()
        output = model(x)
        self.assertGreater(abs((output[0]-output[1]).item()), 1e-9)
        torch.testing.assert_close(output, model(x), rtol=0, atol=0)
        model.eval(); torch.testing.assert_close(output, model(x), rtol=0, atol=0)
        parameters = list(model.parameters())
        gradients = torch.autograd.grad((output*torch.tensor([1.,.3])).sum(), parameters)
        self.assertTrue(all(g.isfinite().all() for g in gradients))
        directions = [torch.randn_like(p) for p in parameters]
        analytical = sum((g*d).sum() for g,d in zip(gradients,directions))
        originals = [p.detach().clone() for p in parameters]
        values = []
        with torch.no_grad():
            for sign in (1,-1):
                for p,original,d in zip(parameters,originals,directions): p.copy_(original+sign*1e-6*d)
                values.append((model(x)*torch.tensor([1.,.3])).sum())
            for p,original in zip(parameters,originals): p.copy_(original)
        torch.testing.assert_close(analytical,(values[0]-values[1])/2e-6,rtol=1e-5,atol=1e-8)

    def test_checkpoint_metadata_and_early_geometry_rejections(self):
        model = build_model('GroupEquivariantConv2DNet',3,hidden_dim=2).double()
        x = F.one_hot(torch.randint(5,(1,9)),5).double()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = save_model(Path(directory)/'group.pt', model, 9)
            loaded, metadata = load_model(path)
            torch.testing.assert_close(loaded(x),model(x),rtol=0,atol=0)
            self.assertIn('D4_rotations_reflections',metadata['spatial_symmetries'])
            self.assertEqual(metadata['architecture_settings'],{'hidden_dim':2})
        for kw in ({'Ly':2},{'pbc':False},{'pbc_x':False},{'pbc_y':False},{'hidden_dim':0}):
            with self.assertRaises(ValueError): GroupEquivariantConv2DNet(3,**kw)
        with self.assertRaises(SystemExit): train_ed.parse_args(['--arch','GroupEquivariantConv2DNet','--Lx','3','--Ly','2'])
        self.assertEqual(len(GENERAL_ARCHITECTURES),6)

    def test_weighted338_loss_and_all_parameter_gradients_match_full19855(self):
        basis=np.asarray(generate_basis(9,9,4),dtype=np.int64)
        x=F.one_hot(torch.tensor(basis),5).double()
        target=torch.linspace(-10,0,len(basis),dtype=torch.float64)
        # Deliberately noninvariant target perturbation checks the within-orbit
        # variance term; reducing just one target value per orbit is incorrect.
        target=target+.01*torch.randn_like(target)
        dataset={'basis':basis,'x':x,'target_log':target,
                 'metadata':{'Lx':3,'Ly':3,'Nmax':4,'pbc_x':True,'pbc_y':True}}
        orbit=make_orbit_data(dataset)
        self.assertEqual(len(orbit['representatives']),338)
        self.assertEqual(orbit['multiplicities'].sum().item(),19855)
        model=GroupEquivariantConv2DNet(3,hidden_dim=2).double()
        full=train_ed.normalized_log_mse(model(x),target)
        full_grad=torch.autograd.grad(full,tuple(model.parameters()))
        rep_logs=model(orbit['x'])
        reduced=orbit_log_mse(rep_logs,orbit['multiplicities'],orbit['target_mean'],orbit['target_variance'])
        reduced_grad=torch.autograd.grad(reduced,tuple(model.parameters()))
        torch.testing.assert_close(full,reduced,rtol=1e-12,atol=1e-12)
        for a,b in zip(full_grad,reduced_grad):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-11)
        full_logZ=torch.logsumexp(2*model(x),dim=0)
        reduced_logZ=torch.logsumexp(2*rep_logs+orbit['multiplicities'].log(),dim=0)
        torch.testing.assert_close(full_logZ,reduced_logZ,rtol=1e-12,atol=1e-12)
        shifted=orbit_log_mse(rep_logs+123.,orbit['multiplicities'],orbit['target_mean'],orbit['target_variance'])
        torch.testing.assert_close(shifted,reduced,rtol=1e-11,atol=1e-12)

    def test_all_six_legacy_checkpoints_against_pre_extension_source(self):
        source=(ROOT/'models.py').read_text()
        old=source.replace('import math\n','').replace('from square_group import D4_ELEMENTS, d4_inverse, d4_product, kernel_pullbacks\n','')
        start=old.index('class GroupEquivariantConv2DNet')
        stop=old.index('BASELINE_ARCHITECTURES',start)
        old=old[:start]+old[stop:]
        old=old.replace('GENERAL_ARCHITECTURES = (*BASELINE_ARCHITECTURES, "Conv2DTransformerNet")\n','')
        old=old.replace(', Conv2DTransformerNet, GroupEquivariantConv2DNet)',', Conv2DTransformerNet)')
        old=old.replace('    if isinstance(model, GroupEquivariantConv2DNet):\n        return ["x_periodic_translations", "y_periodic_translations", "D4_rotations_reflections"]\n','')
        fixture=json.loads((ROOT/'tests/fixtures/legacy-six-models.json').read_text())
        tree=ast.parse(old);tree.body=tree.body[1:]  # module prose is not behavior
        self.assertEqual(hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
                         fixture['legacy_module_AST_without_header_sha256'])
        namespace={'__file__':str(ROOT/'models.py')};exec(compile(old,'pre_group_models.py','exec'),namespace)
        x=F.one_hot(torch.randint(5,(4,9)),5).double()
        checked=0
        for entry in fixture['checkpoints']:
            path=ROOT/entry['path']
            if not path.exists():continue
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),entry['sha256'])
            old_model,old_meta=namespace['load_model'](path,'cpu')
            new_model,new_meta=load_model(path,'cpu')
            self.assertEqual(old_meta,new_meta)
            torch.testing.assert_close(old_model.double()(x),new_model.double()(x),rtol=0,atol=0)
            checked+=1
        if not checked:self.skipTest('optional historical checkpoints absent; legacy AST already checked')


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
