"""Run with .venv/bin/python -m unittest discover -s tests -v."""
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
sys.path.insert(0, str(ROOT / 'src/python'))
import Bose_Hubbard_NQS as one
import bosehubbard_nqs_2d_bridge as two
from bosehubbard_nqs_inference import resolve_device
from pythonTest.bosehubbard2d import models


def settings(arch):
    if arch == 'JastrowNet':
        return {}
    return {'n_hidden' if arch == 'RBMNet' else 'hidden_dim': 4}


class BridgeTests(unittest.TestCase):
    def test_1d_roundtrip_batch_and_validation(self):
        torch.manual_seed(123)
        configs = np.array([[0, 2, 0], [1, 0, 0], [2, 1, 0]], dtype=np.int64)
        for arch in one.ARCH_BUILDERS:
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                model = one.build_model(arch, 3, 2, 4).eval()
                path = Path(directory) / 'model.pt'
                torch.save(dict(arch=arch, n_sites=3, nmax=2, hidden_dim=4,
                                state_dict=model.state_dict()), path)
                loaded = one.load_model(path, device='cpu')
                scalar = [one.predict_scalar(loaded, c) for c in configs.T]
                np.testing.assert_allclose(one.predict_values(loaded, configs, chunk_size=2), scalar,
                                           rtol=2e-6, atol=2e-6)
                np.testing.assert_allclose(one.predict_values(loaded, configs[:, ::-1]), scalar[::-1],
                                           rtol=2e-6, atol=2e-6)
                self.assertEqual(one.predict_values(loaded, configs[:, :0]).shape, (0,))
                for bad in (np.zeros((2, 3), dtype=int), configs.astype(float), configs - 3):
                    with self.assertRaises(ValueError):
                        one.predict_values(loaded, bad)
                for chunk in (0, -1, 1.5, True):
                    with self.assertRaises(ValueError):
                        one.predict_values(loaded, configs, chunk_size=chunk)
                with self.assertRaises(ValueError):
                    one.predict_scalar(loaded, [1, 1])
                with self.assertRaises(ValueError):
                    one.predict_values(loaded, configs, nmax=3)

    def test_all_2d_architectures_dtypes_boundaries(self):
        configs = np.arange(24, dtype=np.int64).reshape(6, 4) % 3
        for arch in models.ARCH_BUILDERS:
            for dtype in (torch.float32, torch.float64):
                boundaries = [(True, True)] if arch.startswith('Group') else [(True, True), (False, False), (False, True)]
                for px, py in boundaries:
                    with self.subTest(arch=arch, dtype=dtype, boundaries=(px, py)):
                        # Outputs must stay under the existing guarded 2D root.
                        with tempfile.TemporaryDirectory(dir=models.ROOT, prefix='.bridge-test-') as directory:
                            model = models.build_model(arch, 3, 2, 2, pbc_x=px, pbc_y=py, **settings(arch)).to(dtype=dtype).eval()
                            path = models.save_model(Path(directory) / 'model.pt', model, Ntot=4, t=.7, U=2.3, mu=.2)
                            loaded, metadata = two.load_model(path)
                            self.assertEqual(next(loaded.parameters()).dtype, dtype)
                            self.assertEqual((metadata['Lx'], metadata['Ly'], metadata['pbc_x'], metadata['pbc_y']), (3, 2, px, py))
                            self.assertEqual((metadata['Ntot'], metadata['t'], metadata['U'], metadata['mu']), (4, .7, 2.3, .2))
                            expected = model(torch.nn.functional.one_hot(torch.tensor(configs.T.copy()), 3).to(dtype)).detach().numpy()
                            np.testing.assert_allclose(two.predict_values(loaded, configs, chunk_size=3), expected, rtol=2e-6, atol=2e-6)
                            np.testing.assert_allclose([two.predict_scalar(loaded, c) for c in configs.T], expected, rtol=2e-6, atol=2e-6)
                            if arch.startswith('Group'):
                                self.assertIn('D2_rotations_reflections', metadata['spatial_symmetries'])
        with self.assertRaises(ValueError):
            models.build_model('GroupEquivariantConv2DNet', 3, 2, pbc=False)

    def test_namespace_routes_fresh_processes(self):
        snippets = [
            # A foreign generic models module is harmless and must remain untouched.
            "import types; foreign=types.ModuleType('models'); foreign.__file__=None; sys.modules['models']=foreign; from pythonTest.bosehubbard2d import lattice,square_group,exact_diagonalization,models,vmc,observables,leakage_analysis,train,train_ed,evaluate; assert sys.modules['models'] is foreign; assert str(root/'pythonTest/2D') not in sys.path",
            "sys.path.insert(0,str(root/'pythonTest/2D')); import models; from pythonTest.bosehubbard2d import models as other",
            "from pythonTest.bosehubbard2d import models; sys.path.insert(0,str(root/'pythonTest/2D')); import models as other",
        ]
        for index, snippet in enumerate(snippets):
            result = subprocess.run([sys.executable, '-c', 'import sys; from pathlib import Path; root=Path.cwd(); '+snippet], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode == 0, index == 0, result.stderr)
            if index:
                self.assertIn('ImportError', result.stderr)

    def test_workflows_and_parent_state_isolation(self):
        torch.manual_seed(45)
        state = torch.random.get_rng_state().clone()
        flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        with tempfile.TemporaryDirectory(dir=models.ROOT, prefix='.bridge-workflow-') as directory:
            directory = Path(directory)
            common = dict(Lx=3, Ly=1, Ntot=2, Nmax=2, epochs=1, device='cpu', pbc=False)
            foreign = directory / 'foreign'
            (foreign / 'src').mkdir(parents=True)
            (foreign / 'src/__init__.py').write_text("raise RuntimeError('unrelated src package imported')\n")
            with patch.dict(os.environ, {'PYTHONPATH': str(foreign)}):
                ed = two.run_workflow('train_ed', dict(common, width=4, eval_every=1, output=str(directory/'ed.pt')))
            self.assertEqual(ed['report']['status'], 'completed')
            vmc = two.run_workflow('train', dict(common, hidden_dim=4, walkers=2, steps=1, burnin=0, output=str(directory/'vmc.pt')))
            self.assertTrue(Path(vmc['paths']['checkpoint']).is_file())
            result = two.run_workflow('evaluate', dict(checkpoint=vmc['paths']['checkpoint'], walkers=2, steps=2, block_length=1, burnin=0, device='cpu', output=str(directory/'eval.json')))
            self.assertEqual(result['report']['status'], 'completed')
            self.assertTrue(Path(result['paths']['raw_samples']).is_file())
            with self.assertRaises(RuntimeError):
                two.run_workflow('train_ed', dict(common, output=str(directory/'ed.pt')))
            with self.assertRaises(RuntimeError):
                two.run_workflow('train', dict(common, output='/tmp/escape.pt'))
            for device in ('cuda:256', 'cuda:99999'):
                output = directory / ('invalid-' + device.replace(':','-') + '.pt')
                with self.assertRaisesRegex(RuntimeError, 'requested CUDA device is unavailable'):
                    two.run_workflow('train', dict(common, device=device, output=str(output)))
                self.assertFalse(output.exists())

        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(flags, (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))

    def test_open_chain_physics_conversion(self):
        from pythonTest.BoseHubbardED import build_bose_hubbard_hamiltonian as chain
        from pythonTest.bosehubbard2d.exact_diagonalization import build_bose_hubbard_hamiltonian as grid
        h1, basis1 = chain(4, 3, 2, t=.7, U=1.3, mu=.2, pbc=False)
        h2, basis2 = grid(4, 1, Ntot=3, Nmax=2, t=.7, U=2.6, mu=.2, pbc=False)
        self.assertEqual(list(map(tuple, basis1)), list(map(tuple, basis2)))
        np.testing.assert_allclose(h1, h2.toarray(), atol=1e-14, rtol=1e-14)

    def test_devices(self):
        self.assertEqual(resolve_device('auto').type, 'cuda' if torch.cuda.is_available() else 'cpu')
        for value in ('cpu:0', 'mps', 'cuda:256', 'cuda:99999', 'cuda:-1', 'invalid'):
            with self.assertRaises(ValueError):
                resolve_device(value)

    @unittest.skipUnless(os.environ.get('BHNQS_TEST_CUDA') == '1', 'set BHNQS_TEST_CUDA=1 for actual CUDA')
    def test_actual_cuda(self):
        self.assertTrue(torch.cuda.is_available(), 'CUDA gate enabled but no actual CUDA device')
        configs = np.arange(24).reshape(6, 4) % 3
        for arch in models.ARCH_BUILDERS:
            model = models.build_model(arch, 3, 2, 2, **settings(arch)).eval()
            expected = two.predict_values(model, configs)
            model.to(resolve_device('cuda:0'))
            self.assertEqual(next(model.parameters()).device.type, 'cuda')
            np.testing.assert_allclose(two.predict_values(model, configs), expected, rtol=3e-5, atol=3e-5)
        for arch in one.ARCH_BUILDERS:
            with tempfile.TemporaryDirectory() as directory:
                model = one.build_model(arch, 6, 2, 4).eval()
                path = Path(directory)/'one.pt'
                torch.save(dict(arch=arch,n_sites=6,nmax=2,hidden_dim=4,state_dict=model.state_dict()),path)
                expected = one.predict_values(model, configs)
                deserialize_only = one.load_model(path, map_location='cuda:0')
                self.assertEqual(next(deserialize_only.parameters()).device.type, 'cpu')
                np.testing.assert_allclose(one.predict_values(deserialize_only, configs), expected, rtol=3e-5, atol=3e-5)
                loaded = one.load_model(path, device='cuda:0')
                self.assertEqual(next(loaded.parameters()).device.type, 'cuda')
                np.testing.assert_allclose(one.predict_values(loaded, configs), expected, rtol=3e-5, atol=3e-5)
                self.assertAlmostEqual(one.predict_scalar(loaded, configs[:,0]), expected[0], places=4)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
