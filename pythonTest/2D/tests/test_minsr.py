"""Independent spectral/SVD math, SGD, guards and optional MinSR CLI checks."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train
import vmc
from models import GENERAL_ARCHITECTURES, build_model, load_model


def svd_reference(J, E, r=1e-12, a=0.0):
    """Independent singular-basis expression sigma*g(sigma**2)."""
    A = (J-J.mean(0)) / len(J)**0.5
    e = (E-E.mean()) / len(J)**0.5
    U, singular, Vh = torch.linalg.svd(A, full_matrices=False)
    cutoff = r * singular.square().max() + a
    gain = torch.zeros_like(singular)
    active = singular > 0
    gain[active] = 1/singular[active] / (1+(cutoff/singular[active].square())**6)
    return Vh.T @ (gain*(U.T@e))


class MinSRTests(unittest.TestCase):
    def test_svd_reference_both_spaces_rank_deficiency_and_normalization(self):
        torch.manual_seed(246)
        for M, P in ((5, 9), (11, 4), (7, 7)):
            J = torch.randn(M, P, dtype=torch.float64)
            J[:, -1] = 2
            J[:, 1] = J[:, 0]  # dependent columns as well as a gauge direction
            E = torch.randn(M, dtype=torch.float64)
            for r, a in ((1e-12, 0.), (1e-6, .003), (0., .02)):
                expected = svd_reference(J, E, r, a)
                for solver in ('auto', 'sample', 'parameter'):
                    actual, info = vmc.sr_gradient(J, E, regularizer='minsr', solver=solver,
                                                   rpinv=r, apinv=a, return_diagnostics=True)
                    torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)
                    shifted = vmc.sr_gradient(J+7, E-3, regularizer='minsr', solver=solver, rpinv=r, apinv=a)
                    torch.testing.assert_close(actual, shifted, rtol=2e-9, atol=2e-10)
                    repeated = vmc.sr_gradient(J.repeat(3, 1), E.repeat(3), regularizer='minsr', solver=solver, rpinv=r, apinv=a)
                    torch.testing.assert_close(actual, repeated, rtol=2e-9, atol=2e-10)
                    self.assertIsNone(info['tau'])
                    self.assertEqual(info['soft_cutoff_power'], 6)

    def test_soft_filter_values_extremes_zero_and_negative_policy(self):
        spectrum = torch.tensor([-1e-16, 0., 1e-300, .5, 1., 2.], dtype=torch.float64)
        weights, info = vmc._minsr_soft_filter(spectrum, rpinv=0, apinv=1)
        expected = torch.tensor([0., 0., 0., 1/(.5*(1+2**6)), .5,
                                 1/(2*(1+.5**6))], dtype=torch.float64)
        torch.testing.assert_close(weights, expected, atol=1e-15, rtol=1e-14)
        self.assertGreater(weights[3], 0)  # No hard truncation below cutoff.
        self.assertEqual(info['roundoff_negative_eigenvalues'], 1)
        self.assertEqual(info['spectral_cutoff'], 1)
        with self.assertRaisesRegex(RuntimeError, 'materially negative'):
            vmc._minsr_soft_filter(torch.tensor([-1., 2.], dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, 'overflow'):
            vmc._minsr_soft_filter(torch.tensor([1e308], dtype=torch.float64), rpinv=1e308)
        for solver in ('parameter', 'sample'):
            zero, meta = vmc.sr_gradient(torch.ones(4, 6), [1, 2, 3, 4], regularizer='minsr', solver=solver, return_diagnostics=True)
            self.assertEqual(torch.count_nonzero(zero), 0)
            self.assertEqual(meta['spectral_cutoff'], 0)
        # With zero cutoff, nonzero modes use an ordinary pseudoinverse.
        w, _ = vmc._minsr_soft_filter(torch.tensor([0., .5, 2.], dtype=torch.float64), 0, 0)
        torch.testing.assert_close(w, torch.tensor([0., 2., .5], dtype=torch.float64))

    def test_all_five_SGD_steps_and_unused_parameters(self):
        for arch in GENERAL_ARCHITECTURES:
            torch.manual_seed(19)
            settings = {} if arch == 'JastrowNet' else {'n_hidden' if arch == 'RBMNet' else 'hidden_dim': 4}
            model = build_model(arch, 3, 2, Nmax=2, pbc_x=False, pbc_y=True, **settings).double()
            model.register_parameter('unused', nn.Parameter(torch.tensor([.2, -.1], dtype=torch.float64)))
            states = vmc.generate_initial_states(6, 4, 2, 7)
            E = torch.randn(7, dtype=torch.float64)
            J = vmc.compute_log_psi_jacobian(model, states)
            expected = svd_reference(J, E, 1e-6)
            before = torch.cat([p.detach().flatten() for p in model.parameters()])
            direction, info = vmc.sr_update(model, states, E, torch.optim.SGD(model.parameters(), lr=.002),
                                          regularizer='minsr', rpinv=1e-6, jacobian_method='vmap',
                                          jacobian_batch_size=8, return_diagnostics=True)
            actual = torch.cat([p.detach().flatten() for p in model.parameters()])
            torch.testing.assert_close(direction, expected, rtol=1e-8, atol=1e-8)
            torch.testing.assert_close(actual, before-.002*expected, rtol=1e-10, atol=1e-10)
            self.assertEqual(info['regularizer'], 'minsr')
            self.assertTrue(torch.equal(model.unused, torch.tensor([.2, -.1], dtype=torch.float64)))

    def test_guard_order_and_eigensolve_budget(self):
        model = build_model('FCNet', 2, 1, Nmax=2, hidden_dim=3).double()
        states = torch.tensor([[0, 2], [1, 1], [2, 0]])
        ridge = vmc.check_sr_budget(model, len(states))
        spectral = vmc.check_sr_budget(model, len(states), regularizer='minsr')
        self.assertGreater(spectral['estimated_bytes'], ridge['estimated_bytes'])
        self.assertGreater(spectral['additional_eigh_workspace_bytes'], 0)
        with patch.object(model, 'forward', side_effect=AssertionError('forward before guard')), \
                patch.object(torch.linalg, 'eigh', side_effect=AssertionError('eigh before guard')):
            with self.assertRaisesRegex(ValueError, 'workspace'):
                vmc.sr_update(model, states, [1., 2., 3.], torch.optim.SGD(model.parameters(), lr=.01),
                              max_sr_bytes=ridge['estimated_bytes'], regularizer='minsr')
            with self.assertRaisesRegex(ValueError, 'workspace'):
                vmc.sr_gradient(torch.ones(3, 4), [1, 2, 3], regularizer='minsr', max_sr_bytes=1)

    def test_invalid_cutoffs_and_input_fail_before_J(self):
        model = build_model('FCNet', 2, 1, Nmax=2, hidden_dim=2).double()
        opt = torch.optim.SGD(model.parameters(), lr=.01)
        with patch.object(vmc, 'compute_log_psi_jacobian', side_effect=AssertionError('Jacobian before validation')):
            for key in ('rpinv', 'apinv'):
                for bad in (-1., float('inf'), float('nan'), True):
                    with self.assertRaises(ValueError):
                        vmc.sr_update(model, [[0, 2]], [1.], opt, regularizer='minsr', **{key: bad})
            with self.assertRaisesRegex(ValueError, 'regularizer'):
                vmc.sr_update(model, [[0, 2]], [1.], opt, regularizer='bad')
            with self.assertRaisesRegex(ValueError, 'SGD'):
                vmc.sr_update(model, [[0, 2]], [1.], torch.optim.Adam(model.parameters()), regularizer='minsr')
        for J, E in ((torch.ones(2, 3, dtype=torch.complex128), [1., 2.]),
                     (torch.ones(2, 3), [1+1j, 2]),
                     (torch.tensor([[float('nan')]]), [1.]),
                     (torch.ones(1, 2), [float('inf')])):
            with self.assertRaises(ValueError):
                vmc.sr_gradient(J, E, regularizer='minsr')

    def test_ridge_default_and_explicit_are_bitwise_equal(self):
        torch.manual_seed(3)
        J = torch.randn(9, 13, dtype=torch.float64)
        E = torch.randn(9, dtype=torch.float64)
        for solver in ('auto', 'parameter', 'sample'):
            default = vmc.sr_gradient(J, E, .013, solver=solver)
            explicit = vmc.sr_gradient(J, E, .013, solver=solver, regularizer='ridge')
            self.assertTrue(torch.equal(default, explicit))

    def test_cli_preflight_and_saved_minsr_metadata(self):
        bad = [
            ['--sr-regularizer', 'minsr'],
            ['--update-method', 'sr', '--sr-regularizer', 'minsr', '--optimizer', 'adam'],
            ['--update-method', 'sr', '--sr-regularizer', 'minsr', '--tau-end', '.001'],
            ['--minsr-rpinv', '-1'], ['--minsr-apinv', 'nan'],
        ]
        for options in bad:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(['--device', 'cpu', *options])
        self.assertEqual(train.parse_args(['--device', 'cpu']).sr_regularizer, 'ridge')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='test-minsr-') as directory:
            output = Path(directory)/'checkpoint.pt'
            with redirect_stdout(io.StringIO()):
                report = train.main(['--arch','FCNet','--Lx','2','--Ly','1','--Nmax','2','--Ntot','2',
                                     '--hidden-dim','3','--epochs','2','--walkers','4','--steps','2',
                                     '--burnin','2','--initial-burnin','3','--thin','2','--device','cpu',
                                     '--dtype','float64','--update-method','sr','--sr-regularizer','minsr',
                                     '--minsr-rpinv','1e-8','--jacobian-method','vmap','--jacobian-batch-size','8',
                                     '--output',str(output)])
            saved = json.loads(Path(str(output)+'.json').read_text())
            model, _ = load_model(output)
            self.assertTrue(all(p.isfinite().all() for p in model.parameters()))
            self.assertEqual(saved['sr_regularizer'], 'minsr')
            self.assertEqual(saved['minsr_cutoffs']['rpinv'], 1e-8)
            self.assertIsNone(saved['settings']['tau'])
            self.assertIsNone(saved['schedules']['tau_start'])
            self.assertIsNone(saved['schedules']['tau_end'])
            self.assertIn('unused', saved['schedules']['tau_usage'])
            self.assertIn('spectral', saved['sr_convention'])
            for row in report['history']:
                self.assertIsNone(row['tau'])
                self.assertIsNone(row['sr']['tau'])
                self.assertEqual(row['sr']['regularizer'], 'minsr')
                self.assertIn('soft_effective_modes', row['sr'])
            self.assertEqual(saved['final_measurement']['measurement_timing'], 'after_final_update_from_saved_checkpoint')


if __name__ == '__main__':
    unittest.main()
