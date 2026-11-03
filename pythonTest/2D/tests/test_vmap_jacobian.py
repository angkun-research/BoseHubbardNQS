"""Opt-in functional vmap Jacobians, guarded allocation and unchanged VJP defaults."""

from contextlib import redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models import ARCH_BUILDERS, build_model, save_model
from exact_diagonalization import generate_basis
from test_warm_vjp import independent_jacobian
import train
import vmc


class VmapJacobianTests(unittest.TestCase):
    def test_all_five_full_jacobians_and_sgd_with_unused_parameters(self):
        torch.manual_seed(91)
        states=torch.tensor(generate_basis(6,2,2)[:17])
        energies=torch.linspace(-1.3,1.7,len(states),dtype=torch.float64).square()
        for arch in ARCH_BUILDERS:
            for px,py in ((False,True),(True,False)):
                settings={} if arch=='JastrowNet' else {'n_hidden' if arch=='RBMNet' else 'hidden_dim':4}
                model=build_model(arch,3,2,Nmax=2,pbc_x=px,pbc_y=py,**settings).double().eval()
                model.register_parameter('unused_parameter',nn.Parameter(torch.tensor([.1,-.2],dtype=torch.float64)))
                original={n:v.clone() for n,v in model.state_dict().items()}
                expected=independent_jacobian(model,states)
                delta=vmc.sr_gradient(expected,energies,tau=.05)
                for chunk in (1,16,64,128):
                    with self.subTest(arch=arch,axes=(px,py),chunk=chunk):
                        J=vmc.compute_log_psi_jacobian(model,states,jacobian_method='vmap',jacobian_batch_size=chunk)
                        torch.testing.assert_close(J,expected,rtol=1e-10,atol=1e-11)
                        self.assertFalse(J.requires_grad)
                        offset=0
                        for name,p in model.named_parameters():
                            if name=='unused_parameter':
                                self.assertEqual(torch.count_nonzero(J[:,offset:offset+p.numel()]),0)
                            offset+=p.numel()
                        candidate=copy.deepcopy(model)
                        before=torch.cat([p.detach().flatten() for p in candidate.parameters()])
                        optimizer=torch.optim.SGD(candidate.parameters(),lr=.02)
                        actual,info=vmc.sr_update(candidate,states,energies,optimizer,.05,jacobian_method='vmap',jacobian_batch_size=chunk,return_diagnostics=True)
                        torch.testing.assert_close(actual,delta,rtol=1e-9,atol=1e-10)
                        after=torch.cat([p.detach().flatten() for p in candidate.parameters()])
                        torch.testing.assert_close(after,before-.02*delta,rtol=1e-9,atol=1e-10)
                        self.assertEqual(info['jacobian_method'],'vmap')
                        self.assertGreater(info['vmap_gradient_and_input_bytes'],0)
                        self.assertEqual(info['batched_vjp_workspace_bytes'],0)
                        self.assertTrue(all(torch.equal(v,original[n]) for n,v in model.state_dict().items()))

    def test_guards_precede_forward_and_jacobian_allocation(self):
        model=build_model('Conv2DNet',2,2,Nmax=2,hidden_dim=2).double()
        states=torch.tensor(generate_basis(4,2,2)[:5])
        base=vmc.check_sr_budget(model,len(states))
        guarded=vmc.check_sr_budget(model,len(states),jacobian_method='vmap',jacobian_batch_size=128)
        P=sum(p.numel() for p in model.parameters())
        self.assertEqual(guarded['estimated_bytes']-base['estimated_bytes'],8*(2*len(states)*P+len(states)*4*3))
        with patch.object(model,'forward',side_effect=AssertionError('forward before guard')):
            with patch.object(torch,'empty',side_effect=AssertionError('allocated before guard')):
                with self.assertRaisesRegex(ValueError,'workspace'):
                    vmc.compute_log_psi_jacobian(model,states,max_sr_bytes=base['estimated_bytes'],jacobian_method='vmap',jacobian_batch_size=128)
                for kwargs in ({'jacobian_method':'bad'},{'jacobian_method':'vmap','jacobian_batch_size':129},
                               {'jacobian_method':'vmap','jacobian_batch_size':True},
                               {'jacobian_method':'vmap','jacobian_batch_size':0},
                               {'jacobian_method':'vmap','jacobian_batch_size':1.5}):
                    with self.assertRaises(ValueError): vmc.compute_log_psi_jacobian(model,states,**kwargs)
        for flag,value in (('--jacobian-batch-size','129'),('--jacobian-method','bad')):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(['--jacobian-method','vmap',flag,value])
        self.assertEqual(train.parse_args(['--jacobian-method','vmap','--jacobian-batch-size','128']).jacobian_batch_size,128)

    def test_validation_and_nonfinite_forward_retained(self):
        model=build_model('FCNet',2,1,Nmax=2,hidden_dim=2).double()
        for states in (torch.tensor([[3,0]]),torch.tensor([[0,0],[1,0]]),torch.tensor([[1.,1.]])):
            with self.assertRaises(ValueError): vmc.compute_log_psi_jacobian(model,states,jacobian_method='vmap')
        with torch.no_grad(): model.net[-1].bias.fill_(float('inf'))
        with self.assertRaisesRegex(RuntimeError,'non-finite model log'):
            vmc.compute_log_psi_jacobian(model,torch.tensor([[1,1]]),jacobian_method='vmap')

    def test_default_vjp_seeded_numerical_and_budget_parity(self):
        torch.manual_seed(12)
        model=build_model('FCNet',2,1,Nmax=2,hidden_dim=3).double()
        states=torch.tensor([[2,0],[1,1],[0,2]])
        old=independent_jacobian(model,states)
        self.assertTrue(torch.equal(vmc.compute_log_psi_jacobian(model,states),old))
        self.assertTrue(torch.equal(vmc.compute_log_psi_jacobian(model,states,jacobian_method='vjp',jacobian_batch_size=1),old))
        P=old.shape[1]; M=len(states); D=min(P,M)
        self.assertEqual(vmc.check_sr_budget(model,M)['estimated_bytes'],8*(3*M*P+8*D**2+8*P+4*M))
        args=train.parse_args([])
        self.assertEqual((args.jacobian_method,args.jacobian_batch_size),('vjp',1))

    def test_warm_start_cli_vmap_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='test-vmap-warm-') as directory:
            source=Path(directory)/'source.pt'; output=Path(directory)/'new.pt'
            model=build_model('Conv2DNet',2,2,Nmax=2,hidden_dim=2,pbc_x=False,pbc_y=True).double()
            save_model(source,model,Ntot=4)
            with redirect_stdout(io.StringIO()):
                report=train.main(['--arch','Conv2DNet','--Lx','2','--Ly','2','--Nmax','2','--hidden-dim','2',
                                   '--no-pbc-x','--pbc-y','--warm-start',str(source),'--output',str(output),
                                   '--update-method','sr','--optimizer','sgd','--device','cpu','--dtype','float64',
                                   '--epochs','1','--walkers','2','--steps','2','--initial-burnin','3','--burnin','1',
                                   '--jacobian-method','vmap','--jacobian-batch-size','128'])
            disk=json.loads(Path(str(output)+'.json').read_text())
            self.assertEqual(disk['settings']['jacobian_method'],'vmap')
            self.assertEqual(disk['history'][0]['sr']['jacobian_method'],'vmap')
            self.assertEqual(disk['sr_budget']['jacobian_batch_rows'],4)
            self.assertEqual(report['final_measurement'],disk['final_measurement'])
            self.assertEqual(disk['initial_equilibration']['total_walker_proposals'],6)

    def test_tf32_flag_default_and_cpu_do_not_change_backends_or_seeded_path(self):
        self.assertFalse(train.parse_args([]).disable_tf32)
        flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='test-tf32-') as directory:
            reports=[]
            for flag,name in (([], 'default.pt'),(['--disable-tf32'],'flag.pt')):
                with redirect_stdout(io.StringIO()):
                    reports.append(train.main(['--Lx','2','--Nmax','2','--hidden-dim','2','--epochs','1',
                                               '--walkers','2','--steps','1','--burnin','1','--device','cpu',
                                               '--output',str(Path(directory)/name),*flag]))
                self.assertEqual((torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32),flags)
            self.assertEqual(reports[0]['history'],reports[1]['history'])
            self.assertEqual(reports[0]['final_measurement'],reports[1]['final_measurement'])
            self.assertFalse(reports[1]['precision']['disable_tf32_applied'])
            self.assertTrue(reports[1]['precision']['disable_tf32_requested'])
            self.assertEqual(reports[1]['precision']['device'],'cpu')


if __name__=='__main__': unittest.main()
