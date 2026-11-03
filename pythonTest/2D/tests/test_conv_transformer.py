"""Faithful hybrid geometry, deterministic forwards, derivatives and compatibility."""
from contextlib import redirect_stderr
import hashlib,io,json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models import Conv2DTransformerNet,BASELINE_ARCHITECTURES,ARCH_BUILDERS,build_model,load_model,save_model
from vmc import compute_log_psi_jacobian
import train,train_ed

R=ROOT/'benchmarks/results/2026-11-01-backup-extension'


class HybridTests(unittest.TestCase):
    def test_shapes_determinism_gradients_and_fixed_architecture(self):
        torch.manual_seed(11)
        for shape in ((3,2),(1,1)):
            model=Conv2DTransformerNet(*shape,Nmax=2,hidden_dim=8).double()
            states=torch.randint(3,(2,shape[0]*shape[1]));x=F.one_hot(states,3).double()
            model.train();a=model(x);b=model(x)
            self.assertEqual(a.shape,(2,));self.assertTrue(torch.equal(a,b))
            model.eval();torch.testing.assert_close(model(x),a,rtol=1e-12,atol=1e-12)
            with torch.no_grad():torch.testing.assert_close(model(x),a,rtol=1e-12,atol=1e-12)
            model.train();model(x[:1]).square().sum().backward()
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
            self.assertEqual(model.transformer.self_attn.num_heads,4)
            self.assertEqual(model.transformer.linear1.out_features,16)
            self.assertEqual(model.fc[1].in_features,shape[0]*shape[1]*8)
            self.assertTrue(all(m.p==0 for m in model.modules() if isinstance(m,nn.Dropout)))
            self.assertFalse(any('embedding' in n for n,_ in model.named_parameters()))

    def test_manual_boundary_padding_and_physical_token_order(self):
        torch.manual_seed(13)
        x=F.one_hot(torch.tensor([[0,1,2,0,2,1,0,2,1,0,2,1]]),3).double()
        for px,py in ((False,False),(True,True),(False,True),(True,False)):
            model=Conv2DTransformerNet(4,3,Nmax=2,hidden_dim=4,pbc_x=px,pbc_y=py).double()
            expected=torch.zeros(1,12,4,dtype=torch.float64)
            conv=model.conv[0]
            # Independent coordinate-based stencil; outside open axes is a zero
            # FEATURE vector, not an occupation0 one-hot external site.
            for y in range(3):
                for xx in range(4):
                    value=conv.bias.detach().clone()
                    for ky in range(3):
                        for kx in range(3):
                            yy,xp=y+ky-1,xx+kx-1
                            if (not py and not 0<=yy<3) or (not px and not 0<=xp<4):continue
                            yy,xp=yy%3,xp%4
                            value+=(conv.weight[:,:,ky,kx].detach()*x[0,yy*4+xp]).sum(1)
                    expected[0,y*4+xx]=F.silu(value)
            captured=[]
            def identity(tokens):captured.append(tokens.detach().clone());return tokens
            with patch.object(model.transformer,'forward',side_effect=identity):
                actual=model(x)
            torch.testing.assert_close(captured[0],expected,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(actual,model.fc(expected).squeeze(-1),rtol=1e-12,atol=1e-12)

    def test_vmap_matches_independent_per_sample_autograd(self):
        torch.manual_seed(19)
        states=torch.tensor([[0,0,0,0,0,2],[0,0,0,0,1,1],[1,0,0,0,1,0]])
        for px,py in ((False,True),(True,False)):
            model=Conv2DTransformerNet(3,2,Nmax=2,hidden_dim=4,pbc_x=px,pbc_y=py).double().eval()
            model.register_parameter('unused',nn.Parameter(torch.tensor([.1,-.2],dtype=torch.float64)))
            parameters=list(model.parameters());rows=[]
            for s in states:
                value=model(F.one_hot(s[None],3).double()).sum()
                grads=torch.autograd.grad(value,parameters,allow_unused=True)
                rows.append(torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten() for p,g in zip(parameters,grads)]))
            expected=torch.stack(rows)
            actual=compute_log_psi_jacobian(model,states,jacobian_method='vmap',jacobian_batch_size=128)
            torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-11)

    def test_roundtrip_invalid_width_and_cli_preflight(self):
        model=build_model('Conv2DTransformerNet',3,2,Nmax=2,hidden_dim=8,pbc_x=False,pbc_y=True).double().eval()
        x=F.one_hot(torch.tensor([[0,1,2,0,1,2]]),3).double()
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='test-hybrid-') as directory:
            path=Path(directory)/'model.pt';save_model(path,model,6)
            loaded,meta=load_model(path)
            self.assertEqual(meta['arch'],'Conv2DTransformerNet')
            self.assertEqual(meta['architecture_settings'],{'hidden_dim':8})
            self.assertFalse(meta['pbc_x']);self.assertTrue(meta['pbc_y'])
            self.assertTrue(torch.equal(model(x),loaded(x)))
        for bad in (0,-4,1,6,True,4.):
            with self.assertRaises(ValueError):Conv2DTransformerNet(2,hidden_dim=bad)
        with self.assertRaises(ValueError):model(torch.zeros(1,5,3))
        for parser,options in ((train.parse_args,['--hidden-dim','6']),(train_ed.parse_args,['--width','6'])):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parser(['--arch','Conv2DTransformerNet','--device','cpu',*options])
        self.assertEqual(len(BASELINE_ARCHITECTURES),5)
        self.assertIn('Conv2DTransformerNet',ARCH_BUILDERS)

    def test_five_preextension_predictions_metadata_and_1D_unchanged(self):
        if not all((R/name).is_file() for name in ('pre-extension-baseline-gold.json','pre-extension-baseline-gold.npz')):
            self.skipTest('optional local historical checkpoint archive is not available')
        gold=json.loads((R/'pre-extension-baseline-gold.json').read_text())
        checkpoints=[]
        for row in gold['models']:
            # Restore recorded artifacts under this checkout's2D outputs path.
            recorded=Path(row['checkpoint'])
            if 'outputs' in recorded.parts:
                recorded=ROOT.joinpath(*recorded.parts[recorded.parts.index('outputs'):])
            elif not recorded.is_absolute():
                recorded=ROOT/recorded
            checkpoints.append(recorded)
        if not all(cp.is_file() for cp in checkpoints):
            self.skipTest('optional local historical checkpoints are not available')
        with np.load(R/'pre-extension-baseline-gold.npz') as arrays:
            x=F.one_hot(torch.tensor(arrays['occupations']),5).double()
            for row,cp in zip(gold['models'],checkpoints):
                self.assertEqual(hashlib.sha256(cp.read_bytes()).hexdigest(),row['checkpoint_sha256'])
                model,meta=load_model(cp);self.assertEqual(meta,row['metadata'])
                with torch.no_grad():actual=model.double()(x).numpy()
                np.testing.assert_array_equal(actual,arrays[row['family']])
        for path,digest in gold['protected_1D_source'].items():
            self.assertEqual(hashlib.sha256((ROOT.parents[1]/path).read_bytes()).hexdigest(),digest)


if __name__=='__main__':unittest.main()
