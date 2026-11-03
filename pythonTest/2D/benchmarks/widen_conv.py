#!/usr/bin/env python3
"""Prepare a function-preserving Conv32->64 inference checkpoint, without VMC.

Replicate convolution channels and dense hidden units twice. Divide the dense
incoming weights by2 for replicated inputs and the scalar-head weights by2
for replicated hidden units. Optional zero-sum jitter cancels on identical
replicas, but permits their gradients to differ. CPU-double verification uses
both feasible and short frozen-sampler configurations; no optimizer is resumed.
Run --self-test for exact-forward, gradient/splitting and checkpoint tests.
"""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models import Conv2DNet, load_model, save_model, parameter_count
from vmc import generate_initial_states, sample


def widen_conv(model, jitter=0., seed=123):
    if not isinstance(model,Conv2DNet) or model.architecture_settings['hidden_dim']!=32:
        raise ValueError('expected Conv2DNet hidden_dim32')
    if isinstance(jitter,bool) or not math.isfinite(jitter) or not 0<=jitter<=1e-3:
        raise ValueError('jitter must be finite in [0,1e-3] (absolute normal standard deviation)')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**63:
        raise ValueError('seed must be integer in [0,2**63)')
    # Fork constructor RNG so preparation does not change caller randomness.
    with torch.random.fork_rng(devices=[]):
        wide=Conv2DNet(model.Lx,model.Ly,Nmax=model.Nmax,hidden_dim=64,
                       pbc_x=model.pbc_x,pbc_y=model.pbc_y).double().eval()
    old={name:p.detach().cpu().double() for name,p in model.state_dict().items()}
    generator=torch.Generator(device='cpu').manual_seed(seed)
    with torch.no_grad():
        wide.conv.weight.copy_(old['conv.weight'].repeat(2,1,1,1))
        wide.conv.bias.copy_(old['conv.bias'].repeat(2))
        incoming=old['head.2.weight'].repeat(2,2)/2
        if jitter:
            noise=jitter*torch.randn((64,old['head.2.weight'].shape[1]),generator=generator,dtype=torch.float64)
            incoming+=torch.cat((noise,-noise),dim=1)
        wide.head[2].weight.copy_(incoming)
        wide.head[2].bias.copy_(old['head.2.bias'].repeat(2))
        outgoing=old['head.4.weight'].repeat(1,2)/2
        if jitter:
            noise=jitter*torch.randn((1,32),generator=generator,dtype=torch.float64)
            outgoing+=torch.cat((noise,-noise),dim=1)
        wide.head[4].weight.copy_(outgoing)
        wide.head[4].bias.copy_(old['head.4.bias'])
    return wide


def forward_difference(narrow,wide,states):
    maximum=0.
    with torch.no_grad():
        for block in states.split(64):
            x=F.one_hot(block,narrow.Nmax+1).double()
            a,b=narrow(x),wide(x)
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise RuntimeError('non-finite forward verification')
            maximum=max(maximum,(a-b).abs().max().item())
    if maximum>1e-11:
        raise RuntimeError(f'forward mismatch {maximum} exceeds1e-11')
    return maximum


def confined(path):
    path=(ROOT/path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT) or path==ROOT or path.exists():
        raise ValueError('destination must be a new file under2D; refusing overwrite')
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--report',type=Path,help='default: output path plus .json; all destinations must be new')
    parser.add_argument('--jitter',type=float,default=0.,help='absolute zero-sum noise std,0..1e-3; default exact replica symmetry')
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args(argv)
    if args.self_test:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WidenTests))
        if not result.wasSuccessful(): raise SystemExit(1)
        return {'self_tests':'passed','count':result.testsRun}
    try:
        if args.checkpoint is None or args.output is None: raise ValueError('--checkpoint and --output required')
        source=(ROOT/args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint.resolve()
        if not source.is_relative_to(ROOT) or not source.is_file(): raise ValueError('checkpoint must exist under2D')
        output=confined(args.output)
        destination=confined(args.report if args.report is not None else Path(str(output)+'.json'))
        if output==destination: raise ValueError('checkpoint/report must differ')
        if not math.isfinite(args.jitter) or not 0<=args.jitter<=1e-3 or not 0<=args.seed<2**63:
            raise ValueError('invalid jitter or seed')
    except ValueError as error: parser.error(str(error))
    source_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('models.py','vmc.py','lattice.py','benchmarks/widen_conv.py')}
    checksum=hashlib.sha256(source.read_bytes()).hexdigest()
    narrow,metadata=load_model(source,device='cpu')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=checksum: raise RuntimeError('source changed during load')
    narrow=narrow.double().eval()
    wide=widen_conv(narrow,args.jitter,args.seed)
    torch.manual_seed(args.seed)
    feasible=generate_initial_states(narrow.n_sites,metadata['Ntot'],narrow.Nmax,64,'cpu')
    initial=generate_initial_states(narrow.n_sites,metadata['Ntot'],narrow.Nmax,4,'cpu')
    sampled,_,diagnostics=sample(narrow,initial,16,32,64)
    feasible_difference=forward_difference(narrow,wide,feasible)
    sampled_difference=forward_difference(narrow,wide,sampled)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='.widen-stage-') as directory:
        staged=Path(directory)/'model.pt'
        save_model(staged,wide,metadata['Ntot'],metadata['t'],metadata['U'],metadata['mu'])
        saved,new_metadata=load_model(staged,device='cpu')
        reloaded_difference=forward_difference(narrow,saved,torch.cat((feasible,sampled)))
        for name in ('Lx','Ly','pbc_x','pbc_y','Ntot','Nmax','t','U','mu','hamiltonian_convention','bond_convention','flattening','spatial_symmetries'):
            if metadata[name]!=new_metadata[name]: raise RuntimeError(f'metadata changed: {name}')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=checksum: raise RuntimeError('original changed during preparation')
        report={'status':'prepared_no_training','recorded_at':datetime.now(timezone.utc).isoformat(),
                'command':[sys.executable,str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv)],
                'source_sha256_at_start':source_hashes,'source_checkpoint':str(source),'source_checkpoint_sha256':checksum,
                'source_file_unchanged':True,'output_checkpoint':str(output),
                'output_checkpoint_sha256':hashlib.sha256(staged.read_bytes()).hexdigest(),
                'source_metadata':metadata,'output_metadata':new_metadata,'parameters_before':parameter_count(narrow),
                'parameters_after':parameter_count(wide),'seed':args.seed,'jitter':args.jitter,
                'jitter_definition':'absolute normal std; paired +/- input-channel replica weights per dense unit; paired +/- scalar output replica weights',
                'replication':'conv channels2x; dense hidden units2x; incoming dense /2; scalar output /2; biases duplicated',
                'verification':{'dtype':'CPU float64','tolerance':1e-11,'feasible_count':len(feasible),'sampled_count':len(sampled),
                                'feasible_max_abs_log_difference':feasible_difference,'sampled_max_abs_log_difference':sampled_difference,
                                'saved_checkpoint_max_abs_log_difference':reloaded_difference,
                                'sampling':{'walkers':4,'retained_steps':16,'burnin_proposals_per_walker':32,'thin':1,'diagnostics':diagnostics}},
                'interpretation':'Prepared same Conv architecture at increased width; inference weights only, no optimizer/walkers resumed. Function preservation is algebraic up to rounding; checks do not establish larger-system convergence or mixing. Small-geometry qualification is preserved constructively, not a new independently fitted result.'}
        destination.parent.mkdir(parents=True,exist_ok=True)
        # New destinations only. Check again immediately before publication.
        if output.exists() or destination.exists(): raise RuntimeError('destination appeared; refusing overwrite')
        staged.replace(output)
        with destination.open('x') as stream: json.dump(report,stream,indent=2,allow_nan=False); stream.write('\n')
    print(f'prepared {output}; max forward difference={reloaded_difference:.3g}; report={destination}')
    return report


class WidenTests(unittest.TestCase):
    def test_forward_and_finite_gradients_on_both_cylinders_and_sampled_states(self):
        for px,py in ((False,True),(True,False)):
            torch.manual_seed(19)
            model=Conv2DNet(3,2,Nmax=2,hidden_dim=32,pbc_x=px,pbc_y=py).double()
            states=generate_initial_states(6,6,2,19,'cpu')
            sampled,_,_=sample(model,states[:3],5,4)
            for jitter in (0.,1e-4):
                wide=widen_conv(model,jitter=jitter,seed=12)
                self.assertLessEqual(forward_difference(model,wide,torch.cat((states,sampled))),1e-11)
                loss=(wide(F.one_hot(states,3).double())-torch.linspace(-1,1,len(states))).square().mean()
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in wide.parameters()))

    def test_jitter_preserves_initial_function_but_one_update_splits_replicas(self):
        torch.manual_seed(7)
        model=Conv2DNet(2,3,Nmax=2,hidden_dim=32,pbc_x=False,pbc_y=True).double()
        original={n:t.clone() for n,t in model.state_dict().items()}
        wide=widen_conv(model,jitter=1e-4,seed=9)
        states=generate_initial_states(6,6,2,17,'cpu')
        forward_difference(model,wide,states)
        self.assertTrue(torch.equal(wide.conv.weight[:32],wide.conv.weight[32:]))
        optimizer=torch.optim.SGD(wide.parameters(),lr=.01)
        x=F.one_hot(states,3).double()
        (wide(x)-1.).square().mean().backward(); optimizer.step()
        self.assertGreater((wide.conv.weight[:32]-wide.conv.weight[32:]).abs().max().item(),1e-12)
        self.assertTrue(all(torch.isfinite(p).all() for p in wide.parameters()))
        self.assertTrue(all(torch.equal(t,original[n]) for n,t in model.state_dict().items()))
        for jitter in (-1.,float('nan'),.1):
            with self.assertRaises(ValueError): widen_conv(model,jitter)

    def test_checkpoint_metadata_hash_and_saved_prediction(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='test-widen-') as directory:
            source=Path(directory)/'source.pt'; output=Path(directory)/'new.pt'
            model=Conv2DNet(2,2,Nmax=2,hidden_dim=32,pbc_x=False,pbc_y=True).double()
            save_model(source,model,4,.8,3.4,.2)
            checksum=hashlib.sha256(source.read_bytes()).hexdigest()
            with redirect_stdout(io.StringIO()):
                report=main(['--checkpoint',str(source),'--output',str(output),'--jitter','1e-6'])
            self.assertEqual(report['source_checkpoint_sha256'],checksum)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),checksum)
            self.assertEqual(report['output_metadata']['architecture_settings']['hidden_dim'],64)
            self.assertEqual(report['output_metadata']['U'],3.4)
            self.assertLessEqual(report['verification']['saved_checkpoint_max_abs_log_difference'],1e-11)


if __name__=='__main__': main()
