"""Prepare the single authorized FC64d3/RBM64 width128 capacity branches.

No fitting or core edits. Same-geometry lifts only. FC is exact in real
arithmetic, including zero-sum jitter. RBM appends antithetic small columns,
which add a constant plus O(epsilon**2), not an exact amplitude-preserving lift.
Qualification uses raw ED amplitudes and sparse H, independently of trainer
metrics. No dense ED conversion or target-size sector enumeration.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models import FCNet,RBMNet,load_model,save_model,parameter_count
from train_ed import make_ed_dataset
from vmc import generate_initial_states,check_sr_budget

R=ROOT/'benchmarks/results/2026-10-31-all-model-refinement'
O=ROOT/'outputs/2026-10-31-all-model-refinement'
JITTER=1e-5
EPSILON=.01
SEED=123


def grow(model,seed=SEED,jitter=JITTER,epsilon=EPSILON):
    if (not math.isfinite(jitter) or not 0<jitter<=1e-3
            or not math.isfinite(epsilon) or not 0<epsilon<=.05):
        raise ValueError('invalid jitter/epsilon')
    generator=torch.Generator().manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        if isinstance(model,FCNet) and model.architecture_settings=={'hidden_dim':64,'hidden_layers':3}:
            new=FCNet(model.Lx,model.Ly,Nmax=model.Nmax,hidden_dim=128,hidden_layers=3,
                      pbc_x=model.pbc_x,pbc_y=model.pbc_y).double().eval()
        elif isinstance(model,RBMNet) and model.architecture_settings=={'n_hidden':64}:
            new=RBMNet(model.Lx,model.Ly,Nmax=model.Nmax,n_hidden=128,
                       pbc_x=model.pbc_x,pbc_y=model.pbc_y).double().eval()
        else:
            raise ValueError('only FC64depth3 or RBM64 sources are authorized')
    with torch.no_grad():
        if isinstance(model,FCNet):
            old_layers=[layer for layer in model.net if isinstance(layer,torch.nn.Linear)]
            new_layers=[layer for layer in new.net if isinstance(layer,torch.nn.Linear)]
            for index,(old,layer) in enumerate(zip(old_layers,new_layers)):
                W,b=old.weight.detach().cpu().double(),old.bias.detach().cpu().double()
                if index==0:
                    layer.weight.copy_(W.repeat(2,1));layer.bias.copy_(b.repeat(2))
                else:
                    row_repeats=1 if index==len(old_layers)-1 else 2
                    noise=jitter*torch.randn((row_repeats*W.shape[0],64),generator=generator,dtype=torch.float64)
                    layer.weight.copy_(W.repeat(row_repeats,2)/2+torch.cat((noise,-noise),1))
                    layer.bias.copy_(b.repeat(row_repeats))
        else:
            new.visible_bias.copy_(model.visible_bias.detach().cpu().double())
            new.hidden_bias[:64].copy_(model.hidden_bias.detach().cpu().double());new.hidden_bias[64:].zero_()
            new.W[:,:64].copy_(model.W.detach().cpu().double())
            noise=epsilon*torch.randn((model.W.shape[0],32),generator=generator,dtype=torch.float64)
            new.W[:,64:].copy_(torch.cat((noise,-noise),1))
    return new


@torch.no_grad()
def logs(model,states):
    dtype=next(model.parameters()).dtype
    return torch.cat([model(F.one_hot(block,model.Nmax+1).to(dtype)).double() for block in states.split(256)])


def independent_metrics(log_values,dataset):
    psi=np.exp(log_values.numpy()-log_values.max().item())
    psi/=np.linalg.norm(psi)
    applied=dataset['H']@psi
    energy=float(psi@applied)
    return {'raw_ED_abs_unsquared_overlap':float(abs(np.vdot(dataset['raw_ed_state'],psi))),
            'energy':energy,'energy_error_per_site':float((energy-dataset['energy'])/9),
            'variance':float(np.linalg.norm(applied-energy*psi)**2),'norm':float(np.linalg.norm(psi))}


def preservation(model,new,states,expected_constant):
    delta=logs(new,states)-logs(model,states)-expected_constant
    return {'configuration_count':len(states),'max_abs_log_difference_minus_constant':delta.abs().max().item(),
            'minimum_log_difference_minus_constant':delta.min().item(),
            'maximum_log_difference_minus_constant':delta.max().item(),
            'centered_log_difference_max_abs':(delta-delta.mean()).abs().max().item()}


def prepare(source,label,geometry,dataset=None,frozen=None):
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    old,meta=load_model(source.resolve(),'cpu');old=old.double().eval()
    if (meta['Lx'],meta['Ly'])!=geometry:
        raise ValueError('source geometry mismatch')
    for key,value in {'pbc_x':False,'pbc_y':True,'Ntot':geometry[0]*geometry[1],
                      'Nmax':4,'t':1.,'U':16.74,'mu':0.}.items():
        if meta[key]!=value:raise ValueError(f'source metadata mismatch: {key}')
    new=grow(old)
    torch.manual_seed(123)
    random=generate_initial_states(old.n_sites,meta['Ntot'],old.Nmax,128,'cpu')
    values=random
    if frozen:
        info=json.loads(frozen.read_text())
        with np.load(ROOT/info['raw_samples_path']) as raw:
            array=raw['occupations'].reshape(-1,old.n_sites)
            indices=np.random.default_rng(123).choice(len(array),128,replace=False)
            values=torch.cat((random,torch.tensor(array[indices],dtype=torch.long)))
    expected=0. if isinstance(old,FCNet) else 64*math.log(2)
    check=preservation(old,new,values,expected)
    if isinstance(old,FCNet) and check['max_abs_log_difference_minus_constant']>1e-11:
        raise RuntimeError('FC exact forward lift failed')
    output=O/('qualification' if dataset is not None else 'capacity-prepared')/f'{label}-{geometry[0]}x{geometry[1]}-prepared.pt'
    if output.exists():raise RuntimeError('refuse existing prepared checkpoint')
    record={'label':label,'geometry':list(geometry),'source':str(source),'source_sha256':digest,
            'output':str(output),'zero_optimizer_updates':True,'seed':SEED,
            'jitter':JITTER if isinstance(old,FCNet) else None,'epsilon':EPSILON if isinstance(old,RBMNet) else None,
            'expected_constant_log_shift':expected,'random_and_saved_frozen_forward_check':check,
            'frozen_report':str(frozen) if frozen else None,'source_metadata':meta,
            'parameters_before':parameter_count(old),'parameters_after':parameter_count(new),
            'operation':('Exact depth3 neuron duplication with all downstream inputs dividedby2 and zero-sum incoming-column jitter; no depth/geometry change.'
                         if isinstance(old,FCNet) else 'Copy all64old hidden units/visible biases exactly; append32antithetic small-column pairs, zero new hidden biases; constant64log2 plus O(epsilon²), not exactfunction-preserving.')}
    if isinstance(old,RBMNet):
        assert torch.equal(new.W[:,:64],old.W)
        assert torch.equal(new.visible_bias,old.visible_bias)
        assert torch.equal(new.hidden_bias[:64],old.hidden_bias)
        assert torch.count_nonzero(new.hidden_bias[64:])==0
        assert torch.equal(new.W[:,64:96],-new.W[:,96:128])
    if dataset is not None:
        basis=torch.as_tensor(dataset['basis'],dtype=torch.long)
        before,after=logs(old,basis),logs(new,basis)
        full=preservation(old,new,basis,expected)
        if isinstance(old,FCNet) and full['max_abs_log_difference_minus_constant']>1e-11:
            raise RuntimeError('FC full-sector exact lift failed')
        record.update(full_sector_forward_check=full,raw_ED_before=independent_metrics(before,dataset),
                      raw_ED_after=independent_metrics(after,dataset))
        # Verify the quantized weights that will actually initialize FP32 VMC,
        # normalizing in float64. Exact-real lift is not bitwise FP32 forward.
        new32=grow(old).float();old32=old.float()
        after32=logs(new32,basis)
        record['float32_forward_check']=preservation(old32,new32,basis,expected)
        record['raw_ED_prepared_float32']=independent_metrics(after32,dataset)
        record['qualification_threshold_abs_overlap']=.99
        record['qualification_passed']=(record['raw_ED_after']['raw_ED_abs_unsquared_overlap']>=.99
                                        and record['raw_ED_prepared_float32']['raw_ED_abs_unsquared_overlap']>=.99)
        if not record['qualification_passed']:raise RuntimeError('width128 variant fails rawED qualification')
    output.parent.mkdir(parents=True,exist_ok=True)
    save_model(output,new,meta['Ntot'],meta['t'],meta['U'],meta['mu'])
    saved,newmeta=load_model(output,'cpu')
    assert torch.equal(logs(saved,values),logs(new,values))
    for key in ('Lx','Ly','Nmax','Ntot','pbc_x','pbc_y','t','U','mu','hamiltonian_convention',
                'bond_convention','flattening','spatial_symmetries'):
        assert meta[key]==newmeta[key],key
    if isinstance(new,FCNet):assert newmeta['architecture_settings']=={'hidden_dim':128,'hidden_layers':3}
    else:assert newmeta['architecture_settings']=={'n_hidden':128}
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    record.update(source_file_unchanged=True,output_metadata=newmeta,
                  output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    if dataset is None:
        record['ridge_SR_budget']=check_sr_budget(new,2048 if isinstance(new,FCNet) else 4096,
                                                 max_sr_bytes=4*1024**3,jacobian_method='vmap',jacobian_batch_size=128)
    return record


class PreparationTests(unittest.TestCase):
    def test_fc_all_layers_exact_and_one_update_splits_replicas(self):
        torch.manual_seed(8)
        old=FCNet(3,2,Nmax=2,hidden_dim=64,hidden_layers=3,pbc_x=False,pbc_y=True).double()
        new=grow(old)
        states=generate_initial_states(6,6,2,31)
        self.assertLess(preservation(old,new,states,0)['max_abs_log_difference_minus_constant'],1e-11)
        self.assertTrue(torch.equal(new.net[0].weight[:64],new.net[0].weight[64:]))
        loss=(new(F.one_hot(states,3).double())-torch.linspace(-1,1,len(states))).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in new.parameters()))
        torch.optim.SGD(new.parameters(),lr=.01).step()
        self.assertGreater((new.net[0].weight[:64]-new.net[0].weight[64:]).abs().max().item(),1e-12)

    def test_rbm_exact_old_copies_antithetic_and_quadratic_limit(self):
        torch.manual_seed(9)
        old=RBMNet(3,2,Nmax=2,n_hidden=64,pbc_x=False,pbc_y=True).double()
        states=generate_initial_states(6,6,2,23)
        full,half=grow(old,epsilon=.01),grow(old,epsilon=.005)
        self.assertTrue(torch.equal(full.W[:,:64],old.W))
        self.assertTrue(torch.equal(full.W[:,64:96],-full.W[:,96:]))
        self.assertEqual(torch.count_nonzero(full.hidden_bias[64:]),0)
        delta=logs(full,states)-logs(old,states)-64*math.log(2)
        halfdelta=logs(half,states)-logs(old,states)-64*math.log(2)
        self.assertLess((delta-4*halfdelta).abs().max().item(),1e-6)
        full(F.one_hot(states,3).double()).square().mean().backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in full.parameters()))
        with self.assertRaises(ValueError):grow(old,epsilon=float('nan'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.self_test:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PreparationTests))
        if not result.wasSuccessful():raise SystemExit(1)
        return
    path=R/'prepared-width128-qualification.json'
    if path.exists():raise RuntimeError('refuse existing preparation record')
    record={'status':'running','recorded_at':datetime.now(timezone.utc).isoformat(),
            'command':[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],
            'source_sha256_at_start':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                ('models.py','vmc.py','train.py','train_ed.py','lattice.py','exact_diagonalization.py')},
            'helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':[],
            'admission':'RawED absolute unsquared overlap>=.99 for corresponding3x3variants, source-preservation and exactmetadata checks, then explicit checkerapproval beforetargetsizefits. Qualification is not a target-size fidelity/capacity guarantee.',
            'checker_requested_record':'checker-width128-qualification.json'}
    def persist():path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    persist()
    try:
        data=make_ed_dataset(3,3,Ntot=9,Nmax=4,t=1,U=16.74,mu=0,pbc_x=False,pbc_y=True,dtype=torch.float64)
        record['ED_reference']={'dimension':len(data['basis']),'energy':data['energy'],'geometry':[3,3],
                                'method':'guarded sparseED, no denseconversion, raw normalized ED vector, CPUfloat64'}
        cases=[
            (ROOT/'outputs/2026-10-31-cylinder-pilots/FCNet64d3-3x3-cylinder-supervised-e10000.pt','FC128d3',(3,3),True,None),
            (O/'FC64d3-6x4-lower-ridge500-seed5791.pt','FC128d3',(6,4),False,O/'frozen/FC64d3-6x4-lower-ridge500-seed321.json'),
            (O/'qualification/RBM64-3x3-prepared.pt','RBM128',(3,3),True,None),
            (O/'RBM64-4x4-matched-ridge300-seed7913.pt','RBM128',(4,4),False,O/'frozen/RBM64-4x4-matched-ridge300-seed321.json')]
        for source,label,geometry,qualify,frozen in cases:
            record['cases'].append(prepare(source,label,geometry,data if qualify else None,frozen));persist()
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in record['source_sha256_at_start'].items())
        record['status']='qualified_prepared_waiting_explicit_checker_approval'
    except BaseException as error:
        record.update(status='failed_stopped',error=f'{type(error).__name__}: {error}');persist();raise
    persist();print(path,flush=True)


if __name__=='__main__':main()
