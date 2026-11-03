"""One square-torus Adam/log-MSE qualification; optional exact orbit reduction.

The geometry/physics and objective are fixed. Weighted338-representative loss
uses full-sector normalization and full-sector target moments, including the
within-orbit target variance (even ED roundoff is not silently dropped).
Final saved weights are independently evaluated on ALL19855 configurations.
No VMC, infidelity objective or optimizer/architecture grid is implemented here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import GroupEquivariantConv2DNet, load_model, parameter_count, save_model
from square_group import space_group_pullbacks
from train_ed import make_ed_dataset, normalized_log_mse
from periodic_observables import physical_bonds, sparse_bond_operators, check_operator_H, state_observables

RESULTS = ROOT/'benchmarks/results/2026-11-02-square-symmetry'
SOURCES = ('models.py', 'square_group.py', 'train_ed.py', 'exact_diagonalization.py',
           'lattice.py', 'benchmarks/train_square_group.py', 'benchmarks/periodic_observables.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_orbit_data(dataset):
    """Collapse input orbits, retaining multiplicities and full target moments."""
    meta = dataset['metadata']
    if meta['Lx'] != meta['Ly'] or not meta['pbc_x'] or not meta['pbc_y']:
        raise ValueError('orbit data require a square torus')
    basis = np.asarray(dataset['basis'], dtype=np.int64)
    powers = (meta['Nmax']+1)**np.arange(basis.shape[1]-1,-1,-1,dtype=np.int64)
    codes = basis@powers
    if np.any(np.diff(codes) <= 0):
        raise ValueError('basis must be ordered and unique')
    representative = np.arange(len(basis))
    for permutation in space_group_pullbacks(meta['Lx']).numpy():
        transformed_codes = basis[:,permutation]@powers
        indices = np.searchsorted(codes,transformed_codes)
        if np.any(indices >= len(basis)) or not np.array_equal(codes[indices],transformed_codes):
            raise ValueError('basis not closed under space group')
        representative = np.minimum(representative,indices)
    reps,labels,counts = np.unique(representative,return_inverse=True,return_counts=True)
    labels = torch.from_numpy(labels)
    multiplicities = torch.from_numpy(counts).double()
    target = dataset['target_log'].double()
    target = target - .5*torch.logsumexp(2*target,dim=0)
    means = torch.zeros(len(reps),dtype=torch.float64).scatter_add_(0,labels,target)/multiplicities
    variances = torch.zeros_like(means).scatter_add_(0,labels,(target-means[labels]).square())/multiplicities
    return {'representatives':torch.from_numpy(reps),'labels':labels,'multiplicities':multiplicities,
            'target_mean':means,'target_variance':variances,'x':dataset['x'][reps]}


def orbit_log_mse(pred_log, multiplicities, target_mean, target_variance):
    """Full-sector log-MSE, for invariant model logs on orbit representatives."""
    if pred_log.ndim != 1 or not len(pred_log) or any(x.shape != pred_log.shape for x in (multiplicities,target_mean,target_variance)):
        raise ValueError('orbit arrays must be matching nonempty vectors')
    if (not all(torch.isfinite(x).all() for x in (pred_log,multiplicities,target_mean,target_variance))
            or (multiplicities <= 0).any() or (target_variance < 0).any()):
        raise ValueError('orbit values must be finite, positive multiplicities, nonnegative variance')
    weights = multiplicities.to(device=pred_log.device,dtype=pred_log.dtype)
    means = target_mean.to(device=pred_log.device,dtype=pred_log.dtype)
    variance = target_variance.to(device=pred_log.device,dtype=pred_log.dtype)
    normalized = pred_log - .5*torch.logsumexp(2*pred_log+weights.log(),dim=0)
    return (weights*((normalized-means).square()+variance)).sum()/weights.sum()


@torch.no_grad()
def metrics(logs, dataset):
    """Physical metrics from a complete sector vector, never floored ED state."""
    logs = logs.detach().double().cpu()
    normalized = logs-.5*torch.logsumexp(2*logs,dim=0)
    psi = normalized.exp().numpy()
    applied = dataset['H']@psi
    energy = float(psi@applied)
    overlap = float(abs(dataset['raw_ed_state']@psi))
    return {'normalized_log_mse':normalized_log_mse(logs,dataset['target_log'].double()).item(),
            'F_abs':overlap,'energy':energy,'energy_per_site':energy/9,
            'energy_error_per_site':(energy-dataset['energy'])/9,
            'variance':float(np.linalg.norm(applied-energy*psi)**2),
            'sector_norm':float(np.linalg.norm(psi)),'working_F99_passed':bool(overlap>=.99)}


@torch.no_grad()
def full_logs(model,x,batch=256):
    parameter=next(model.parameters())
    return torch.cat([model(block.to(device=parameter.device,dtype=parameter.dtype)).double().cpu()
                      for block in x.split(batch)])


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=10000)
    parser.add_argument('--width',type=int,default=16)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--device',choices=['cpu','cuda:1'],default='cpu')
    parser.add_argument('--dtype',choices=['float32','float64'],default='float64')
    parser.add_argument('--eval-every',type=int,default=100)
    parser.add_argument('--output',type=Path,default=Path('outputs/2026-11-02-square-symmetry/GroupEquivariantConv2DNet16-seed123.pt'))
    args=parser.parse_args(argv)
    if not 1<=args.epochs<=10000 or args.width<=0 or args.eval_every<=0 or not 0<=args.seed<2**63 or not math.isfinite(args.lr) or args.lr<=0:
        parser.error('invalid bounded epoch/width/lr/seed/reporting settings')
    args.output=(ROOT/args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    if not args.output.is_relative_to(ROOT/'outputs') or args.output.exists() or Path(str(args.output)+'.json').exists():
        parser.error('output must be new under2D/outputs')
    return args


def main(argv=None):
    args=parse_args(argv)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device=torch.device(args.device)
    if device.type=='cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count()<2:
            raise ValueError('GPU1 unavailable; no GPU0 fallback')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    precision={'device':str(device),'forward_dtype':args.dtype,'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
               'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'final_evaluation':'CPUfloat64, ALL19855 states'}
    source_hashes={name:sha(ROOT/name) for name in SOURCES}
    started=time.perf_counter()
    RESULTS.mkdir(parents=True,exist_ok=True)
    report_path=RESULTS/(args.output.stem+'-trainer.json')
    if report_path.exists(): raise ValueError('refuse existing retained report')
    report={'status':'initializing','recorded_at':datetime.now(timezone.utc).isoformat(),
            'command':[sys.executable,str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv)],
            'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'source_sha256_at_start':source_hashes,'precision':precision,'history':[],
            'objective':'Adam, globally normalized full-sector uniform log-MSE via exact multiplicity-weighted orbit moments; no infidelity',
            'selection':'strict best post-update log-MSE including initialization, ties earliest; not selected by F/energy',
            'stop':'fixed budget; no early stop, no tuning grid',
            'snapshot_scope':'Full-sector physical metrics from invariant representative logs expanded to all states; final checkpoint also directly forwards ALLstates independently.',
            'working_gate':'absolute unsquared F>=.99; energy/site and variance independently reported, no capacity claim on miss'}
    def persist():
        tmp=report_path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(report_path)
    persist()
    try:
        dataset=make_ed_dataset(3,3,9,4,dtype=torch.float64)
        orbit=make_orbit_data(dataset)
        if len(dataset['basis'])!=19855 or len(orbit['representatives'])!=338:
            raise RuntimeError('unexpected full/orbit dimensions')
        model=GroupEquivariantConv2DNet(3,hidden_dim=args.width).to(device=device,dtype=getattr(torch,args.dtype))
        optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
        x=orbit['x'].to(device=device,dtype=next(model.parameters()).dtype)
        report.update(status='training',parameter_count=parameter_count(model),dataset=dataset['metadata'],
                      full_dimension=19855,orbit_dimension=338)
        def loss(logs):return orbit_log_mse(logs,orbit['multiplicities'],orbit['target_mean'],orbit['target_variance'])
        with torch.no_grad():
            initial_logs=model(x);best_loss=loss(initial_logs).item()
        best_epoch=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        report['history'].append({'epoch':0,**metrics(initial_logs.double().cpu()[orbit['labels']],dataset)})
        for epoch in range(1,args.epochs+1):
            optimizer.zero_grad(set_to_none=True)
            objective=loss(model(x));objective.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('nonfinite Adam gradient')
            optimizer.step()
            with torch.no_grad():
                post_logs=model(x);post_loss=loss(post_logs).item()
            if post_loss<best_loss:
                best_epoch,best_loss=epoch,post_loss
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch%args.eval_every==0 or epoch==args.epochs:
                snapshot={'epoch':epoch,**metrics(post_logs.double().cpu()[orbit['labels']],dataset)}
                report['history'].append(snapshot);persist()
                print(f"epoch={epoch} log_mse={snapshot['normalized_log_mse']:.9g} F_abs={snapshot['F_abs']:.10f} energy/site={snapshot['energy_per_site']:.10g} variance={snapshot['variance']:.8g}",flush=True)
        report.update(best_epoch=best_epoch,best_training_loss=best_loss,final_metrics=report['history'][-1],
                      training_wall_seconds=time.perf_counter()-started)
        model.load_state_dict(best_state);model=model.cpu().double().eval()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output.parent,prefix='.group-ed-stage-') as directory:
            staged=Path(directory)/'model.pt';save_model(staged,model,9)
            loaded,metadata=load_model(staged,'cpu')
            logs=full_logs(loaded,dataset['x'])
            representative_logs=full_logs(loaded,orbit['x'])
            discrepancy=(logs-representative_logs[orbit['labels']]).abs().max().item()
            if discrepancy>1e-10:raise RuntimeError('final full-basis invariance check failed')
            selected=metrics(logs,dataset)
            psi=(logs-.5*torch.logsumexp(2*logs,dim=0)).exp().numpy()
            basis=np.asarray(dataset['basis'],dtype=np.int64)
            bonds=physical_bonds(3,3);operators=sparse_bond_operators(basis,4,bonds)
            operator_error=check_operator_H(dataset['H'],basis,operators,1.,16.74,0.)
            obs=state_observables(psi,basis,dataset['H'],bonds,operators=operators)
            normalized=logs-.5*torch.logsumexp(2*logs,dim=0)
            raw=np.abs(dataset['raw_ed_state'])
            reps=orbit['representatives'].numpy()
            report.update(status='completed',best_checkpoint_metrics=selected,checkpoint_metadata=metadata,
                final_full_basis_check={'dimension':len(logs),'representative_expansion_max_abs_log_difference':discrepancy,
                    'weighted_vs_full_loss_abs':abs(orbit_log_mse(representative_logs,orbit['multiplicities'],orbit['target_mean'],orbit['target_variance']).item()-selected['normalized_log_mse']),
                    'independent_sparse_operator_H_max_abs':operator_error},
                observables=obs,
                orbit_predictions={'representative_basis_index':reps.tolist(),'orbit_size':orbit['multiplicities'].long().tolist(),
                    'ed_normalized_log_amplitude':np.log(raw[reps]).tolist(),'nqs_normalized_log_amplitude':normalized[orbit['representatives']].tolist(),
                    'floored_target_normalized_log_mean':orbit['target_mean'].tolist()},
                wall_seconds_including_final_evaluation=time.perf_counter()-started)
            # Best inference CP + report staged/validated before individually atomic replaces.
            persist()
            if args.output.exists() or Path(str(args.output)+'.json').exists():raise ValueError('output appeared during fit')
            staged.replace(args.output)
            report['checkpoint_sha256']=sha(args.output)
            persist()
            Path(str(args.output)+'.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(f'checkpoint={args.output} report={report_path}',flush=True)
        return report
    except BaseException as error:
        report.update(status='failed_retained',error=f'{type(error).__name__}: {error}')
        persist();raise


if __name__=='__main__':
    main()
