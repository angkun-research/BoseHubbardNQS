"""Frozen predeclared-geometry leakage: nearby sectors, then affordablefullcutoff.

CPUfloat64 only. No model fitting, full-Fock Hamiltonian or SR guard. One
canonical sparseED reference is used separately for within-sector F/E/variance.
"""
import argparse
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import sys,time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models import load_model
from leakage_analysis import analyze_sectors,evaluate_logs,apply_number_gauge,full_overlap,projected_sector_probabilities

R=ROOT/'benchmarks/results/2026-11-01-physics-focus'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_primary(entry):
    path=Path(entry['checkpoint'])
    if sha(path)!=entry['checkpoint_sha256']:raise RuntimeError('primarycheckpoint changed before load')
    model,metadata=load_model(path.resolve(),'cpu')
    if sha(path)!=entry['checkpoint_sha256']:raise RuntimeError('primarycheckpoint changed during load')
    for key in ('arch','Lx','Ly','Ntot','Nmax','pbc_x','pbc_y','t','U','mu','architecture_settings'):
        if key in entry.get('metadata',{}) and metadata[key]!=entry['metadata'][key]:
            raise ValueError(f'checkpoint metadata changed: {key}')
    return model.double().eval(),metadata


def selection_settings(selection):
    """Honor explicit geometry/axes; historical cylinderselection remainsdefault."""
    if not selection.get('models'):
        raise ValueError('selection must contain primary models')
    first=selection['models'][0]['metadata']
    geometry=selection.get('geometry',[first['Lx'],first['Ly']])
    from lattice import lattice_shape,resolve_boundaries,_integer
    Lx,Ly=lattice_shape(*geometry)
    N0=_integer(selection.get('N0',first.get('Ntot',9)),'N0')
    Nmax=_integer(selection.get('Nmax',first.get('Nmax',4)),'Nmax')
    px,py=resolve_boundaries(True,selection.get('pbc_x',first.get('pbc_x',False)),selection.get('pbc_y',first.get('pbc_y',True)))
    expected={'Lx':Lx,'Ly':Ly,'Ntot':N0,'Nmax':Nmax,'pbc_x':px,'pbc_y':py,
              't':selection.get('t',first.get('t',1.)), 'U':selection.get('U',first.get('U',16.74)), 'mu':selection.get('mu',first.get('mu',0.))}
    if N0>Lx*Ly*Nmax:
        raise ValueError('reference sector exceeds cutoff')
    for entry in selection['models']:
        if any(entry['metadata'].get(key)!=value for key,value in expected.items()):
            raise ValueError('predeclared models must match selection geometry/axes/physics')
    return expected


@torch.no_grad()
def canonical_metrics(model,raw,basis,H,reference_energy,batch):
    logs=np.concatenate([evaluate_logs(model,basis[start:start+batch]) for start in range(0,len(basis),batch)])
    psi=np.exp(logs-logs.max());psi/=np.linalg.norm(psi)
    applied=H@psi;energy=float(psi@applied)
    return {'F_N0':float(abs(raw@psi)),'overlap_convention':'absolute unsquared amplitude overlap to normalized RAW canonical ED vector; no floor',
            'energy':energy,'energy_per_site':energy/model.n_sites,'energy_error_to_ed':energy-reference_energy,
            'variance':float(np.linalg.norm(applied-energy*psi)**2),'reference_ED_energy':reference_energy,
            'sector_dimension':len(basis),'forward_dtype':'torch.float64','forward_device':'cpu',
            'logZ_N0_crosscheck':float(np.logaddexp.reduce(2*logs))}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,default=R/'primary-checkpoint-selection.json')
    parser.add_argument('--output',type=Path,default=R/'research-focus-leakage-results.json')
    parser.add_argument('--batch-size',type=int,default=2048)
    parser.add_argument('--full-time-budget',type=float,default=600,
                        help='extrapolated total full-enumeration seconds; no SR workspace constraint')
    parser.add_argument('--skip-full',action='store_true')
    args=parser.parse_args(argv)
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'benchmarks/results') or output.exists():parser.error('output must be new under 2Dbenchmarks/results')
    output.parent.mkdir(parents=True,exist_ok=True)
    if args.batch_size<=0 or args.batch_size>16384 or not math.isfinite(args.full_time_budget) or args.full_time_budget<=0:
        parser.error('invalid batch/time budget')
    torch.set_num_threads(1)
    selection=json.loads(args.selection.read_text())
    protocol=selection_settings(selection)
    Lx,Ly,N0,Nmax=protocol['Lx'],protocol['Ly'],protocol['Ntot'],protocol['Nmax']
    n_sites=Lx*Ly
    window=list(range(max(0,N0-2),min(n_sites*Nmax,N0+2)+1))
    full_sectors=list(range(n_sites*Nmax+1))
    full_count=(Nmax+1)**n_sites
    record={'status':'running_nearby','recorded_at':datetime.now(timezone.utc).isoformat(),
            'command':[sys.executable,str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv)],
            'selection':str(args.selection.resolve()),'selection_sha256':sha(args.selection),
            'source_sha256_at_start':{name:sha(ROOT/name) for name in ('leakage_analysis.py','benchmarks/run_leakage_analysis.py','models.py','exact_diagonalization.py')},
            'settings':{'geometry':[Lx,Ly],'N0':N0,'Nmax':Nmax,'pbc_x':protocol['pbc_x'],'pbc_y':protocol['pbc_y'],
                        't':protocol['t'],'U':protocol['U'],'mu':protocol['mu'],
                        'nearby_sectors':window,'full_sectors':full_sectors,
                        'full_configuration_count':full_count,'batch_size':args.batch_size,
                        'forward_device':'cpu','forward_dtype':'torch.float64','accumulation_dtype':'float64'},
            'interpretation':[
                'Raw unprojected extension, not canonicalprojectedVMC error or wasted capacity.',
                f'Nearby probabilities are conditioned on{window}, never global. Full probabilities require all{full_count} configurations.',
                'Mean squared raw amplitude perconfiguration is logZ_N-log(d_N); absolute weights are global-log-amplitude-gauge dependent.',
                'Different number-projected states are orthogonal; these are not transition probabilities.',
                f'A fixedN{N0} number penalty is exactlyzero and cannot constrain unseen sectors.',
                'mu0 specifies the canonical Hamiltonian, not a full-Fock grandcanonicalgroundstate.'],
            'models':[]}
    def persist():
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');tmp.replace(output)
    persist();started=time.perf_counter()
    try:
        for entry in selection['models']:
            model,meta=load_primary(entry)
            row={'family':entry['family'],'checkpoint':entry['checkpoint'],'checkpoint_sha256':entry['checkpoint_sha256'],
                 'checkpoint_metadata':meta,'canonical':None,'nearby':None,'full':None,'projection':None,'gauge_demonstration':None}
            record['models'].append(row);persist()
            row['nearby']=analyze_sectors(model,N0,window,window=window,batch_size=args.batch_size)
            assert sha(entry['checkpoint'])==entry['checkpoint_sha256']
            print('nearby',entry['family'],'seconds',row['nearby']['wall_seconds'],flush=True);persist()
        projected=sum(row['nearby']['wall_seconds'] for row in record['models'])*(full_count/record['models'][0]['nearby']['enumerated_state_count'])
        record['full_affordability']={'projected_total_seconds_from_exact_nearby':projected,'budget_seconds':args.full_time_budget,
                                      'rationale':'Full vectorized base counting avoids DFS overhead, so window extrapolation is conservative; fixed-batch inference only, no H/SR arrays.'}
        # Canonical diagnostic only; NEVER evaluate H/ED across the full basis.
        record['status']=f'canonical_N{N0}_ED_diagnostics';persist()
        from exact_diagonalization import exact_ground_state
        reference_started=time.perf_counter()
        energy,raw,basis,H=exact_ground_state(Lx,Ly,N0,Nmax,protocol['t'],protocol['U'],protocol['mu'],True,
                       max_dimension=20000,max_sparse_bytes=512*1024**2,pbc_x=protocol['pbc_x'],pbc_y=protocol['pbc_y'])
        raw=np.asarray(raw,dtype=np.float64);raw/=np.linalg.norm(raw)
        basis=np.asarray(basis,dtype=np.int64)
        record['canonical_ED_reference']={'energy':float(energy),'dimension':len(basis),
            'wall_seconds':time.perf_counter()-reference_started,'scope':f'OnlyN{N0} canonical sparseED; no full-Fock/offsectorH or diagonalization, no traininglabels',
            'residual_norm':float(np.linalg.norm(H@raw-energy*raw))}
        for entry,row in zip(selection['models'],record['models']):
            model,_=load_primary(entry)
            row['canonical']=canonical_metrics(model,raw,basis,H,float(energy),args.batch_size)
            ref=next(x['logZ_N'] for x in row['nearby']['sectors'] if x['N']==N0)
            if abs(ref-row['canonical']['logZ_N0_crosscheck'])>1e-9:raise RuntimeError('canonical and window logZ9 disagree')
            row['projection']={'diagnostic':'Exact delta projection, computed algebraically from same canonicalstate, no model/weight mutation or retraining',
                'P_full_N0_after':1.,'leakage_after':0.,
                'F_N0_before':row['canonical']['F_N0'],'F_N0_after':row['canonical']['F_N0'],'F_full_after':row['canonical']['F_N0'],
                'canonical_energy_before':row['canonical']['energy'],'canonical_energy_after':row['canonical']['energy'],
                'canonical_variance_unchanged':row['canonical']['variance'],
                'sector_probabilities':projected_sector_probabilities(full_sectors,N0)}
            print('canonical',entry['family'],'F_N0',row['canonical']['F_N0'],flush=True);persist()
        run_full=not args.skip_full and projected<=args.full_time_budget
        record['full_status']='running' if run_full else ('skipped_explicitly' if args.skip_full else 'skipped_projected_runtime_budget')
        record['status']='running_full' if run_full else 'finishing_window_only';persist()
        for entry,row in zip(selection['models'],record['models']):
            if run_full:
                model,_=load_primary(entry)
                row['full']=analyze_sectors(model,N0,window=window,batch_size=args.batch_size)
                for nearby in row['nearby']['sectors']:
                    full=next(x for x in row['full']['sectors'] if x['N']==nearby['N'])
                    if abs(full['logZ_N']-nearby['logZ_N'])>1e-9:raise RuntimeError('full/nearby exact logZ differs')
                row['full']['F_full_zero_embedded_ED']=full_overlap(row['canonical']['F_N0'],row['full']['P_full_N0'])
                row['full']['log_F_full_zero_embedded_ED']=math.log(row['canonical']['F_N0'])+.5*row['full']['logP_full_N0']
                print('full',entry['family'],'seconds',row['full']['wall_seconds'],'P9',row['full']['P_full_N0'],'Ffull',row['full']['F_full_zero_embedded_ED'],flush=True)
            base=row['full'] if row['full'] is not None else row['nearby']
            cases=[]
            for a in (-4.,-2.,0.,2.,4.):
                shifted=apply_number_gauge(base,a)
                if row['full'] is not None:
                    shifted['F_full_zero_embedded_ED']=full_overlap(row['canonical']['F_N0'],shifted['P_full_N0'])
                cases.append(shifted)
            row['gauge_demonstration']={'formula':f'f_a(n)=f(n)+a*(N-{N0}); logZ_N(a)=logZ_N+2*a*(N-{N0})',
                'canonical_F_and_E_identical_for_all_a':True,'canonical_F_N0':row['canonical']['F_N0'],
                'canonical_energy':row['canonical']['energy'],'number_penalty_on_fixedN9_samples':0.,
                'cases':cases,'interpretation':f'Each offsector ratio can change arbitrarily while canonicalstate is identical. A single linear tilt does not independently set every sector or guarantee arbitraryP_N0; general h(N) with h({N0})=0 is also unconstrained.'}
            if sha(entry['checkpoint'])!=entry['checkpoint_sha256']:raise RuntimeError('immutablecheckpoint changed during analysis')
            row['checkpoint_hash_unchanged']=True;persist()
        record['status']='completed';record['full_status']=f'completed_all_{len(full_sectors)}_sectors' if run_full else record['full_status']
        record['wall_seconds_total']=time.perf_counter()-started;persist()
    except BaseException as error:
        record.update(status='failed_retained',error=f'{type(error).__name__}: {error}');persist();raise
    print(output,flush=True)
    return record


if __name__=='__main__':main()
