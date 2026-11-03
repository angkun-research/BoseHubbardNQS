"""Actual matched PBC density/factorial/NN benchmarks; frozen inputs only.

3x3: raw ED and five predeclared primary supervised fits, exact sector sums.
2x3: existing SR+SGD500 exact saved states and stored frozen321/654 chains.
No optimizer, sampling, model changes or larger lattice reference.
"""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys,time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'benchmarks'))
from models import load_model
from exact_diagonalization import exact_ground_state
from leakage_analysis import evaluate_logs
from periodic_observables import physical_bonds,sparse_bond_operators,check_operator_H,state_observables,residuals,frozen_observables

R=ROOT/'benchmarks/results/2026-11-01-periodic-focus'
OLD=ROOT/'benchmarks/results/2026-11-01-physics-focus'
P=R/'periodic-observable-benchmarks.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_coefficients(path,expected_hash,basis):
    if sha(path)!=expected_hash:raise RuntimeError('inputcheckpoint changed before load')
    model,meta=load_model(Path(path).resolve(),'cpu')
    if sha(path)!=expected_hash:raise RuntimeError('inputcheckpoint changed during load')
    model=model.double().eval()
    logs=np.concatenate([evaluate_logs(model,basis[start:start+512]) for start in range(0,len(basis),512)])
    psi=np.exp(logs-logs.max());psi/=np.linalg.norm(psi)
    return psi,meta


def reference(Lx,Ly):
    Ns=Lx*Ly;Nmax=4
    E0,raw,basis,H=exact_ground_state(Lx,Ly,Ns,Nmax,1.,16.74,0.,True,max_dimension=20000,max_sparse_bytes=512*1024**2,pbc_x=True,pbc_y=True)
    raw=np.asarray(raw,dtype=np.float64);raw/=np.linalg.norm(raw);basis=np.asarray(basis,dtype=np.int64)
    bonds=physical_bonds(Lx,Ly,True,True);ops=sparse_bond_operators(basis,Nmax,bonds)
    error=check_operator_H(H,basis,ops,1.,16.74,0.)
    obs=state_observables(raw,basis,H,bonds,operators=ops)
    obs.update(F_N0=1.,reference_ED_energy=float(E0),eigen_residual_norm=float(np.linalg.norm(H@raw-E0*raw)))
    return raw,basis,H,bonds,ops,obs,error


def geometry(Lx,Ly,bonds):
    return {'Lx':Lx,'Ly':Ly,'Ntot':Lx*Ly,'Nmax':4,'pbc_x':True,'pbc_y':True,'t':1.,'U':16.74,'mu':0.,
            'n_sites':Lx*Ly,'bonds':bonds,'bond_counts':{a:sum(b['axis']==a for b in bonds) for a in ('x','y')},
            'physical_indices':'i=y*Lx+x; site arrays follow physicali; bonds ordered x theny.',
            'NN_definition':'G_b=Re(rho_ij+rho_ji)/2; Gx/Gy average unique physical bonds within that direction; K=-2t*(Bx*Gx+By*Gy).'}


def table_row(label,obs,statistics=None):
    return {'label':label,**{key:obs[key] for key in ('meanD','Gx','Gy','energy')},
            **{key+'_sem':statistics[key]['conditional_sem'] if statistics else None for key in ('meanD','Gx','Gy','energy')}}


def main():
    if P.exists():raise RuntimeError('refuse existing observablebenchmark')
    torch.set_num_threads(1)
    selection_path=R/'primary-checkpoint-selection.json';selection=json.loads(selection_path.read_text())
    small_path=OLD/'small-vmc-proof-results.json';small=json.loads(small_path.read_text())
    if small['status']!='completed':raise RuntimeError('existing smallVMC evidence incomplete')
    inputs={str(selection_path.relative_to(ROOT)):sha(selection_path),str(small_path.relative_to(ROOT)):sha(small_path)}
    for entry in selection['models']:
        for path in (Path(entry['checkpoint']),Path(entry['trainer_report'])):inputs[str(path.relative_to(ROOT))]=sha(path)
        assert sha(entry['checkpoint'])==entry['checkpoint_sha256']
    for model in small['models']:
        cp=Path(model['checkpoint']);assert sha(cp)==model['checkpoint_sha256'];inputs[str(cp.relative_to(ROOT))]=sha(cp)
        for seed in ('321','654'):
            report_path=Path(model['frozen'][seed]['report']);report=json.loads(report_path.read_text());raw_path=ROOT/report['raw_samples_path']
            for path in (report_path,raw_path):inputs[str(path.relative_to(ROOT))]=sha(path)
    result={'status':'running','recorded_at':datetime.now(timezone.utc).isoformat(),
            'command':[sys.executable,str(Path(__file__).resolve())],
            'source_sha256_at_start':{name:sha(ROOT/name) for name in ('benchmarks/run_periodic_benchmarks.py','benchmarks/periodic_observables.py','models.py','exact_diagonalization.py','lattice.py','observables.py','leakage_analysis.py')},
            'immutable_input_sha256':inputs,'datasets':[],
            'definitions':{'density':'<n_i>, not rho_ij','onsite_factorial':'D_i=<n_i(n_i-1)>; meanD=siteaverageD_i, NOT generally Var(n_i).',
                'rho':'rho[i,j]=<b_i^dag b_j>, exact row-ladder values/raw unsymmetrized frozen means.',
                'bond_G':'Hermitian NNoperator=(b_i^dag b_j+b_j^dag b_i)/2 on each unique unordered bond.',
                'uncertainty':'Each meanD/directionmean/kineticenergy sample is composited before timeblocking, preserving site/bond/direction covariance. ConditionalSEM=max(blockSEM,between-chainSEM), no ESS/CI/convergence guarantee.',
                'geometry':'All datasets PBC BOTHaxes; no x/yisotropy assumption on finite2x3 rectangle.',
                'scope':'All states/samples canonical fixedN, matching cutoff/bounds/physics; no training or fullFock Hamiltonians.'}}
    def persist():
        tmp=P.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(P)
    persist();started=time.perf_counter()
    try:
        raw,basis,H,bonds,ops,ed,Herror=reference(3,3)
        assert len(bonds)==18 and sum(b['axis']=='x' for b in bonds)==9
        uniform_error=float(abs(np.asarray(ed['density'])-1).max())
        if uniform_error>1e-10:raise RuntimeError('3x3PBC EDdensity not uniform to tolerance')
        data={'id':'supervised_3x3_torus','geometry':geometry(3,3,bonds),'sector_dimension':len(basis),
              'reference':ed,'reference_checks':{'independent_sparse_ladder_H_max_abs_difference':Herror,'ED_density_minus_1_max_abs':uniform_error},'models':[]}
        result['datasets'].append(data);persist()
        for entry in selection['models']:
            psi,meta=model_coefficients(entry['checkpoint'],entry['checkpoint_sha256'],basis)
            if (meta['Lx'],meta['Ly'],meta['Ntot'],meta['Nmax'],meta['pbc_x'],meta['pbc_y'])!=(3,3,9,4,True,True):raise RuntimeError('supervisedgeometry mismatch')
            obs=state_observables(psi,basis,H,bonds,operators=ops);obs['F_N0']=float(abs(raw@psi))
            row={'family':entry['family'],'checkpoint':entry['checkpoint'],'checkpoint_sha256':entry['checkpoint_sha256'],
                 'checkpoint_metadata':meta,'exact':obs,'residuals_to_ED':residuals(obs,ed),'expectation':'exact full19855state probability-weighting CPUfloat64'}
            data['models'].append(row);persist();print('supervised',entry['family'],'F',obs['F_N0'],'maxdensity',row['residuals_to_ED']['max_abs_density'],'GxGy',obs['Gx'],obs['Gy'],flush=True)
        raw,basis,H,bonds,ops,ed,Herror=reference(2,3)
        assert len(bonds)==9 and [b['axis'] for b in bonds]==['x']*3+['y']*6
        data={'id':'vmc_2x3_torus','geometry':geometry(2,3,bonds),'sector_dimension':len(basis),
              'reference':ed,'reference_checks':{'independent_sparse_ladder_H_max_abs_difference':Herror},
              'models':[],'appendix_rows':[table_row('ED',ed)]}
        result['datasets'].append(data);persist()
        for old in small['models']:
            psi,meta=model_coefficients(old['checkpoint'],old['checkpoint_sha256'],basis)
            if (meta['Lx'],meta['Ly'],meta['Ntot'],meta['Nmax'],meta['pbc_x'],meta['pbc_y'])!=(2,3,6,4,True,True):raise RuntimeError('VMCgeometry mismatch')
            obs=state_observables(psi,basis,H,bonds,operators=ops);obs['F_N0']=float(abs(raw@psi))
            if abs(obs['energy']-old['exact_final']['energy'])>1e-11:raise RuntimeError('existing exactVMC energy mismatch')
            np.testing.assert_allclose(obs['rho'],old['exact_final']['rho'],atol=1e-12,rtol=0)
            row={'family':old['family'],'label':old['label'],'checkpoint':old['checkpoint'],'checkpoint_sha256':old['checkpoint_sha256'],
                 'exact':obs,'residuals_to_ED':residuals(obs,ed),'frozen':[],'expectation':'exact saved500update state, full426state CPUfloat64'}
            data['models'].append(row);data['appendix_rows'].append(table_row('exact '+old['label'],obs));persist()
            for seed in (321,654):
                report_path=Path(old['frozen'][str(seed)]['report']);report=json.loads(report_path.read_text())
                if report['checkpoint_sha256']!=old['checkpoint_sha256'] or report['evaluation_dtype']!='float64':raise RuntimeError('frozen identity/dtype mismatch')
                raw_path=ROOT/report['raw_samples_path']
                with np.load(raw_path) as arrays:
                    summary=frozen_observables(arrays,bonds,block_length=report['settings']['block_length'])
                for key in ('density','rho'):
                    old_key='density_all_samples' if key=='density' else 'one_body_matrix_all_samples'
                    np.testing.assert_allclose(summary['observed'][key],report[old_key],atol=1e-12,rtol=0)
                frozen={'seed':seed,'report':str(report_path),'raw_npz':str(raw_path),
                        'report_sha256':sha(report_path),'raw_npz_sha256':sha(raw_path),**summary,
                        'residuals_to_exact':residuals(summary['observed'],obs),
                        'residuals_to_ED':residuals(summary['observed'],ed),
                        'sampling':report['sampling'],'time_units':report['time_units']}
                row['frozen'].append(frozen)
                data['appendix_rows'].append(table_row(old['label']+' MC'+str(seed),summary['observed'],summary['statistics']))
                print('frozen',old['label'],seed,'meanD/Gx/Gy',*[summary['observed'][k] for k in ('meanD','Gx','Gy')],flush=True);persist()
        for name,digest in inputs.items():
            if sha(ROOT/name)!=digest:raise RuntimeError('immutable source changed during export: '+name)
        for name,digest in result['source_sha256_at_start'].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('source changed during benchmark: '+name)
        result.update(status='completed',wall_seconds=time.perf_counter()-started,input_hashes_unchanged=True);persist()
    except BaseException as error:
        result.update(status='failed_retained',error=f'{type(error).__name__}: {error}');persist();raise
    print(P,flush=True)
    return result


if __name__=='__main__':main()
