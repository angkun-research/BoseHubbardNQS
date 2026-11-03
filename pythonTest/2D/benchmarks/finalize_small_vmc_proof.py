"""Add raw ED density/rho reference and plot labels to completed smallVMC data.

Diagnostic only, no optimizer/model updates. Preserve all stochastic outcomes.
"""
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from exact_diagonalization import exact_ground_state
from lattice import bonds_2d

R=ROOT/'benchmarks/results/2026-11-01-physics-focus'


def reference_observables(raw,basis,H):
    p=raw**2;density=p@basis;moment=p@(basis*(basis-1));rho=np.zeros((6,6))
    lookup={tuple(row):i for i,row in enumerate(basis)}
    for i in range(6):
        rho[i,i]=density[i]
        for j in range(6):
            if i==j:continue
            for k,n in enumerate(basis):
                if n[i]==0 or n[j]==4:continue
                other=n.copy();other[i]-=1;other[j]+=1
                rho[i,j]+=raw[k]*raw[lookup[tuple(other)]]*math.sqrt(float(n[i]*(n[j]+1)))
    energy=float(raw@(H@raw))
    reconstructed=-sum(rho[i,j]+rho[j,i] for i,j in bonds_2d(2,3))+16.74/2*moment.sum()
    assert abs(reconstructed-energy)<1e-11
    assert abs(rho-rho.T).max()<1e-13 and abs(np.trace(rho)-6)<1e-12
    return {'energy':energy,'variance':float(np.linalg.norm(H@raw-energy*raw)**2),
            'density':density.tolist(),'onsite_factorial_moment':moment.tolist(),'rho':rho.tolist(),
            'energy_from_observables':float(reconstructed),'rho_trace':float(np.trace(rho)),
            'rho_raw_hermiticity_max_abs':float(abs(rho-rho.T).max()),
            'physical_indices':'i=y*2+x; rho[i,j]=<b_i^dag b_j>, no symmetrization'}


def main():
    torch.set_num_threads(1)
    path=R/'small-vmc-proof-results.json';results=json.loads(path.read_text())
    if results['status'] not in ('completed','completed_with_failures_retained'):
        raise RuntimeError('training/frozen runner must finish before postprocessing')
    if 'diagnostic_postprocessing' in results:raise RuntimeError('refuse repeat postprocessing')
    E0,raw,basis,H=exact_ground_state(2,3,6,4,1.,16.74,0.,True)
    raw=np.asarray(raw,dtype=np.float64);raw/=np.linalg.norm(raw);basis=np.asarray(basis,dtype=np.int64)
    ed=reference_observables(raw,basis,H)
    assert abs(results['ED_reference']['energy']-ed['energy'])<1e-11
    results['ED_reference'].update(ed)
    for model in results['models']:
        for key in ('exact_initial','exact_final'):
            if model.get(key) is None:continue
            q=model[key]
            q['comparison_to_ED']={
                'density_difference':(np.asarray(q['density'])-ed['density']).tolist(),
                'density_max_abs_difference':float(abs(np.asarray(q['density'])-ed['density']).max()),
                'rho_difference':(np.asarray(q['rho'])-ed['rho']).tolist(),
                'rho_max_abs_difference':float(abs(np.asarray(q['rho'])-ed['rho']).max())}
        model['exact_points']=[{'completed_updates':0,'metrics':model.get('exact_initial')},
                               {'completed_updates':500,'metrics':model.get('exact_final')}]
        for point in model['trajectory_MC']:
            point['completed_updates']=point['epoch']-1
            point['expectation']='MonteCarlo beforeupdate, not exact; local variance is not a mean-energy errorbar'
        for seed,info in model['frozen'].items():
            if info['status']!='completed':continue
            frozen=json.loads(Path(info['report']).read_text())
            info['comparison_to_ED']={
                'MC_energy_minus_ED':float(frozen['statistics']['energy']['mean']-ed['energy']),
                'density_difference':(np.asarray(frozen['density_all_samples'])-ed['density']).tolist(),
                'rho_difference':(np.asarray(frozen['one_body_matrix_all_samples'])-ed['rho']).tolist(),
                'meaning':'Includes both learned-state error and sampling deviation. comparison_to_exact isolates sampling deviation relative to exact finalNQS expectation.'}
            info['comparison_to_exact']['rho_pair_order']=frozen['one_body_pairs']
    results['diagnostic_postprocessing']={'recorded_at':datetime.now(timezone.utc).isoformat(),
        'script':'benchmarks/finalize_small_vmc_proof.py',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Separate canonical426state ED diagnostics only, no data/labels to optimizer. Original training source hash/MC outcomes retained.'}
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    print(path)


if __name__=='__main__':main()
