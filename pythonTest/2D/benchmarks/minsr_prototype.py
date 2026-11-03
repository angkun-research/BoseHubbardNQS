"""CPU-only real MinSR spectral-filter prototype and bounded calibration.

Chen/Heyl, Nature Physics20 (2024), Methods20-23. No core imports from this
prototype, no persistent optimizer updates and no target-size training.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import load_model
from train_ed import make_ed_dataset
from vmc import compute_log_psi_jacobian


@torch.no_grad()
def soft_spectral_filter(eigenvalues, rpinv=1e-12, apinv=0.0):
    """PSD soft pseudoinverse; stable below cutoff and exactly zero at zero."""
    if eigenvalues.dtype != torch.float64 or eigenvalues.ndim != 1 or not len(eigenvalues):
        raise ValueError('eigenvalues must be nonempty float64 vector')
    if not torch.isfinite(eigenvalues).all():
        raise ValueError('eigenvalues must be finite')
    if any(isinstance(x, bool) or not math.isfinite(x) or x < 0 for x in (rpinv, apinv)):
        raise ValueError('rpinv and apinv must be finite nonnegative scalars')
    scale = eigenvalues.abs().max().item()
    tolerance = 64 * torch.finfo(torch.float64).eps * len(eigenvalues) * scale
    if eigenvalues.min().item() < -tolerance:
        raise RuntimeError('materially negative PSD Gram eigenvalue')
    cutoff = rpinv * scale + apinv
    if not math.isfinite(cutoff):
        raise ValueError('spectral cutoff overflow')
    weights = torch.zeros_like(eigenvalues)
    positive = eigenvalues > 0
    above = positive & (eigenvalues >= cutoff)
    # Ratios are bounded by one in both branches. Never form (c/lambda)**6
    # when lambda<c, never multiply an infinite reciprocal by a zero weight.
    if cutoff == 0:
        weights[positive] = 1 / eigenvalues[positive]
    else:
        large = eigenvalues[above]
        weights[above] = (1 / large) / (1 + (cutoff / large)**6)
        below = positive & ~above
        ratio = eigenvalues[below] / cutoff
        weights[below] = (ratio**5 / cutoff) / (1 + ratio**6)
    if not torch.isfinite(weights).all():
        raise RuntimeError('non-finite soft pseudoinverse weights')
    diagnostics = {
        'regularizer': 'minsr', 'soft_cutoff_power': 6, 'rpinv': rpinv, 'apinv': apinv,
        'spectral_cutoff': cutoff, 'max_abs_eigenvalue': scale,
        'min_eigenvalue': eigenvalues.min().item(), 'max_eigenvalue': eigenvalues.max().item(),
        'positive_eigenvalues': positive.sum().item(),
        'zero_eigenvalues': (eigenvalues == 0).sum().item(),
        'roundoff_negative_eigenvalues': (eigenvalues < 0).sum().item(),
        'negative_roundoff_tolerance': tolerance,
        'positive_modes_at_or_above_cutoff': above.sum().item(),
        'soft_effective_modes': (weights * eigenvalues.clamp_min(0)).sum().item(),
        'maximum_spectral_gain': weights.max().item(),
        'zero_negative_policy': 'exact zero and roundoff-negative PSD eigenvalues map0; materially negative spectrum raises; no extra positive-eigenvalue truncation',
    }
    return weights, diagnostics


@torch.no_grad()
def minsr_direction(J, energies, *, solver='auto', rpinv=1e-12, apinv=0.0,
                    max_bytes=512*1024**2):
    if J.ndim != 2 or not min(J.shape) or J.is_complex():
        raise ValueError('nonempty real Jacobian required')
    M, P = J.shape
    selected = ('parameter' if P <= M else 'sample') if solver == 'auto' else solver
    if selected not in ('sample', 'parameter'):
        raise ValueError('invalid solver')
    D = M if selected == 'sample' else P
    estimate = 8 * (3*M*P + 16*D*D + 8*P + 8*M)
    if estimate > max_bytes:
        raise ValueError('prototype Jacobian/eigh allocation budget exceeded')
    J = J.double()
    E = torch.as_tensor(energies, device=J.device, dtype=torch.float64).reshape(-1)
    if len(E) != M or not torch.isfinite(J).all() or not torch.isfinite(E).all():
        raise ValueError('finite matching energies/Jacobian required')
    A = (J-J.mean(0)) / math.sqrt(M)
    e = (E-E.mean()) / math.sqrt(M)
    gram = A @ A.T if selected == 'sample' else A.T @ A
    if not torch.isfinite(gram).all():
        raise RuntimeError('non-finite Gram')
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    weights, diagnostics = soft_spectral_filter(eigenvalues, rpinv, apinv)
    rhs = e if selected == 'sample' else A.T @ e
    solution = eigenvectors @ (weights * (eigenvectors.T @ rhs))
    direction = A.T @ solution if selected == 'sample' else solution
    if not torch.isfinite(direction).all():
        raise RuntimeError('non-finite MinSR direction')
    return direction, {**diagnostics, 'solver': selected, 'solve_dimension': D,
                       'estimated_bytes': estimate, 'tau': None,
                       'direction_norm': direction.norm().item()}


def self_checks():
    torch.manual_seed(8)
    records = []
    for M, P in ((5, 9), (11, 4), (7, 7)):
        J = torch.randn(M, P, dtype=torch.float64)
        J[:, -1] = 2  # unused/gauge direction, not full rank
        E = torch.randn(M, dtype=torch.float64)
        A, e = (J-J.mean(0))/math.sqrt(M), (E-E.mean())/math.sqrt(M)
        U, singular, Vh = torch.linalg.svd(A, full_matrices=False)
        for r in (1e-12, 1e-8, 1e-6):
            lam = singular.square()
            c = r * lam.max()
            # Independent SVD direction: sigma*g(sigma**2) in singular basis.
            gain = torch.zeros_like(singular)
            active = singular > 1e-14
            gain[active] = 1/singular[active] / (1+(c/lam[active])**6)
            expected = Vh.T @ (gain * (U.T @ e))
            values = []
            for space in ('sample', 'parameter', 'auto'):
                actual, diagnostics = minsr_direction(J, E, solver=space, rpinv=r)
                torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
                shifted, _ = minsr_direction(J+7, E+3, solver=space, rpinv=r)
                torch.testing.assert_close(actual, shifted, atol=2e-11, rtol=2e-10)
                values.append((actual-expected).abs().max().item())
            records.append({'M':M,'P':P,'rpinv':r,'max_error_vs_independent_SVD':max(values)})
    zero, info = minsr_direction(torch.ones(5, 8), torch.arange(5.))
    assert torch.count_nonzero(zero) == 0 and info['spectral_cutoff'] == 0
    vals = torch.tensor([-1e-16, 0, 1e-300, 1e-12, 1.], dtype=torch.float64)
    weights, info = soft_spectral_filter(vals, 1e-12, 0)
    assert torch.isfinite(weights).all() and weights[0] == weights[1] == 0
    assert weights[2] == 0 and abs(weights[3].item()-5e11) < 1e-3
    try: soft_spectral_filter(torch.tensor([-1.,2.],dtype=torch.float64))
    except RuntimeError: pass
    else: raise AssertionError('materially negative spectrum was accepted')
    try: minsr_direction(torch.ones(5,8), torch.arange(5.), max_bytes=1)
    except ValueError: pass
    else: raise AssertionError('allocation guard missing')
    return records


@torch.no_grad()
def exact_metrics(model, dataset, batch=256):
    logs = torch.cat([model(x) for x in dataset['x'].split(batch)]).double()
    psi = (logs - .5*torch.logsumexp(2*logs,0)).exp().numpy()
    Hpsi = dataset['H'] @ psi
    energy = float(psi @ Hpsi)
    variance = float(np.linalg.norm(Hpsi-energy*psi)**2)
    return {'exact_energy':energy,'exact_variance':variance,
            'absolute_amplitude_overlap':float(abs(psi @ dataset['raw_ed_state'])),
            'exact_energy_error_per_site':float((energy-dataset['metadata']['raw_ed_reference']['energy'])/9)}, psi


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/minsr-prototype-calibration.json')
    parser.add_argument('--self-tests-only',action='store_true')
    args=parser.parse_args(argv)
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'benchmarks/results') or output.exists():
        parser.error('output must be new and under benchmarks/results')
    torch.set_num_threads(1)
    from vmc import sr_gradient
    from torch.nn import functional as F
    r=ROOT/'benchmarks/results/2026-10-31-all-model-refinement'
    report={'status':'running','recorded_at':datetime.now(timezone.utc).isoformat(),
            'command':[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],
            'source_sha256_at_start':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('models.py','vmc.py','train.py','train_ed.py','benchmarks/minsr_prototype.py')},
            'environment':{'torch':str(torch.__version__),'threads':1,'device':'cpu','dtype':'float64'},
            'self_checks':[],'frozen_batches':[],'tiny_ED_steps':[],
            'scope':'Real positive-amplitude spectral MinSR component only. Read-only frozen batches and ephemeral cloned one-step3x3 checks; no larger training, no universal convergence/architecture claims.'}
    def persist():output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    persist()
    try:
        report['self_checks']=self_checks();persist()
        if not args.self_tests_only:
            for label,filename in (('FC6','FC64d3-6x4-lower-ridge500-seed5791.pt'),('RBM4','RBM64-4x4-lower-ridge500-seed6802.pt')):
                path=ROOT/'outputs/2026-10-31-all-model-refinement'/filename
                model,meta=load_model(path.resolve(),'cpu');model=model.double()
                frozen_report=ROOT/'outputs/2026-10-31-all-model-refinement/frozen'/f'{Path(filename).stem.rsplit("-seed",1)[0]}-seed321.json'
                frozen=ROOT/json.loads(frozen_report.read_text())['raw_samples_path']
                with np.load(frozen) as z:
                    print('NPZ arrays',frozen.name,z.files,flush=True)
                    states=torch.tensor(z['occupations'].reshape(-1,meta['Lx']*meta['Ly'])[:128],dtype=torch.long)
                started=time.perf_counter();J=compute_log_psi_jacobian(model,states,jacobian_method='vmap',jacobian_batch_size=128)
                from vmc import local_energy
                E=local_energy(model,states,meta['t'],meta['U'],meta['mu'],128)
                base=sr_gradient(J,E,tau=1e-5)
                for cutoff in (1e-12,1e-8,1e-6):
                    t0=time.perf_counter();d,info=minsr_direction(J,E,rpinv=cutoff)
                    report['frozen_batches'].append({'label':label,'checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'NPZ':str(frozen),'M':len(states),'rpinv':cutoff,'local_energy_mean_MC':E.mean().item(),
                        'ridge_direction_norm':base.norm().item(),'direction_cosine_to_ridge':torch.nn.functional.cosine_similarity(d,base,dim=0).item(),
                        'direction_diagnostics':info,'spectral_seconds':time.perf_counter()-t0,'elapsed_J_localE_and_solutions_seconds':time.perf_counter()-started})
                    persist()
            data=make_ed_dataset(3,3,Nmax=4,Ntot=9,pbc_x=False,pbc_y=True,max_dimension=20000)
            # No dense Hamiltonian conversion. Guarded sparse ED dimension19855.
            for label,path in (('RBM64',ROOT/'outputs/2026-10-31-all-model-refinement/qualification/RBM64-3x3-prepared.pt'),
                               ('Conv32',ROOT/'outputs/2026-10-31-3x3-cylinder-supervised/Conv2DNet-seed123.pt')):
                model,meta=load_model(path.resolve(),'cpu');model=model.double()
                initial,psi=exact_metrics(model,data)
                assert initial['absolute_amplitude_overlap']>=.99
                rng=np.random.default_rng(246)
                index=rng.choice(len(psi),size=128,replace=True,p=psi**2)
                states=torch.as_tensor(np.asarray(data['basis'])[index],dtype=torch.long)
                J=compute_log_psi_jacobian(model,states,jacobian_method='vmap',jacobian_batch_size=128)
                Hpsi=data['H']@psi
                E=torch.from_numpy(Hpsi[index]/psi[index])
                cases=[('ridge',None)] + [('minsr',c) for c in (1e-12,1e-8,1e-6)]
                for method,cutoff in cases:
                    start=time.perf_counter()
                    d,info=minsr_direction(J,E,rpinv=cutoff) if method=='minsr' else (sr_gradient(J,E,tau=1e-3),{'tau':1e-3})
                    seconds=time.perf_counter()-start
                    for eta in (1e-4,1e-3):
                        updated=copy.deepcopy(model)
                        opt=torch.optim.SGD(updated.parameters(),lr=eta)
                        offset=0
                        for p in updated.parameters():
                            p.grad=d[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
                        opt.step()
                        after,_=exact_metrics(updated,data)
                        report['tiny_ED_steps'].append({'model':label,'checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'method':method,'rpinv':cutoff,'eta':eta,'M':128,'batch_expectation':'iid draws from exact normalized CURRENT model amplitudes, seed246; MC force/metric, not exact fullsector force',
                            'initial':initial,'after':after,'exact_energy_change':after['exact_energy']-initial['exact_energy'],
                            'parameter_step_norm':eta*d.norm().item(),'direction_norm':d.norm().item(),'direction_diagnostics':info,'direction_seconds':seconds})
                        print(label,method,cutoff,eta,'Echange',after['exact_energy']-initial['exact_energy'],'var',after['exact_variance'],flush=True)
                        persist()
        report['status']='completed';persist()
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');persist();raise
    print(output,flush=True)


if __name__=='__main__':
    main()
