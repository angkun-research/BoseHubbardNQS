"""Read-only M2048 RBM ridge/spectral directions, prior to paired training."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models import load_model
from vmc import compute_log_psi_jacobian, local_energy, sr_gradient


def main():
    torch.set_num_threads(1)
    R=ROOT/'benchmarks/results/2026-10-31-all-model-refinement'
    checkpoint=ROOT/'outputs/2026-10-31-all-model-refinement/RBM64-4x4-lower-ridge500-seed6802.pt'
    frozen_report=ROOT/'outputs/2026-10-31-all-model-refinement/frozen/RBM64-4x4-lower-ridge500-seed321.json'
    raw=ROOT/json.loads(frozen_report.read_text())['raw_samples_path']
    output=ROOT/'benchmarks/results/minsr-frozen-M2048-calibration.json'
    if output.exists():raise RuntimeError('refuse existing calibration')
    report={'status':'running','recorded_at':datetime.now(timezone.utc).isoformat(),
            'command':[sys.executable,str(Path(__file__).resolve())],
            'source_sha256_at_start':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('vmc.py','models.py','train.py','benchmarks/check_minsr_frozen.py')},
            'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'frozen_report':str(frozen_report),'raw_samples':str(raw),'raw_samples_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
            'M':2048,'max_sr_bytes':4*1024**3,'cases':[],
            'scope':'Read-only directions, no updates. Seed246 samples2048 without replacement from retained321 frozen trajectory; correlated MC batch, not independent exact expectations. CPU float32 and float64 model derivatives; all SR algebra float64.'}
    def persist():output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    persist()
    try:
        with np.load(raw) as z:
            states=z['occupations'].reshape(-1,16)
            index=np.random.default_rng(246).choice(len(states),2048,replace=False)
            states=torch.tensor(states[index],dtype=torch.long)
        report['retained_flat_sample_indices']=index.tolist()
        for dtype in ('float64','float32'):
            model,meta=load_model(checkpoint,'cpu');model=model.to(dtype=getattr(torch,dtype))
            started=time.perf_counter()
            J=compute_log_psi_jacobian(model,states,jacobian_method='vmap',jacobian_batch_size=128,max_sr_bytes=4*1024**3,regularizer='minsr')
            E=local_energy(model,states,meta['t'],meta['U'],meta['mu'],256)
            compute_seconds=time.perf_counter()-started
            ridge=sr_gradient(J,E,tau=5e-6,max_sr_bytes=4*1024**3)
            for rpinv in (1e-12,1e-8,1e-6):
                t0=time.perf_counter()
                direction,info=sr_gradient(J,E,regularizer='minsr',rpinv=rpinv,max_sr_bytes=4*1024**3,return_diagnostics=True)
                centered=J-J.mean(0)
                force=centered.T@(E.double()-E.double().mean())/len(J)
                report['cases'].append({'model_dtype':dtype,'rpinv':rpinv,'direction':info,
                    'direction_seconds':time.perf_counter()-t0,'J_and_local_energy_seconds':compute_seconds,
                    'ridge_tau':5e-6,'ridge_direction_norm':ridge.norm().item(),
                    'direction_cosine_to_ridge':torch.nn.functional.cosine_similarity(direction,ridge,dim=0).item(),
                    'parameter_step_norm_at_eta_1e_3':.001*direction.norm().item(),
                    'linearized_log_amplitude_change_rms_at_eta_1e_3':(.001*(centered@direction)).square().mean().sqrt().item(),
                    'predicted_first_order_energy_change_at_eta_1e_3':-.002*torch.dot(force,direction).item(),
                    'sampled_local_energy_mean':E.double().mean().item()})
                print(dtype,rpinv,'norm',direction.norm().item(),'cosine',report['cases'][-1]['direction_cosine_to_ridge'],flush=True)
                persist()
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==report['checkpoint_sha256']
        report['status']='completed_readonly';persist()
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');persist();raise


if __name__=='__main__':main()
