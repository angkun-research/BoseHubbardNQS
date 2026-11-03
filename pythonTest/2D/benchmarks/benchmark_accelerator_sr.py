#!/usr/bin/env python3
"""Bounded local-energy/Jacobian/SR+SGD timing; no sampler or training loop.

Same fixed-N feasible states and frozen weights for each backend/chunk. CUDA
runs only on an idle GPU without other compute processes. One discarded small
warmup and one full update per case; CPU float32 is a numerical reference only.
"""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import build_model, load_model, parameter_count
from vmc import check_sr_budget, compute_log_psi_jacobian, generate_initial_states, local_energy, sr_gradient


def gpu_snapshot():
    def query(kind, fields):
        result = subprocess.run(['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=10, check=True)
        return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))
    try:
        return {'gpus': query('gpu', 'index,uuid,name,memory.used,memory.total,utilization.gpu'),
                'processes': query('compute-apps', 'gpu_uuid,pid,process_name,used_memory')}
    except (OSError, subprocess.SubprocessError) as error:
        return {'error': str(error), 'gpus': [], 'processes': []}


def external_processes(snapshot, uuid):
    return [p for p in snapshot['processes'] if p[0] == uuid and p[1] != str(os.getpid())]


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def components(model, states, chunk):
    energies = local_energy(model, states, 1., 16.74, 0., 256)
    jacobian = compute_log_psi_jacobian(model, states, jacobian_batch_size=chunk)
    direction = sr_gradient(jacobian, energies, .01)
    return energies, jacobian, direction


def timed_update(frozen, states_cpu, device, dtype, chunk):
    model = copy.deepcopy(frozen).to(device=device, dtype=dtype).eval()
    states = states_cpu.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    check_sr_budget(model, len(states), jacobian_batch_size=chunk)
    # Small warmup is discarded; timing starts from the exact frozen weights.
    components(model, states[:8], chunk)
    synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    timings = {}
    started_all = time.perf_counter()
    start = time.perf_counter()
    E = local_energy(model, states, 1., 16.74, 0., 256)
    synchronize(device)
    timings['local_energy_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    J = compute_log_psi_jacobian(model, states, jacobian_batch_size=chunk)
    synchronize(device)
    timings['jacobian_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    delta = sr_gradient(J, E, .01)
    synchronize(device)
    timings['sr_direction_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            p.grad = delta[offset:offset+p.numel()].reshape_as(p).to(p.dtype).clone()
            offset += p.numel()
    optimizer.step()
    synchronize(device)
    timings['sgd_update_seconds'] = time.perf_counter() - start
    timings['complete_seconds'] = time.perf_counter() - started_all
    timings['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    timings['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None
    timings['cuda_peak_scope'] = ('allocator peak reset after model/input allocation and small warmup; includes resident model/input, '
                                  'local energies, J, centered SR solve/workspace, SGD gradients; excludes setup and CPU reference') if device.type == 'cuda' else None
    with torch.no_grad():
        before = torch.cat([p.detach().flatten() for p in frozen.parameters()]).to(device=device, dtype=dtype)
        after = torch.cat([p.detach().flatten() for p in model.parameters()])
        expected = before - .01 * delta.to(dtype)
        update_error = (after - expected).abs().max().item()
        assert torch.isfinite(after).all()
        torch.testing.assert_close(after, expected, rtol=2e-6, atol=2e-7)
    row = {'backend': str(device), 'dtype': str(dtype), 'chunk': chunk, **timings,
           'sr_dtype': str(delta.dtype), 'energy_mean': E.double().mean().item(),
           'direction_norm': delta.norm().item(), 'update_max_abs_error': update_error,
           'budget': check_sr_budget(model, len(states), jacobian_batch_size=chunk)}
    return row, (E.detach().cpu().double(), J.detach().cpu(), delta.detach().cpu())


def differences(actual, reference):
    difference = actual.double() - reference.double()
    return {'max_abs': difference.abs().max().item(),
            'relative_l2': (difference.norm() / reference.double().norm().clamp_min(1e-30)).item()}


def logs(model, states):
    with torch.no_grad():
        p = next(model.parameters())
        return model(F.one_hot(states.to(p.device), model.Nmax+1).to(p.dtype)).detach().cpu().double()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/accelerator-sr-workload.json'))
    args = parser.parse_args(argv)
    output = (ROOT/'benchmarks'/args.output).resolve()
    if not output.is_relative_to(ROOT/'benchmarks/results') or output.exists():
        parser.error('output must be new and under benchmarks/results/')
    torch.set_num_threads(1)
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    snapshot = gpu_snapshot()
    idle = [g for g in snapshot['gpus'] if not external_processes(snapshot,g[1]) and int(g[5]) <= 10 and int(g[3]) < 1024]
    selected = min(idle, key=lambda g: int(g[3])) if idle and torch.cuda.is_available() else None
    report = {'status': 'running', 'recorded_at': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
              'source_sha256_at_start': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                                         ('models.py','vmc.py','lattice.py','train.py','benchmarks/benchmark_accelerator_sr.py')},
              'environment': {'torch': str(torch.__version__), 'python':sys.version, 'platform':platform.platform(),
                              'threads':1, 'TF32':False}, 'gpu_snapshot_at_start': snapshot,
              'selected_gpu': selected, 'settings': {'Lx':4,'Ly':4,'pbc_x':False,'pbc_y':True,'Ntot':16,'Nmax':4,
                                                    't':1.,'U':16.74,'mu':0.,'M':512,'chunks':[1,8,16],
                                                    'lr':.01,'tau':.01,'timed_updates_per_case':1,'warmup_samples':8,
                                                    'states':'same fixed-N feasible initializer states, not equilibrium samples'},
              'tolerances': {'cuda_vs_cpu_float32_jacobian_relative_l2':5e-5,
                             'cuda_vs_cpu_float32_direction_relative_l2':5e-3,
                             'cuda_vs_cpu_float32_local_energy_relative_l2':2e-4},
              'models': [], 'limitations':'Shared workstation single timings; no sampling or optimizer trajectory/campaign. '
                            'Solve timing includes centering, Gram construction and float64 solve. Memory budget excludes activations. '
                            'Forward rounding comparisons concern representative finite inputs, not energy accuracy or convergence.'}
    def persist():
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    output.parent.mkdir(parents=True,exist_ok=True)
    persist()
    states = generate_initial_states(16,16,4,512,'cpu')
    for arch,width in [('Conv2DNet',32),('TransformerNet',64)]:
        source = ROOT/'outputs/2026-10-31-cylinder-pilots/Conv2DNet-4x4-seed123.pt'
        if arch == 'Conv2DNet':
            checksum=hashlib.sha256(source.read_bytes()).hexdigest()
            frozen, metadata = load_model(source,device='cpu')
            assert hashlib.sha256(source.read_bytes()).hexdigest()==checksum
            assert (metadata['arch'],metadata['Lx'],metadata['Ly'],metadata['pbc_x'],metadata['pbc_y'],metadata['Ntot'],metadata['Nmax'],metadata['t'],metadata['U'],metadata['mu']) == (arch,4,4,False,True,16,4,1.,16.74,0.)
            assert metadata['architecture_settings']=={'hidden_dim':32}
            origin={'checkpoint':str(source.relative_to(ROOT)),'sha256_at_load':checksum}
            frozen=frozen.double().eval()
        else:
            # Native float64 initialization makes weight rounding measurable.
            torch.manual_seed(123)
            prior=torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float64)
                frozen=build_model(arch,4,4,Nmax=4,pbc_x=False,pbc_y=True,hidden_dim=width).eval()
            finally:
                torch.set_default_dtype(prior)
            origin={'initialization':'fresh native float64 seed123; qualified width64; no width32 weights loaded'}
        rounded=copy.deepcopy(frozen).float().double()
        reference32=copy.deepcopy(frozen).float().eval()
        representative=states[:32]
        rounding={'original64_vs_float32_weights_recast64': differences(logs(rounded,representative),logs(frozen,representative)),
                  'cpu_float32_forward_vs_rounded_weights_cpu64': differences(logs(reference32,representative),logs(rounded,representative)),
                  'representative_count':len(representative)}
        E32,J32,d32=components(reference32,states,8)
        reference=(E32.double(),J32,d32)
        model_record={'arch':arch,'width':width,'parameters':parameter_count(frozen),'origin':origin,
                      'forward_rounding':rounding,'cpu_float32_reference_chunk':8,'cases':[]}
        report['models'].append(model_record)
        for chunk in (1,8,16):
            row,_=timed_update(frozen,states,torch.device('cpu'),torch.float64,chunk)
            model_record['cases'].append(row); persist()
            print(arch,'cpu float64',chunk,row['complete_seconds'],flush=True)
        if selected is not None:
            device=torch.device('cuda:'+selected[0])
            for chunk in (1,8,16):
                now=gpu_snapshot()
                if 'error' in now or external_processes(now,selected[1]):
                    model_record['cases'].append({'backend':str(device),'chunk':chunk,'status':'skipped_busy_or_unavailable','gpu_snapshot':now}); persist(); continue
                row,actual=timed_update(frozen,states,device,torch.float32,chunk)
                checks={label:differences(a,b) for label,a,b in zip(('local_energy','jacobian','direction'),actual,reference)}
                row['vs_cpu_float32']=checks
                row['precision_checks_passed']=all(checks[k]['relative_l2']<=limit for k,limit in
                                                  [('local_energy',2e-4),('jacobian',5e-5),('direction',5e-3)])
                model_record['cases'].append(row); persist()
                if not row['precision_checks_passed']:
                    raise RuntimeError('float32 cross-device discrepancy exceeded documented tolerance')
                print(arch,'cuda float32',chunk,row['complete_seconds'],checks,flush=True)
                del actual
            gpu_model=copy.deepcopy(frozen).to(device=device,dtype=torch.float32).eval()
            rounding['cuda_float32_forward_vs_rounded_weights_cpu64']=differences(logs(gpu_model,representative),logs(rounded,representative))
            del gpu_model
        else:
            model_record['cuda_status']='skipped: no idle accessible GPU'
        del E32,J32,d32,reference,reference32,rounded,frozen
        persist()
    report['core_hashes_unchanged_at_end']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in report['source_sha256_at_start'].items()}
    assert all(report['core_hashes_unchanged_at_end'].values())
    report['status']='completed'
    persist()
    print(output)
    return report


if __name__=='__main__':
    main()
