#!/usr/bin/env python3
"""Standalone guarded torch.func per-example gradient prototype; no core edits.

Check all five models and unused parameters, then time M512 Jacobians only:
current chunk16 batched VJP versus vmap(grad) chunks16/64/128. No updates,
local energies, solve or sampler. CPU double and idle-GPU float32 are separate
workloads, not a precision-neutral hardware comparison.
"""

import argparse
from contextlib import nullcontext
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import ARCH_BUILDERS, build_model, parameter_count
from vmc import DEFAULT_MAX_SR_BYTES, _states, _log_amplitudes, check_sr_budget, compute_log_psi_jacobian, generate_initial_states
from benchmark_accelerator_sr import gpu_snapshot, external_processes, synchronize, differences


def prototype_budget(model, M, chunk, max_bytes=DEFAULT_MAX_SR_BYTES):
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk not in (16,64,128):
        raise ValueError('prototype chunk must be16,64,128')
    base=check_sr_budget(model,M,max_sr_bytes=max_bytes)
    B=min(chunk,M)
    element=next(model.parameters()).element_size()
    # Existing SR guard plus returned per-parameter gradient and flattened copy,
    # and one-hot block. Activations are bounded in batch size, not hard bytes.
    extra=2*B*parameter_count(model)*element+B*model.n_sites*model.channels*element
    estimate=base['estimated_bytes']+extra
    if estimate>max_bytes:
        raise ValueError(f'prototype estimated workspace {estimate} exceeds max_bytes={max_bytes}')
    return {**base,'vmap_chunk':chunk,'vmap_rows':B,'extra_gradient_and_input_bytes':extra,
            'estimated_bytes':estimate,'excludes':'model activations, total process memory'}


def vmap_jacobian(model, states, chunk=16, max_bytes=DEFAULT_MAX_SR_BYTES):
    states=_states(model,states)  # Same integer/range/fixed-sector validation as core.
    budget=prototype_budget(model,len(states),chunk,max_bytes)
    params={name:p.detach() for name,p in model.named_parameters() if p.requires_grad}
    buffers={name:b.detach() for name,b in model.named_buffers()}
    def single(parameters, onehot):
        return functional_call(model,(parameters,buffers),(onehot.unsqueeze(0),)).squeeze()
    per_sample=vmap(grad(single),in_dims=(None,0),randomness='error')
    attention=any(isinstance(module,nn.MultiheadAttention) for module in model.modules())
    context=torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if attention else nullcontext()
    with context:
        with torch.no_grad():
            _log_amplitudes(model,states,256)  # Preserve finite-forward validation outside transform.
        J=torch.empty((len(states),budget['parameters']),device=states.device,dtype=torch.float64)
        for start in range(0,len(states),budget['vmap_rows']):
            block=states[start:start+budget['vmap_rows']]
            onehot=F.one_hot(block,model.Nmax+1).to(next(model.parameters()).dtype)
            gradients=per_sample(params,onehot)
            offset=0
            for name,p in params.items():
                J[start:start+len(block),offset:offset+p.numel()]=gradients[name].detach().reshape(len(block),-1)
                offset+=p.numel()
    if not torch.isfinite(J).all():
        raise RuntimeError('non-finite vmap Jacobian')
    return J


def small_checks(device, report, persist):
    for arch in ARCH_BUILDERS:
        for axes in ((False,True),(True,False)):
            torch.manual_seed(17)
            settings={} if arch=='JastrowNet' else {'n_hidden' if arch=='RBMNet' else 'hidden_dim':4}
            model=build_model(arch,3,2,Nmax=2,pbc_x=axes[0],pbc_y=axes[1],**settings).double().eval()
            model.register_parameter('unused_parameter',nn.Parameter(torch.tensor([.2,-.1],dtype=torch.float64)))
            states=generate_initial_states(6,2,2,17,'cpu')
            if device.type=='cuda':
                model=model.float()
            expected=compute_log_psi_jacobian(model,states)  # CPU same-dtype scalar reference.
            model=model.to(device)
            original={n:t.detach().clone() for n,t in model.state_dict().items()}
            for chunk in (16,64,128):
                actual=vmap_jacobian(model,states.to(device),chunk).cpu()
                tolerance=(1e-10,1e-11) if device.type=='cpu' else (5e-5,5e-6)
                torch.testing.assert_close(actual,expected,rtol=tolerance[0],atol=tolerance[1])
                offset=0
                for name,p in model.named_parameters():
                    if name=='unused_parameter':
                        assert torch.count_nonzero(actual[:,offset:offset+p.numel()])==0
                    offset+=p.numel()
                assert all(torch.equal(t,original[n]) for n,t in model.state_dict().items())
                report['correctness'].append({'arch':arch,'axes':list(axes),'backend':str(device),
                                               'dtype':str(next(model.parameters()).dtype),'chunk':chunk,
                                               'comparison':differences(actual,expected),'unused_gradient_zero':True,
                                               'state_dict_unchanged':True,'rtol':tolerance[0],'atol':tolerance[1]})
            # Guard must reject before calling forward, and retain state validation.
            with patch.object(model,'forward',side_effect=AssertionError('forward before guard')):
                try: vmap_jacobian(model,states.to(device),16,max_bytes=1)
                except ValueError: pass
                else: raise AssertionError('missing allocation guard')
            invalid=states.clone(); invalid[0,0]=3
            try: vmap_jacobian(model,invalid.to(device),16)
            except ValueError: pass
            else: raise AssertionError('missing occupation validation')
            invalid=states.clone(); invalid[0]=0
            try: vmap_jacobian(model,invalid.to(device),16)
            except ValueError: pass
            else: raise AssertionError('missing fixed-sector validation')
            persist()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/vmap-jacobian-prototype.json'))
    args=parser.parse_args(argv)
    output=(ROOT/'benchmarks'/args.output).resolve()
    if not output.is_relative_to(ROOT/'benchmarks/results') or output.exists():
        parser.error('output must be new and under benchmarks/results/')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    snapshot=gpu_snapshot()
    idle=[g for g in snapshot['gpus'] if g[0]=='1' and not external_processes(snapshot,g[1]) and int(g[5])<=10 and int(g[3])<1024]
    selected=idle[0] if idle and torch.cuda.is_available() else None
    report={'status':'running','recorded_at':datetime.now(timezone.utc).isoformat(),
            'command':[sys.executable,str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv)],
            'source_sha256_at_start':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                                     ('models.py','vmc.py','train.py','lattice.py','benchmarks/benchmark_vmap_jacobian.py','benchmarks/benchmark_accelerator_sr.py')},
            'environment':{'torch':str(torch.__version__),'threads':1,'TF32':False},
            'gpu_snapshot_at_start':snapshot,'selected_gpu':selected,'correctness':[],'timings':[],
            'limits':'One8-state warmup and one M512 Jacobian timing per case; no solver, optimizer update, sampling or training. '
                     'CPU double and GPU float32 differ in arithmetic. Workspace guards exclude activations; peak VRAM recorded. '
                     'Finite-forward and occupation/sector validation retained; no forward or architecture bypass.'}
    def persist():
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    output.parent.mkdir(parents=True,exist_ok=True)
    persist()
    try:
        small_checks(torch.device('cpu'),report,persist)
        devices=[torch.device('cpu')]
        if selected:
            now=gpu_snapshot()
            if external_processes(now,selected[1]) or 'error' in now:
                report['cuda_status']='skipped: GPU became busy/unavailable'
            else:
                small_checks(torch.device('cuda:'+selected[0]),report,persist)
                devices.append(torch.device('cuda:'+selected[0]))
        else: report['cuda_status']='skipped: no idle accessible GPU1'
        print('All-five small reference/unused-parameter and validation checks passed',flush=True)
        for arch,width in (('Conv2DNet',32),('TransformerNet',64)):
            torch.manual_seed(123)
            frozen=build_model(arch,4,4,Nmax=4,pbc_x=False,pbc_y=True,hidden_dim=width).double().eval()
            states_cpu=generate_initial_states(16,16,4,512,'cpu')
            for device in devices:
                if device.type=='cuda':
                    now=gpu_snapshot()
                    if 'error' in now or external_processes(now,selected[1]):
                        report['cuda_status']='skipped remaining timings: GPU became busy/unavailable'; persist(); continue
                model=copy.deepcopy(frozen).to(device=device,dtype=torch.float64 if device.type=='cpu' else torch.float32)
                states=states_cpu.to(device)
                original={n:t.detach().clone() for n,t in model.state_dict().items()}
                reference=None
                reference_seconds=None
                for method,chunk in (('current_vjp',16),('vmap_grad',16),('vmap_grad',64),('vmap_grad',128)):
                    if device.type=='cuda':
                        now=gpu_snapshot()
                        if 'error' in now or external_processes(now,selected[1]):
                            report['cuda_status']='skipped remaining timings: GPU became busy/unavailable'; break
                    function=(lambda x:compute_log_psi_jacobian(model,x,jacobian_batch_size=chunk)) if method=='current_vjp' else (lambda x:vmap_jacobian(model,x,chunk))
                    function(states[:8]); synchronize(device)
                    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
                    started=time.perf_counter(); J=function(states); synchronize(device); seconds=time.perf_counter()-started
                    peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
                    if reference is None:
                        reference=J.detach().cpu(); reference_seconds=seconds
                    difference=differences(J.cpu(),reference)
                    tolerance=(1e-10,1e-11) if device.type=='cpu' else (5e-5,5e-6)
                    torch.testing.assert_close(J.cpu(),reference,rtol=tolerance[0],atol=tolerance[1])
                    assert all(torch.equal(t,original[n]) for n,t in model.state_dict().items())
                    row={'arch':arch,'width':width,'geometry':[4,4],'boundaries':[False,True],'Ntot':16,'Nmax':4,
                         'M':512,'backend':str(device),'dtype':str(next(model.parameters()).dtype),'method':method,'chunk':chunk,
                         'parameters':parameter_count(model),'seconds':seconds,'speedup_vs_current_vjp16':reference_seconds/seconds,
                         'difference_vs_current_same_device_dtype':difference,'cuda_peak_allocated_bytes':peak,
                         'peak_scope':'reset after resident model/input and small warmup; includes resident baseline and Jacobian construction; CPU reference is off-device',
                         'budget':check_sr_budget(model,512,jacobian_batch_size=16) if method=='current_vjp' else prototype_budget(model,512,chunk),
                         'weights':'same fresh seed123 frozen weights rounded to selected dtype','states':'same feasible initializer states; not target-distributed MC samples'}
                    report['timings'].append(row); persist()
                    print(arch,str(device),method,chunk,f'{seconds:.5f}s',f'speedup={reference_seconds/seconds:.3f}',flush=True)
                    del J
                del model,states,reference,original
        report['source_hashes_unchanged_at_end']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in report['source_sha256_at_start'].items()}
        assert all(report['source_hashes_unchanged_at_end'].values())
        report['status']='completed'
    except Exception as error:
        report['status']='failed_stopped'; report['error']=f'{type(error).__name__}: {error}'
        persist()
        raise
    persist()
    print(output)
    return report


if __name__=='__main__':
    main()
