"""Fixed backup-only hybrid qualification and two missing cylinder baselines.

Prepare publishes commands, guards and source hashes without launching jobs.
Run requires the independent checker readiness record and refuses old outputs.
Only the predeclared fixed budgets run; a missed scientific gate is retained.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import build_model, load_model, save_model, parameter_count
from vmc import check_sr_budget
from exact_diagonalization import exact_ground_state
from run_leakage_analysis import canonical_metrics
from periodic_observables import physical_bonds, sparse_bond_operators, check_operator_H, state_observables, frozen_observables

R = ROOT / 'benchmarks/results/2026-11-01-backup-extension'
O = ROOT / 'outputs/2026-11-01-backup-extension'
PROTOCOL = R / 'backup-extension-protocol.json'
STATUS = R / 'backup-extension-status.json'
SOURCES = ('models.py', 'train.py', 'train_ed.py', 'vmc.py', 'evaluate.py', 'observables.py',
           'lattice.py', 'exact_diagonalization.py', 'leakage_analysis.py',
           'benchmarks/run_backup_extension.py', 'benchmarks/run_leakage_analysis.py',
           'benchmarks/periodic_observables.py', 'benchmarks/assess_refinement.py',
           'benchmarks/verify_gpu.py', 'benchmarks/run_ed_fits.py', 'benchmarks/run_baseline.py')
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MPLCONFIGDIR='/tmp/bhnqs-2d-mpl')
LOCK = threading.RLock()
CPU = threading.Semaphore(4)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        tmp.replace(path)


def command(script, *args):
    return [sys.executable, '-u', str(ROOT / script), *map(str, args)]


def training(label, arch, Lx, Ly, width, device, epochs, walkers, steps, thin, lr, lr_end, tau, tau_end):
    checkpoint = O / (label + '.pt')
    axes = ['--no-pbc-x', '--pbc-y'] if Lx == 6 else ['--pbc-x', '--pbc-y']
    cmd = command('train.py', '--arch', arch, '--hidden-dim', width, '--Lx', Lx, '--Ly', Ly,
        '--Ntot', Lx*Ly, '--Nmax', 4, *axes, '--t', 1, '--U', 16.74, '--mu', 0,
        '--epochs', epochs, '--walkers', walkers, '--steps', steps, '--initial-burnin', 4096,
        '--burnin', 64, '--thin', thin, '--update-method', 'sr', '--optimizer', 'sgd',
        '--sr-regularizer', 'ridge', '--lr', lr, '--lr-end', lr_end, '--tau', tau, '--tau-end', tau_end,
        '--jacobian-method', 'vmap', '--jacobian-batch-size', 128, '--max-sr-bytes', 4294967296,
        '--device', device, '--dtype', 'float64' if device == 'cpu' else 'float32', '--seed', 123,
        '--evaluation-batch-size', 256, '--output', checkpoint)
    if device != 'cpu':
        cmd.append('--disable-tf32')
    frozen = {}
    for seed in (321, 654):
        output = O / 'frozen' / f'{label}-seed{seed}.json'
        frozen[str(seed)] = command('evaluate.py', '--checkpoint', checkpoint, '--device', 'cpu', '--dtype', 'float64',
            '--walkers', 128 if Lx == 6 else 64, '--steps', 512, '--burnin', 4096,
            '--thin', 16 if Lx == 6 else 8, '--block-length', 64, '--batch-size', 256,
            '--max-observable-bytes', 1073741824, *([] if Lx == 6 else ['--full-rho']),
            '--seed', seed, '--output', output)
    row = {'label': label, 'family': arch, 'checkpoint': str(checkpoint), 'command': cmd,
           'geometry': [Lx, Ly], 'dtype': 'float64' if device == 'cpu' else 'float32',
           'fresh_initialization': True, 'seed': 123, 'frozen_commands': frozen}
    if Lx == 6:
        row['assessment_command'] = command('benchmarks/assess_refinement.py', '--reference',
            ROOT / 'outputs/dmrg-6x4-cylinder-chi128', '--phase', 'chi128', '--reports',
            *[O / 'frozen' / f'{label}-seed{seed}.json' for seed in (321, 654)], '--block-length', 64,
            '--output', R / (label + '-assessment.json'))
    return row


def prepare():
    if PROTOCOL.exists() or STATUS.exists():
        raise RuntimeError('refuse replacing an existing protocol/run')
    torch.set_num_threads(1)
    rows = [training('Jastrow-basic-6x4-sr1000-seed123', 'JastrowNet', 6, 4, 32, 'cpu', 1000, 64, 16, 16, .02, .002, .001, .0001),
            training('RBM128-6x4-sr1000-seed123', 'RBMNet', 6, 4, 128, 'cuda:1', 1000, 1024, 2, 32, .02, .002, .001, .00001),
            training('Hybrid32-2x3-sr500-seed123', 'Conv2DTransformerNet', 2, 3, 32, 'cuda:1', 500, 64, 16, 8, .02, .002, .001, .0001)]
    for row in rows:
        Lx, Ly = row['geometry']
        settings = {} if row['family'] == 'JastrowNet' else {'n_hidden' if row['family'] == 'RBMNet' else 'hidden_dim': 128 if row['family'] == 'RBMNet' else 32}
        model = build_model(row['family'], Lx, Ly, Nmax=4, pbc_x=Lx != 6, pbc_y=True, **settings)
        row['parameter_count'] = parameter_count(model)
        M = 2048 if row['family'] == 'RBMNet' else 1024
        row['SR_allocation_preflight'] = check_sr_budget(model, M, max_sr_bytes=4294967296,
            jacobian_method='vmap', jacobian_batch_size=128)
    qualifications = []
    for Lx in (2, 3):
        label = f'Hybrid32-{Lx}x3-ed-logmse10000-seed123'
        checkpoint = O / (label + '.pt')
        qualifications.append({'label': label, 'checkpoint': str(checkpoint), 'geometry': [Lx, 3],
            'command': command('train_ed.py', '--arch', 'Conv2DTransformerNet', '--width', 32,
                '--Lx', Lx, '--Ly', 3, '--Ntot', Lx*3, '--Nmax', 4, '--pbc-x', '--pbc-y',
                '--t', 1, '--U', 16.74, '--mu', 0, '--loss', 'log_mse', '--epochs', 10000,
                '--lr', .001, '--seed', 123, '--device', 'cuda:1', '--dtype', 'float32',
                '--disable-tf32', '--eval-every', 1000, '--output', checkpoint)})
    protocol = {'status': 'prepared_before_any_benchmark_launch', 'recorded_at': now(),
        'scope': 'Backup only. Fixed final VMC seed123, both frozen seeds retained; no favorable seed selection, retuning or accuracy chasing.',
        'source_sha256': {name: sha(ROOT/name) for name in SOURCES},
        'resources': {'GPU_training_max': 2, 'GPU_indices': [1], 'CPU_jobs_max': 4, 'CPU_threads': 1},
        'training_runs': rows, 'supervised_runs': qualifications, 'maximum_training_run_count': 5,
        'phase_order': 'Basic J CPU and fresh RBM GPU may start together with sequential hybrid ED fits on the second GPU slot. Hybrid VMC only after BOTH saved-checkpoint independent CPUdouble qualification gates. No hybrid6x4.',
        'qualification': {'2x3': 'raw unsquared F>=.9999 AND energy_error/site<=.001', '3x3': 'raw unsquared F>=.99',
            'RBM128': 'existing 3x3 cylinder savedweights independently checked F>=.99 before new RBM6 training',
            'Jastrow': 'original displacement physical baseline exempt from admission/accuracy requirement; all targets still reported'},
        'leakage_command': command('benchmarks/run_leakage_analysis.py', '--selection', R/'hybrid-primary-selection.json',
            '--output', R/'hybrid-periodic-leakage-results.json', '--batch-size', 2048, '--full-time-budget', 600),
        'selection_semantics': 'Supervised: strict best full-sector normalized logMSE, Adam. VMC: fixed final post-update checkpoint, fresh SGD/walkers, no ED labels/warmstart; exact diagnostics do not enter optimization.',
        'precision': 'GPUmodelFP32 TF32off, SRfloat64 unchanged; every saved-state exact/frozen diagnostic CPUfloat64. Memory guard excludes activations/vendor workspace.',
        'results': str(R/'backup-extension-results.json')}
    write(PROTOCOL, protocol)
    write(R/'backup-extension-schema.json', {'protocol': str(PROTOCOL), 'status': str(STATUS),
        'results': str(R/'backup-extension-results.json'), 'field_paths': {
            'cylinder_baselines': 'results.cylinders.<JastrowNet|RBMNet>: checkpoint, parameter_count, trainer_report, frozen.<321|654>, assessment',
            'hybrid_supervised': 'results.hybrid.supervised.<2x3|3x3>: checkpoint, trainer_report, independent_CPUfloat64, admission_passed',
            'hybrid_VMC': 'results.hybrid.vmc: exact_initial, exact_final, ED, frozen.<321|654>, trainer_report',
            'hybrid_leakage': 'hybrid-periodic-leakage-results.json: models[0].canonical/nearby.sectors/full.sectors/projection; same established leakage schema'},
        'binary_and_raw_samples': str(O), 'text_reports_and_logs': str(R), 'tables_use_seed': 654,
        'missed_gates': 'scientific observation, not process failure; both seeds always retained'})
    print(PROTOCOL, flush=True)


def run():
    protocol = json.loads(PROTOCOL.read_text())
    approval = R/'checker-backup-extension-readiness.json'
    review = json.loads(approval.read_text())
    if not review.get('training_readiness_approved', False):
        raise RuntimeError('independent checker has not approved frozen protocol yet')
    for name, digest in protocol['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError(f'source changed after protocol: {name}')
    if STATUS.exists():
        raise RuntimeError('refuse existing run status')
    for row in protocol['training_runs'] + protocol['supervised_runs']:
        if Path(row['checkpoint']).exists() or Path(row['checkpoint']+'.json').exists():
            raise RuntimeError('refuse existing benchmark checkpoint/report')
    torch.set_num_threads(1)
    status = {'status': 'running', 'started_at': now(), 'expected_training_run_count': 5,
              'completed_training_run_count': 0, 'skipped_training_run_count': 0, 'failures': [], 'runs': {},
              'source_sha256_at_start': protocol['source_sha256'], 'checker_approval_sha256': sha(approval)}
    results = {'status': 'running', 'cylinders': {}, 'hybrid': {'supervised': {}},
               'protocol': str(PROTOCOL), 'source_sha256_at_start': protocol['source_sha256'],
               'interpretation': 'All data backup only; Jastrow is basic displacement physical baseline; no target is waived in recorded assessment.'}
    def persist():
        with LOCK:
            write(STATUS, status)
            write(R/'backup-extension-results.json', results)
    def execute(label, cmd, training_job=False):
        logpath = R/(label+'.log')
        record = {'status': 'prepared', 'command': cmd, 'log': str(logpath)}
        with LOCK:
            status['runs'][label] = record
            persist()
        started = time.perf_counter()
        with logpath.open('x') as log:
            child = subprocess.Popen(cmd, env=ENV, cwd=ROOT.parents[1], stdout=log, stderr=subprocess.STDOUT)
            with LOCK:
                record.update(status='running', pid=child.pid, started_at=now())
                persist()
            code = child.wait()
        with LOCK:
            record.update(status='completed' if code == 0 else 'execution_failed', exit_code=code,
                wall_seconds=time.perf_counter()-started, finished_at=now())
            if training_job and code == 0:
                status['completed_training_run_count'] += 1
            if code:
                status['failures'].append({'label': label, 'exit_code': code, 'log': str(logpath)})
            persist()
        if code:
            raise RuntimeError(f'{label} failed: see {logpath}')
        print('completed', label, 'seconds', record['wall_seconds'], flush=True)
        return record
    def retain(source, target):
        target = R/target
        if target.exists():
            raise RuntimeError('refuse replacing retained report')
        shutil.copyfile(source, target)
        return str(target)
    def exact(checkpoint, Lx, Ly, px, full=False):
        digest = sha(checkpoint)
        model, metadata = load_model(Path(checkpoint).resolve(), 'cpu')
        if sha(checkpoint) != digest:
            raise RuntimeError('checkpoint changed during load')
        model = model.double().eval()
        expected = {'Lx': Lx, 'Ly': Ly, 'Ntot': Lx*Ly, 'Nmax': 4, 'pbc_x': px, 'pbc_y': True, 't': 1., 'U': 16.74, 'mu': 0.}
        if any(metadata[k] != v for k,v in expected.items()):
            raise RuntimeError('physical checkpoint mismatch')
        E, raw, basis, H = exact_ground_state(Lx, Ly, Lx*Ly, 4, 1., 16.74, 0., True, pbc_x=px, pbc_y=True)
        basis = np.asarray(basis, dtype=np.int64)
        metrics = canonical_metrics(model, np.asarray(raw), basis, H, E, 256)
        metrics.update(checkpoint_sha256=digest, checkpoint_metadata=metadata, parameter_count=parameter_count(model))
        if full:
            from leakage_analysis import evaluate_logs
            logs = np.concatenate([evaluate_logs(model, basis[i:i+256]) for i in range(0,len(basis),256)])
            psi = np.exp(logs-logs.max()); psi /= np.linalg.norm(psi)
            bonds = physical_bonds(Lx,Ly,px,True)
            operators = sparse_bond_operators(basis,4,bonds)
            check_operator_H(H,basis,operators,1.,16.74,0.)
            metrics['observables'] = state_observables(psi,basis,H,bonds,operators=operators)
            metrics['ED'] = state_observables(raw,basis,H,bonds,operators=operators)
        return metrics
    def frozen(row, result):
        result['frozen'] = {}
        for seed, cmd in row['frozen_commands'].items():
            with CPU:
                execute(row['label']+'-frozen'+seed, cmd)
                output = Path(cmd[cmd.index('--output')+1])
                report = json.loads(output.read_text())
                if report['checkpoint_sha256'] != result['checkpoint_sha256']:
                    raise RuntimeError('frozen loaded different weights')
                item = {'report': retain(output, row['label']+'-frozen'+seed+'.json'),
                        'raw_samples': str(output)+'.samples.npz', 'checkpoint_sha256': report['checkpoint_sha256']}
                if row['geometry'] == [2,3]:
                    with np.load(item['raw_samples']) as raw:
                        item['observables'] = frozen_observables(raw, physical_bonds(2,3), block_length=64)
                    item['energy_statistics'] = report['statistics']['energy']
                result['frozen'][seed] = item
                persist()
        if 'assessment_command' in row:
            with CPU:
                execute(row['label']+'-assessment', row['assessment_command'])
            result['assessment'] = row['assessment_command'][-1]
            result['assessment_summary'] = json.loads(Path(result['assessment']).read_text())
        result['status'] = 'completed'
        persist()
    def cylinder(row):
        result = {'status': 'training', 'checkpoint': row['checkpoint'], 'parameter_count': row['parameter_count'],
                  'baseline': 'original displacement' if row['family']=='JastrowNet' else 'fresh128',
                  'fixed_final_selection': True}
        with LOCK:
            results['cylinders'][row['family']] = result
            persist()
        if row['family']=='JastrowNet':
            with CPU:
                execution = execute(row['label'],row['command'],True)
        else:
            execution = execute(row['label'],row['command'],True)
        result.update(checkpoint_sha256=sha(row['checkpoint']),training_wall_seconds=execution['wall_seconds'],
            trainer_report=retain(row['checkpoint']+'.json',row['label']+'-trainer.json'))
        frozen(row,result)
    def hybrid():
        passed = True
        for row in protocol['supervised_runs']:
            execute(row['label'],row['command'],True)
            with CPU:
                Lx,Ly = row['geometry']
                metrics = exact(row['checkpoint'],Lx,Ly,True)
            gate = metrics['F_N0'] >= (.9999 if Lx==2 else .99)
            if Lx==2:
                gate = gate and metrics['energy_error_to_ed']/6 <= .001
            key = f'{Lx}x{Ly}'
            result = {'checkpoint': row['checkpoint'], 'trainer_report': retain(row['checkpoint']+'.json',row['label']+'-trainer.json'),
                      'independent_CPUfloat64': metrics, 'admission_passed': bool(gate)}
            results['hybrid']['supervised'][key] = result
            passed = passed and gate
            persist()
        row3 = results['hybrid']['supervised']['3x3']
        m = row3['independent_CPUfloat64']
        write(R/'hybrid-primary-selection.json', {'geometry':[3,3],'N0':9,'Nmax':4,'pbc_x':True,'pbc_y':True,
            't':1.,'U':16.74,'mu':0.,'selection':'predeclared primary Adam logMSE, seed123, width32,10000updates; no favorable CP family selection',
            'models':[{'family':'Conv2DTransformerNet','checkpoint':row3['checkpoint'],
                       'checkpoint_sha256':m['checkpoint_sha256'],'metadata':m['checkpoint_metadata']}]})
        # Leakage is diagnostic even when a supervised admission gate is missed.
        leak_future = cpu_pool.submit(leakage)
        if passed:
            row = protocol['training_runs'][2]
            with CPU:
                torch.manual_seed(123)
                model=build_model('Conv2DTransformerNet',2,3,Nmax=4,hidden_dim=32,pbc_x=True,pbc_y=True)
                initial=O/'Hybrid32-2x3-initial-seed123.pt'
                save_model(initial,model,6,1.,16.74,0.)
                initial_metrics=exact(initial,2,3,True,True)
            result={'status':'training','checkpoint':row['checkpoint'],'parameter_count':row['parameter_count'],
                    'exact_initial':initial_metrics,'initial_checkpoint':str(initial),
                    'initialization':'Exact fresh CPUdefaultFP32 constructor seed123, then CUDAFP32 in trainer. Initial diagnostic CP is NEVER a warmstart.',
                    'trajectory_scope':'Initial/final exact weights only. Trainer history is pre-update Monte Carlo, not exact curve.'}
            results['hybrid']['vmc']=result;persist()
            execution=execute(row['label'],row['command'],True)
            with CPU:
                final=exact(row['checkpoint'],2,3,True,True)
            result.update(exact_final=final,checkpoint_sha256=final['checkpoint_sha256'],
                training_wall_seconds=execution['wall_seconds'],trainer_report=retain(row['checkpoint']+'.json',row['label']+'-trainer.json'))
            frozen(row,result)
        else:
            results['hybrid']['vmc']={'status':'skipped_supervised_admission_missed','reason':'No stochastic VMC without both predeclared raw-ED gates.'}
            status['skipped_training_run_count']+=1;persist()
        leak_future.result()
    def leakage():
        with CPU:
            execute('Hybrid32-3x3-periodic-leakage',protocol['leakage_command'])
        results['hybrid']['leakage']=str(R/'hybrid-periodic-leakage-results.json');persist()
    persist()
    # Admission is diagnostic and precedes any RBM6 launch; existing data untouched.
    with CPU:
        admission=exact(ROOT/'outputs/2026-10-31-all-model-refinement/qualification/RBM128-3x3-prepared.pt',3,3,False)
    write(R/'RBM128-existing-qualification-independent.json',admission)
    if admission['F_N0']<.99:
        raise RuntimeError('existing RBM128 admission failed')
    # One CPU baseline, one GPU RBM workflow, one GPU hybrid workflow; never GPU0.
    with ThreadPoolExecutor(max_workers=4) as cpu_pool, ThreadPoolExecutor(max_workers=2) as gpu_pool:
        futures=[cpu_pool.submit(cylinder,protocol['training_runs'][0]),
                 gpu_pool.submit(cylinder,protocol['training_runs'][1]),gpu_pool.submit(hybrid)]
        for future in futures:
            try:
                future.result()
            except Exception as error:
                with LOCK:
                    status['failures'].append({'error':f'{type(error).__name__}: {error}'})
                    persist()
    status['status']=results['status']='completed_with_execution_failures' if status['failures'] else 'completed'
    status['finished_at']=now();persist()
    print(R/'backup-extension-results.json',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    args=parser.parse_args()
    prepare() if args.action=='prepare' else run()


if __name__=='__main__':
    main()
