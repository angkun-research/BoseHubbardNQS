"""Collect compact GCNN results, preserving exact versus sampled distinctions.

Campaign-specific: the study description, physics block, deferrals, limitations
and pilot notes describe the 2026-11-13 GCNN study. Numeric rows are read from
the reports under --study-dir.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from observables import block_mean_statistics

DEFAULT_STUDY_DIR = Path('outputs/2026-11-13-gcnn-study')
DEFAULT_OUTPUT = Path('docs/gcnn-study-results.json')
FROZEN_SEEDS = (321, 654)
BLOCK_LENGTHS = (16, 32, 64, 128)


def _resolve(path):
    path = Path(path)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _relative(path):
    """Report paths relative to pythonTest/2D when possible."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read(path):
    return json.loads(path.read_text())


def supervised_rows(folder):
    rows = []
    for shape in ('2x3', '3x4'):
        path = folder / f'ed-{shape}/report.json'
        if not path.exists():
            continue
        report = read(path)
        row = {'shape': shape, 'status': report['status'], 'report': _relative(path)}
        for key in ('amplitude_overlap_abs_unsquared', 'fidelity_squared', 'energy', 'energy_error_per_site',
                    'variance', 'best_epoch', 'best_training_log_mse', 'elapsed_seconds', 'orbit_dimension'):
            if key in report:
                row[key] = report[key]
        if 'dataset' in report:
            row.update(dimension=report['dataset']['dimension'], ED=report['dataset']['raw_ed_reference'],
                       floored_target_count=report['dataset']['clamped_target_count'])
        if report['status'] == 'completed':
            row['working_overlap_gate'] = {'threshold_abs_unsquared': .99,
                                           'passed': report['amplitude_overlap_abs_unsquared'] >= .99}
            if shape == '2x3':
                row['tiny_system_gate_passed'] = (report['amplitude_overlap_abs_unsquared'] >= .9999
                                                  and report['energy_error_per_site'] <= .001)
        rows.append(row)
    path = folder / 'ed-3x3-width8.pt.json'
    if path.exists():
        report = read(path)
        if report['status'] == 'completed':
            metric = report['best_checkpoint_metrics']
            rows.append({'shape': '3x3', 'status': 'completed', 'report': _relative(path),
                         'amplitude_overlap_abs_unsquared': metric['F_abs'], 'fidelity_squared': metric['F_abs']**2,
                         'energy': metric['energy'], 'energy_error_per_site': metric['energy_error_per_site'],
                         'variance': metric['variance'], 'best_epoch': report['best_epoch']})
    return rows


def frozen_row(path, size, seed):
    report = read(path)
    with np.load(str(path) + '.samples.npz') as raw:
        energies = raw['local_energy']
    blocks = []
    for length in BLOCK_LENGTHS:
        stats = block_mean_statistics(torch.from_numpy(energies), length)
        blocks.append({'block_length': length, 'energy_mean': stats['mean'].item(),
                       'conditional_block_SEM': stats['block_standard_error'].item(),
                       'between_chain_SEM': stats['between_chain_standard_error'].item()})
    return {'seed': seed, 'energy': float(energies.mean()),
            'energy_per_site': float(energies.mean()) / (size * size),
            'local_energy_variance': float(energies.var()),
            'sampling': report['sampling'], 'block_stability': blocks,
            'conditional_errors_only': True, 'report': _relative(path)}


def vmc_rows(folder):
    rows = []
    for size in (3, 4, 5):
        directory = folder / f'vmc-{size}x{size}-seed123'
        path = directory / 'suite.json'
        if not path.exists():
            continue
        row = {'shape': f'{size}x{size}', 'status': read(path)['status'], 'suite': _relative(path),
               'training_seed': 123, 'frozen': []}
        training = directory / 'final.pt.json'
        if training.exists():
            report = read(training)
            row.update(final_training_measurement=report['final_measurement'],
                       training_seconds=report['training_and_final_measurement_seconds'],
                       checkpoint_sha256=hashlib.sha256((directory / 'final.pt').read_bytes()).hexdigest())
            # Explicitly label noisy means from the final 100 pre-update measurements.
            history = report['history'][-100:]
            split = max(1, len(history) // 2)
            row['training_tail'] = {
                'first_half_mean_energy': float(np.mean([x['energy'] for x in history[:split]])),
                'second_half_mean_energy': (float(np.mean([x['energy'] for x in history[split:]]))
                                            if len(history) > split else None),
                'number_of_measurements': len(history),
                'interpretation': 'descriptive correlated training measurements; no convergence test'}
        exact = directory / 'exact.json'
        if exact.exists():
            row['exact_saved_state'] = read(exact)
        for seed in FROZEN_SEEDS:
            path = directory / f'frozen-{seed}.json'
            if path.exists():
                row['frozen'].append(frozen_row(path, size, seed))
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=DEFAULT_STUDY_DIR,
                        help='campaign directory; relative paths start at pythonTest/2D')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help='compact JSON destination; relative paths start at pythonTest/2D')
    args = parser.parse_args(argv)
    folder, destination = _resolve(args.study_dir), _resolve(args.output)
    if not folder.is_dir():
        parser.error(f'study directory does not exist: {folder}')

    result = {
        'study': 'minimal D4-tied GCNN,2026-11-13', 'parameter_count': 4993,
        'environment_at_collection': {
            'python': sys.version, 'torch': torch.__version__,
            'numpy': np.__version__, 'scipy': scipy.__version__, 'torch_cuda_build': torch.version.cuda,
            'hardware': 'two NVIDIA RTX A6000 48GiB;125GiB host RAM',
            'timing_scope': 'concurrent jobs; wall times are not isolated performance benchmarks'},
        'physics': {'t': 1., 'U': 16.74, 'mu': 0., 'Nmax': 4, 'filling': 1, 'pbc_x': True, 'pbc_y': True},
        'ansatz': 'width8 lifting3x3+SiLU,group3x3+SiLU,space/orientation mean,linear head',
        'rectangular_symmetry': 'D4-tied local filters; only D2 and translations on each rectangle',
        'supervised': supervised_rows(folder), 'vmc': vmc_rows(folder),
        'deferred': {'3x5': 'existing full ED exceeds resource budget; new orbit-reduced solver explicitly deferred',
                     '4x5': 'existing full ED exceeds resource budget; replaced with2x3 exact test'},
        'limitations': ['single training seed per size; frozen seeds are independent measurements, not optimization replicas',
                        'fixed training budget does not establish convergence or an architecture capacity limit',
                        'no exact fidelity for4x4 or5x5; sampled energy is not an exact variational bound',
                        'block errors are conditional; stability diagnostics do not establish mixing or ESS',
                        '2x3 periodic length-two axis aliases opposite convolution offsets'],
        'pilots': {'3x3_float64': 'completed; pipeline check only',
                   '5x5_float32_lr.02_tau.001': 'failed: explosive sampled energy then nonfinite local energies',
                   '5x5_float64_lr.002_tau.01': '10 stable updates; conservative protocol selected'},
    }
    audit = folder / 'saved-fidelity-audit.json'
    if audit.exists():
        result['independent_saved_checkpoint_fidelity_audit'] = read(audit)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(destination)


if __name__ == '__main__':
    main()
