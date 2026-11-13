"""Run a bounded GCNN VMC study using the existing trainer and evaluator.

For each square size, train one width-8 GroupEquivariantConv2DNet with the
existing SR + native SGD trainer, evaluate the saved 3x3 state exactly against
ED, and take two frozen measurements (seeds 321 and 654) with evaluate.py.
Every job keeps its log; failures are recorded in suite.json and re-raised.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exact_diagonalization import exact_ground_state
from models import load_model

DEFAULT_OUTPUT_DIR = Path('outputs/2026-11-13-gcnn-study')
TRAINING_SEED = 123
FROZEN_SEEDS = (321, 654)


def _output_dir(value):
    """Resolve relative paths from pythonTest/2D; stay under its outputs/."""
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT / 'outputs'):
        raise ValueError('output directory must be under pythonTest/2D/outputs/')
    return path


def exact_saved_state(checkpoint, device):
    """Full-sector float64 metrics of a saved checkpoint against its own ED sector."""
    model, metadata = load_model(checkpoint, device=device)
    model = model.double()
    energy, raw, basis, H = exact_ground_state(
        metadata['Lx'], metadata['Ly'], metadata['Ntot'], metadata['Nmax'],
        metadata['t'], metadata['U'], metadata['mu'],
        pbc_x=metadata['pbc_x'], pbc_y=metadata['pbc_y'],
    )
    states = torch.tensor(basis, device=device)
    with torch.no_grad():
        logs = torch.cat([model(F.one_hot(block, metadata['Nmax'] + 1).double()).cpu()
                          for block in states.split(256)])
    psi = (logs - 0.5 * torch.logsumexp(2 * logs, dim=0)).exp().numpy()
    applied = H @ psi
    predicted = float(psi @ applied)
    overlap = float(abs(raw @ psi))
    return {
        'ed_energy': energy, 'energy': predicted,
        'energy_error_per_site': (predicted - energy) / model.n_sites,
        'variance': float(np.linalg.norm(applied - predicted * psi)**2),
        'amplitude_overlap_abs_unsquared': overlap, 'fidelity_squared': overlap**2,
        'basis_dimension': len(basis),
        'precision': 'direct full-sector float64 saved-state evaluation',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', required=True, choices=[3, 4, 5])
    parser.add_argument('--device', required=True)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float64')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help='campaign directory under pythonTest/2D/outputs/; '
                             'existing per-size subdirectories are never overwritten')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    try:
        folder = _output_dir(args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    folder.mkdir(parents=True, exist_ok=True)
    common = [sys.executable, '-B', '-u']

    for size in args.sizes:
        output = folder / f'vmc-{size}x{size}-seed{TRAINING_SEED}'
        output.mkdir(exist_ok=False)
        manifest = {'status': 'running', 'size': size, 'training_seed': TRAINING_SEED, 'jobs': []}

        def persist():
            (output / 'suite.json').write_text(json.dumps(manifest, indent=2) + '\n')

        def run(label, command):
            started = time.perf_counter()
            print('starting', size, label, flush=True)
            with (output / f'{label}.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            manifest['jobs'].append({'label': label, 'command': command, 'exit_code': result.returncode,
                                     'elapsed_seconds': time.perf_counter() - started})
            persist()
            print('finished', size, label, 'exit', result.returncode, flush=True)
            if result.returncode:
                raise RuntimeError(f'{label} failed; see retained log')

        persist()
        try:
            checkpoint = output / 'final.pt'
            run('training', common + [
                str(ROOT / 'train.py'), '--arch', 'GroupEquivariantConv2DNet',
                '--Lx', str(size), '--hidden-dim', '8', '--epochs', str(args.epochs),
                '--walkers', '64', '--steps', '16', '--initial-burnin', '4096', '--burnin', '64',
                '--thin', str(size * size), '--lr', '.002', '--lr-end', '.0002',
                '--tau', '.01', '--tau-end', '.001', '--update-method', 'sr', '--optimizer', 'sgd',
                '--jacobian-method', 'vmap', '--jacobian-batch-size', '128',
                '--dtype', args.dtype, '--device', args.device, '--disable-tf32',
                '--seed', str(TRAINING_SEED), '--output', str(checkpoint),
            ])
            if size == 3:
                exact = exact_saved_state(checkpoint, args.device)
                (output / 'exact.json').write_text(json.dumps(exact, indent=2) + '\n')
                print('exact', exact, flush=True)
            for seed in FROZEN_SEEDS:
                run(f'frozen-{seed}', common + [
                    str(ROOT / 'evaluate.py'), '--checkpoint', str(checkpoint),
                    '--device', args.device, '--dtype', args.dtype, '--seed', str(seed),
                    '--walkers', '64', '--steps', '256', '--burnin', '4096', '--thin', str(size * size),
                    '--block-length', '32', '--pair', '0', '0',
                    '--output', str(output / f'frozen-{seed}.json'),
                ])
            manifest['status'] = 'completed'
            persist()
        except BaseException as error:
            manifest.update(status='failed_retained', error=f'{type(error).__name__}: {error}')
            persist()
            raise


if __name__ == '__main__':
    main()
