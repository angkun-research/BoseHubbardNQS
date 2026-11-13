"""Bounded D4-tied GCNN log-MSE fit on a rectangular torus with full ED.

Only translations and D2 reflections are used to reduce the training inputs.
Orbit multiplicities and target moments reproduce the full-sector objective.
Final fidelity uses direct batched predictions of the reloaded saved checkpoint
on the complete raw ED vector.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import GroupEquivariantConv2DNet, load_model, parameter_count, save_model
from train_ed import make_ed_dataset
from train_square_group import orbit_log_mse

# Independent Burnside counts for translation x D2 orbits at unit filling, Nmax=4.
EXPECTED_ORBITS = {(2, 3): 46, (3, 2): 46, (3, 4): 21600, (4, 3): 21600}
SOURCES = ('models.py', 'square_group.py', 'exact_diagonalization.py', 'train_ed.py',
           'benchmarks/train_rectangular_group.py', 'benchmarks/train_square_group.py')


def rectangular_orbits(basis, lx, ly, nmax):
    """Return representative indices, full-basis labels and multiplicities."""
    basis = np.asarray(basis, dtype=np.int64)
    if basis.ndim != 2 or basis.shape[1] != lx * ly or not len(basis):
        raise ValueError('basis must be nonempty with Lx*Ly sites')
    if (nmax + 1)**(lx * ly) > np.iinfo(np.int64).max:
        raise ValueError('base-occupation codes exceed int64')
    powers = (nmax + 1)**np.arange(lx * ly - 1, -1, -1, dtype=np.int64)
    codes = basis @ powers
    if np.any(np.diff(codes) <= 0):
        raise ValueError('basis must be unique in lexicographic order')
    canonical = codes.copy()
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        for tx in range(lx):
            for ty in range(ly):
                permutation = [((sy * (y - ty)) % ly) * lx + (sx * (x - tx)) % lx
                               for y in range(ly) for x in range(lx)]
                canonical = np.minimum(canonical, basis[:, permutation] @ powers)
    representatives, labels, counts = np.unique(canonical, return_inverse=True, return_counts=True)
    indices = np.searchsorted(codes, representatives)
    if np.any(indices >= len(codes)) or not np.array_equal(codes[indices], representatives):
        raise ValueError('basis must contain every orbit representative')
    return indices, labels, counts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--Lx', type=int, default=3)
    parser.add_argument('--Ly', type=int, default=4)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--output', type=Path, required=True,
                        help='new directory under pythonTest/2D/outputs/')
    args = parser.parse_args(argv)
    args.output = (ROOT / args.output).resolve()
    if not args.output.is_relative_to(ROOT / 'outputs') or args.output.exists():
        parser.error('output must be new under 2D/outputs')
    if min(args.Lx, args.Ly, args.width, args.epochs) <= 0:
        parser.error('dimensions, width and epochs must be positive')
    args.device = torch.device(args.device)
    if args.device.type not in ('cpu', 'cuda'):
        parser.error('device must be cpu or cuda[:index]')
    if args.device.type == 'cuda' and (not torch.cuda.is_available()
                                       or (args.device.index or 0) >= torch.cuda.device_count()):
        parser.error('requested CUDA device is unavailable')
    return args


def main(argv=None):
    args = parse_args(argv)
    output, device = args.output, args.device
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    report = {
        'status': 'building_ED', 'recorded_at': datetime.now(timezone.utc).isoformat(),
        'command': sys.argv,
        'settings': {k: str(v) if isinstance(v, (Path, torch.device)) else v for k, v in vars(args).items()},
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
        'objective': 'full-sector normalized uniform log-MSE, exact D2-translation orbit moments',
        'selection': 'best post-update log-MSE including initialization; fixed epoch budget',
        'symmetry': 'D4-tied local filters; D2 and translations on rectangle', 'history': [],
    }

    def persist():
        temporary = output / 'report.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(output / 'report.json')

    persist()
    try:
        # Explicit bounds permit 3x4, while still rejecting the larger sectors.
        dataset = make_ed_dataset(args.Lx, args.Ly, max_dimension=1_000_000,
                                  max_sparse_bytes=8 * 1024**3, dtype=torch.float64)
        meta = dataset['metadata']
        print('ED ready', meta, flush=True)
        report.update(dataset=meta, ED_seconds=time.perf_counter() - started,
                      physics={key: meta[key] for key in ('t', 'U', 'mu', 'Nmax', 'Ntot', 'pbc_x', 'pbc_y')})
        indices, labels, counts = rectangular_orbits(dataset['basis'], args.Lx, args.Ly, meta['Nmax'])
        expected = EXPECTED_ORBITS.get((args.Lx, args.Ly))
        if expected is not None and meta['Nmax'] == 4 and meta['Ntot'] == args.Lx * args.Ly \
                and len(indices) != expected:
            raise RuntimeError(f'orbit count {len(indices)} disagrees with Burnside count {expected}')
        np.savez_compressed(output / 'ed-reference.npz',
                            basis=np.asarray(dataset['basis'], dtype=np.int8),
                            raw_ed_state=dataset['raw_ed_state'], energy=dataset['energy'],
                            representatives=indices, labels=labels, counts=counts)

        target = dataset['target_log'].numpy()
        target = target - 0.5 * np.logaddexp.reduce(2 * target)
        means = np.bincount(labels, weights=target) / counts
        variances = np.bincount(labels, weights=(target - means[labels])**2) / counts
        model = GroupEquivariantConv2DNet(args.Lx, args.Ly, meta['Nmax'], hidden_dim=args.width).to(device)
        x = dataset['x'][indices].to(device=device, dtype=torch.float32)
        moments = [torch.as_tensor(a, device=device, dtype=torch.float32) for a in (counts, means, variances)]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        def objective(logs):
            return orbit_log_mse(logs, *moments)

        with torch.no_grad():
            best_loss = objective(model(x)).item()
        best_epoch = 0
        best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        report.update(status='training', orbit_dimension=len(indices), parameter_count=parameter_count(model),
                      training_dtype='float32; TF32 disabled', initial_loss=best_loss)
        persist()
        for epoch in range(1, args.epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(x))
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('nonfinite gradient')
            optimizer.step()
            with torch.no_grad():
                post_loss = objective(model(x)).item()
            if not np.isfinite(post_loss):
                raise RuntimeError('nonfinite loss')
            if post_loss < best_loss:
                best_loss, best_epoch = post_loss, epoch
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 100 == 0 or epoch == args.epochs:
                report['history'].append({'epoch': epoch, 'log_mse': post_loss})
                persist()
                print('epoch', epoch, 'log_mse', post_loss, flush=True)

        model.load_state_dict(best)
        save_model(output / 'best.pt', model.double(), meta['Ntot'], meta['t'], meta['U'], meta['mu'])
        # Evaluate the reloaded saved checkpoint directly on the full basis,
        # not by representative expansion.
        model, _ = load_model(output / 'best.pt', device=device)
        with torch.no_grad():
            logs = torch.cat([model(block.to(device)).cpu() for block in dataset['x'].split(1024)])
            rep_logs = model(dataset['x'][indices].to(device)).cpu()
        discrepancy = (logs - rep_logs[labels]).abs().max().item()
        if discrepancy > 1e-9:
            raise RuntimeError('full-sector D2 expansion failed')
        psi = (logs - 0.5 * torch.logsumexp(2 * logs, dim=0)).exp().numpy()
        applied = dataset['H'] @ psi
        energy = float(psi @ applied)
        overlap = float(abs(dataset['raw_ed_state'] @ psi))
        report.update(status='completed', best_epoch=best_epoch, best_training_log_mse=best_loss,
                      amplitude_overlap_abs_unsquared=overlap, fidelity_squared=overlap**2,
                      energy=energy, energy_error_per_site=(energy - dataset['energy']) / model.n_sites,
                      variance=float(np.linalg.norm(applied - energy * psi)**2),
                      full_basis_vs_orbit_max_log_difference=discrepancy,
                      elapsed_seconds=time.perf_counter() - started)
        persist()
        print(json.dumps({k: v for k, v in report.items() if k != 'history'}), flush=True)
    except BaseException as error:
        report.update(status='failed_retained', error=f'{type(error).__name__}: {error}',
                      elapsed_seconds=time.perf_counter() - started)
        persist()
        raise


if __name__ == '__main__':
    main()
