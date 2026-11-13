"""Inference in process; established 2D workflows in isolated child interpreters."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from bosehubbard_nqs_inference import predict_values as _predict_values, predict_scalar as _predict_scalar, resolve_device


def load_model(path, device=None):
    models = importlib.import_module('pythonTest.bosehubbard2d.models')
    return models.load_model(str(Path(path).resolve()), device=resolve_device('cpu' if device is None else device))


def predict_values(model, configs, nmax=None, chunk_size=1024):
    return _predict_values(model, configs, nmax, chunk_size, preserve_dtype=True)


def predict_scalar(model, config, nmax=None):
    return _predict_scalar(model, config, nmax, preserve_dtype=True)


def run_workflow(workflow, options):
    if workflow not in ('train', 'train_ed', 'evaluate'):
        raise ValueError('unknown 2D workflow')
    argv = []
    for key, value in options.items():
        if value is None:
            continue
        flag = '--' + key.replace('_', '-')
        if key == 'pair':
            pairs = value if value and isinstance(value[0], (list, tuple)) else [value]
            for pair in pairs:
                if len(pair) != 2:
                    raise ValueError('pair requires (i,j) or a sequence of pairs')
                argv.extend([flag, *(str(v) for v in pair)])
            continue
        if isinstance(value, bool):
            if key in ('pbc', 'pbc_x', 'pbc_y'):
                argv.append(flag if value else '--no-' + key.replace('_', '-'))
            elif value:
                argv.append(flag)
            else:
                raise ValueError(f'{key}=false has no CLI form; omit the option')
        else:
            argv.append(flag)
            argv.extend(str(v) for v in (value if isinstance(value, (list, tuple)) else [value]))
    root = Path(__file__).resolve().parents[2]
    # Report IPC is separate from guarded checkpoint/report paths, and removed on failure.
    with tempfile.TemporaryDirectory(prefix='bhnqs-workflow-') as directory:
        report_path = Path(directory) / 'result.json'
        code = ('import importlib,json,sys; from pathlib import Path; '
                'sys.path.insert(0,str(Path.cwd()/"src"/"python")); '
                'from bosehubbard_nqs_inference import resolve_device; '
                'm=importlib.import_module(sys.argv[1]); args=sys.argv[3:]; '
                'resolve_device(args[args.index("--device")+1]) if "--device" in args else None; '
                'r=m.main(args); '
                'open(sys.argv[2],"w").write(json.dumps(r,allow_nan=False))')
        completed = subprocess.run([sys.executable, '-c', code,
            'pythonTest.bosehubbard2d.' + workflow, str(report_path), *argv],
            cwd=root, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(f'{workflow} failed (exit {completed.returncode}):\n'
                               + completed.stdout + completed.stderr)
        report = json.loads(report_path.read_text())
    settings = report['settings']
    paths = {'output': settings['output']}
    if workflow == 'evaluate':
        paths.update(checkpoint=report['checkpoint'],
                     raw_samples=str(root / 'pythonTest/2D' / report['raw_samples_path']))
    else:
        paths.update(checkpoint=settings['output'], sidecar=settings['sidecar'])
    return {'report': report, 'stdout': completed.stdout, 'stderr': completed.stderr,
            'paths': paths}
