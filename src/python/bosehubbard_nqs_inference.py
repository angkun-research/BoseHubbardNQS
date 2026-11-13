"""Shared Torch inference transport; occupations arrive as sites x walkers."""
from numbers import Integral
import re
import numpy as np
import torch
from torch.nn import functional as F


def resolve_device(device):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not isinstance(device, str) or re.fullmatch(r'cpu|cuda(?::[0-9]+)?', device) is None:
        raise ValueError('device must be auto, cpu, or cuda[:index]')
    if device.startswith('cuda'):
        # Parse the original integer: torch.device internally wraps huge indices.
        index = int(device.split(':')[1]) if ':' in device else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError('requested CUDA device is unavailable')
    return torch.device(device)


@torch.no_grad()
def predict_values(model, configs, nmax=None, chunk_size=1024, *, preserve_dtype=False):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    cutoff = getattr(model, 'nmax', getattr(model, 'Nmax', None))
    nmax = cutoff if nmax is None else nmax
    if isinstance(nmax, bool) or not isinstance(nmax, Integral) or nmax < 0:
        raise ValueError('nmax must be a nonnegative integer')
    if cutoff is not None and nmax != cutoff:
        raise ValueError('nmax disagrees with model cutoff')
    array = np.asarray(configs)
    if array.ndim != 2 or array.shape[0] != model.n_sites:
        raise ValueError(f'configs must have shape ({model.n_sites}, walkers)')
    if array.dtype.kind not in 'iu':
        raise ValueError('occupations must be integers')
    if array.size and (array.min() < 0 or array.max() > nmax):
        raise ValueError(f'occupation values must be within [0, {nmax}]')
    parameter = next(model.parameters())
    # Copy views/negative strides once, then create all one-hot features on Torch device.
    states = torch.as_tensor(np.array(array.T, dtype=np.int64, order='C', copy=True),
                             device=parameter.device)
    dtype = parameter.dtype if preserve_dtype else torch.float32
    if states.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    result = torch.cat([model(F.one_hot(part, nmax + 1).to(dtype=dtype)).reshape(-1)
                        for part in states.split(int(chunk_size))])
    # One bulk device-to-host copy, including CUDA; no per-walker scalar extraction.
    return result.to(device='cpu', dtype=torch.float64).numpy()


def predict_scalar(model, config, nmax=None, *, preserve_dtype=False):
    array = np.asarray(config)
    if array.ndim != 1:
        raise ValueError('configuration must be a vector')
    return float(predict_values(model, array[:, None], nmax, 1,
                                preserve_dtype=preserve_dtype)[0])
