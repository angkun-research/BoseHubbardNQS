"""Stream raw NQS weights across cutoff sectors without Hamiltonians/training.

Off-sector mass concerns the unprojected extension of a fixed-N learned state.
It is not projected canonical-VMC error or evidence of wasted capacity. Sector
weights include degeneracies; cross-sector states are orthogonal, not transitions.
"""
import itertools
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from lattice import _integer
from exact_diagonalization import count_basis_states


DEFAULT_MAX_STATES = 2_000_000


def _geometry(n_sites, Nmax, batch_size, max_states):
    n_sites = _integer(n_sites, 'n_sites', 1)
    Nmax = _integer(Nmax, 'Nmax')
    batch_size = _integer(batch_size, 'batch_size', 1)
    max_states = _integer(max_states, 'max_states', 1)
    if n_sites > 64 or batch_size > 16384:
        raise ValueError('streaming analyzer supports at most64sites and batch16384')
    return n_sites, Nmax, batch_size, max_states


def _sector_tuples(n_sites, Nmax, N):
    state = [0] * n_sites
    def visit(site, remaining):
        if site == n_sites:
            if remaining == 0:
                yield tuple(state)
            return
        lower = max(0, remaining-(n_sites-site-1)*Nmax)
        for occupation in range(lower, min(Nmax, remaining)+1):
            state[site] = occupation
            yield from visit(site+1, remaining-occupation)
    yield from visit(0, N)


def state_batches(n_sites, Nmax, sectors=None, batch_size=2048,
                  max_states=DEFAULT_MAX_STATES):
    """Return (iterator, exact sector dimensions), validated before iteration.

    Full basis uses vectorized base counting, last physical site fastest. Nearby
    sectors use bounded lexicographic DFS. At most one state/one-hot batch lives
    in memory, apart from model inference workspace. No full basis is retained.
    """
    n_sites, Nmax, batch_size, max_states = _geometry(n_sites, Nmax, batch_size, max_states)
    if sectors is None:
        total = (Nmax+1)**n_sites
        if total > max_states or total > np.iinfo(np.int64).max:
            raise ValueError('full truncated basis exceeds max_states/int64 guard')
        numbers = list(range(n_sites*Nmax+1))
    else:
        numbers = [_integer(n, 'sector') for n in sectors]
        if not numbers or len(numbers) != len(set(numbers)) or max(numbers) > n_sites*Nmax:
            raise ValueError('sectors must be distinct nonempty physical particle numbers')
    dimensions = {n: count_basis_states(n_sites, n, Nmax) for n in numbers}
    total = sum(dimensions.values())
    if total > max_states:
        raise ValueError('enumerated sector window exceeds max_states guard')
    def batches():
        if sectors is None:
            strides = np.array([(Nmax+1)**i for i in range(n_sites-1, -1, -1)], dtype=np.int64)
            for start in range(0, total, batch_size):
                index = np.arange(start, min(total, start+batch_size), dtype=np.int64)
                yield (index[:, None]//strides[None, :]) % (Nmax+1)
        else:
            for number in numbers:
                stream = _sector_tuples(n_sites, Nmax, number)
                while True:
                    rows = list(itertools.islice(stream, batch_size))
                    if not rows:
                        break
                    yield np.asarray(rows, dtype=np.int64)
    return batches(), dimensions


@torch.no_grad()
def evaluate_logs(model, occupations):
    """Finite real logs on a supplied integer batch, without sector restriction."""
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device('cpu')
    dtype = parameter.dtype if parameter is not None else torch.float64
    states = torch.as_tensor(occupations, device=device)
    if states.is_floating_point() or states.is_complex() or states.dtype == torch.bool:
        raise ValueError('integer occupations required')
    states = states.long()
    if states.ndim != 2 or states.shape[1] != model.n_sites:
        raise ValueError('occupations must have shape(batch,n_sites)')
    if torch.any(states < 0) or torch.any(states > model.Nmax):
        raise ValueError('occupations exceed cutoff')
    output = model(F.one_hot(states, model.Nmax+1).to(dtype=dtype))
    if output.shape != (len(states),) or output.is_complex() or not torch.isfinite(output).all():
        raise ValueError('model must return finite real logs shaped(batch,)')
    return output.double().cpu().numpy()


def _logsum(values):
    return float(np.logaddexp.reduce(np.asarray(values, dtype=np.float64)))


def summarize_sector_weights(logZ, dimensions, N0, window):
    """Separate window-conditional probabilities from genuine full-cutoff mass."""
    N0 = _integer(N0, 'N0')
    if N0 not in logZ or not logZ:
        raise ValueError('reference sector must be evaluated')
    numbers = sorted(logZ)
    if set(logZ) != set(dimensions) or any(dimensions[n] <= 0 for n in numbers):
        raise ValueError('positive sector dimensions must match weights')
    if not all(math.isfinite(logZ[n]) for n in numbers):
        raise ValueError('raw sector log weights must be finite')
    window = list(window)
    if not window or len(window) != len(set(window)) or any(n not in logZ for n in window) or N0 not in window:
        raise ValueError('window must contain distinct evaluated sectors including N0')
    log_window = _logsum([logZ[n] for n in window])
    total_log = _logsum([logZ[n] for n in numbers])
    rows = [{'N':n, 'dimension':int(dimensions[n]), 'logZ_N':float(logZ[n]),
             'logZ_N_minus_logZ_N0':float(logZ[n]-logZ[N0]),
             'log_mean_squared_raw_amplitude_per_configuration':float(logZ[n]-math.log(dimensions[n])),
             'window_conditional_probability':float(math.exp(logZ[n]-log_window)) if n in window else None}
            for n in numbers]
    return {'N0':N0, 'enumerated_state_count':sum(dimensions.values()), 'sectors':rows,
            'window_sectors':window, 'logZ_window':log_window,
            'window_conditional_P_N0':float(math.exp(logZ[N0]-log_window)),
            'window_conditional_outside_N0':float(-math.expm1(logZ[N0]-log_window)),
            'logZ_enumerated':total_log,
            'mean_weight_units':'Natural logarithm of Z_N/d_N; absolute raw weights depend on additive log-amplitude gauge.'}


def _add_full_normalization(result):
    total_log = result['logZ_enumerated']
    N0 = result['N0']
    off = []
    for row in result['sectors']:
        row['P_full'] = float(math.exp(row['logZ_N']-total_log))
        row['logP_full'] = row['logZ_N']-total_log
        if row['N'] != N0:
            off.append(row['logZ_N'])
    canonical = next(row for row in result['sectors'] if row['N'] == N0)
    result.update(normalization_scope='all_truncated_Fock_states', logZ_full=total_log,
                  P_full_N0=canonical['P_full'], logP_full_N0=canonical['logP_full'],
                  leakage_full=float(math.exp(_logsum(off)-total_log)) if off else 0.,
                  window_mass_in_full=float(math.exp(result['logZ_window']-total_log)))
    return result


@torch.no_grad()
def analyze_sectors(model, N0, sectors=None, *, window=None, batch_size=2048,
                    max_states=DEFAULT_MAX_STATES):
    """Exact enumeration in floating-point arithmetic, no H/ED/optimizer calls."""
    batches, dimensions = state_batches(model.n_sites, model.Nmax, sectors, batch_size, max_states)
    N0 = _integer(N0, 'N0')
    if N0 not in dimensions:
        raise ValueError('reference sector must be evaluated')
    window = sorted(dimensions) if window is None else list(window)
    if N0 not in window or len(window) != len(set(window)) or not set(window) <= set(dimensions):
        raise ValueError('window must contain distinct evaluated sectors including N0')
    model.eval()
    logZ = {n:-math.inf for n in dimensions}
    counts = {n:0 for n in dimensions}
    started = time.perf_counter()
    for states in batches:
        numbers = states.sum(axis=1)
        weights = 2*evaluate_logs(model, states)
        if not np.isfinite(weights).all():
            raise ValueError('squared log amplitudes overflow')
        for n in np.unique(numbers):
            rows = numbers == n
            logZ[int(n)] = float(np.logaddexp(logZ[int(n)], _logsum(weights[rows])))
            counts[int(n)] += int(rows.sum())
    if counts != dimensions:
        raise RuntimeError('streamed counts differ from independent exact bounded counts')
    result = summarize_sector_weights(logZ, dimensions, N0, window)
    parameter = next(model.parameters(), None)
    result.update(wall_seconds=time.perf_counter()-started, batch_size=batch_size,
                  forward_dtype=str(parameter.dtype) if parameter is not None else 'torch.float64',
                  forward_device=str(parameter.device) if parameter is not None else 'cpu',
                  accumulation_dtype='float64', no_H_or_ED=True,
                  numerical_scope='Every configuration enumerated, finite-precision model forwards/logsumexp; floating underflow can round tiny probabilities to0, log probabilities remain available.')
    if set(dimensions) == set(range(model.n_sites*model.Nmax+1)):
        _add_full_normalization(result)
    else:
        result['normalization_scope'] = 'conditional_on_window_only_NOT_global'
    return result


def projected_sector_probabilities(sectors, N0):
    """Exact delta projection, not a finite negative sentinel in log space."""
    sectors = list(sectors)
    if N0 not in sectors:
        raise ValueError('projection reference sector is absent')
    return {n:float(n == N0) for n in sectors}


def apply_number_gauge(result, a):
    """Analytic f->f+a(N-N0), with no model mutation or new forward pass."""
    if isinstance(a, bool) or not math.isfinite(a):
        raise ValueError('gauge a must be finite real')
    N0 = result['N0']
    logZ = {row['N']:row['logZ_N']+2*a*(row['N']-N0) for row in result['sectors']}
    dimensions = {row['N']:row['dimension'] for row in result['sectors']}
    changed = summarize_sector_weights(logZ, dimensions, N0, result['window_sectors'])
    if result['normalization_scope'] == 'all_truncated_Fock_states':
        _add_full_normalization(changed)
    else:
        changed['normalization_scope'] = 'conditional_on_window_only_NOT_global'
    return {'a':a, 'canonical_log_amplitude_change':0., 'canonical_state_identical':True, **changed}


def full_overlap(F_N0, P_full_N0):
    """Absolute unsquared overlap against the zero-embedded canonical target."""
    if not (math.isfinite(F_N0) and math.isfinite(P_full_N0)
            and 0 <= F_N0 <= 1+1e-12 and 0 <= P_full_N0 <= 1+1e-12):
        raise ValueError('overlap/probability must be finite within[0,1]')
    return math.sqrt(min(1., P_full_N0))*min(1., F_N0)
