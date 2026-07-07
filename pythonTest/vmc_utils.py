import numpy as np
import random
import torch
import gc
from typing import Optional, List, Tuple

from torch.func import vmap, jacrev, functional_call

# State convention for Bose-Hubbard:
# state = (n0, n1, ..., n_{L-1}), with
#   0 <= ni <= Nmax
#   sum_i ni = Ntot
# Network convention:
#   psi(x) returns log|Psi(x)| (real scalar per sample).

def state_to_onehot(state: Tuple[int, ...], L: int, Nmax: int) -> np.ndarray:
    x = np.zeros((L, Nmax + 1), dtype=np.float32)
    for i, n in enumerate(state):
        x[i, n] = 1.0
    return x

def generate_initial_state(L: int, Ntot: int, Nmax: int) -> Tuple[int, ...]:
    if Ntot < 0:
        raise ValueError("Ntot must be >= 0")
    if Ntot > L * Nmax:
        raise ValueError(f"Ntot={Ntot} exceeds L*Nmax={L*Nmax}")

    occ = [0] * L
    remaining = Ntot
    sites = list(range(L))
    random.shuffle(sites)

    for i in sites:
        if remaining == 0:
            break
        put = random.randint(0, min(Nmax, remaining))
        occ[i] = put
        remaining -= put

    while remaining > 0:
        i = random.randrange(L)
        if occ[i] < Nmax:
            occ[i] += 1
            remaining -= 1

    return tuple(occ)


def _bonds_1d(L: int, pbc: bool = True) -> List[Tuple[int, int]]:
    bonds = [(i, i + 1) for i in range(L - 1)]
    if pbc and L > 1:
        bonds.append((L - 1, 0))
    return bonds


def GlobalSampler(state: Tuple[int, ...], L: int, Nmax: int, pbc: bool = True) -> Tuple[int, ...]:
    """
    Symmetric one-particle proposal for Bose-Hubbard:
    1) Pick an undirected bond uniformly.
    2) Pick direction uniformly (src->dst or dst->src).
    3) If move is valid (n_src>0 and n_dst<Nmax), execute; else self-propose.

    This guarantees q(s'|s) = q(s|s') for each valid one-hop move,
    so standard Metropolis acceptance min(1, |psi(s')|^2 / |psi(s)|^2) is valid.
    """
    occ = list(state)
    bonds = _bonds_1d(L, pbc=pbc)
    if len(bonds) == 0:
        return state

    a, b = random.choice(bonds)
    if random.random() < 0.5:
        src, dst = a, b
    else:
        src, dst = b, a

    if occ[src] > 0 and occ[dst] < Nmax:
        occ[src] -= 1
        occ[dst] += 1
        return tuple(occ)

    return state


def local_energy_on_the_fly(
    state: Tuple[int, ...],
    psi,
    L: int,
    Nmax: int,
    t: float,
    U: float,
    mu: float = 0.0,
    pbc: bool = True,
    device: str = "cpu",
):
    """
    Bose-Hubbard local energy:
        E_loc(s) = H_ss + sum_{s'!=s} H_{s',s} * exp(logpsi(s') - logpsi(s))
    with:
        H = -t sum_<ij> (b_i^dag b_j + h.c.) + sum_i [U n_i(n_i-1) - mu n_i]

    This matches the matrix elements used in build_bose_hubbard_hamiltonian in BoseHubbardED.py.
    """
    s_onehot = torch.tensor(state_to_onehot(state, L, Nmax), dtype=torch.float32, device=device).unsqueeze(0)
    logpsi_s = psi(s_onehot).reshape(())
    dtype = logpsi_s.dtype

    occ = list(state)

    diag = 0.0
    for n in occ:
        diag += U * n * (n - 1) - mu * n

    E_loc = torch.tensor(diag, dtype=dtype, device=device)

    for i, j in _bonds_1d(L, pbc=pbc):
        n_i = occ[i]
        n_j = occ[j]

        # b_i^dag b_j : j -> i
        if n_j > 0 and n_i < Nmax:
            new_occ = occ.copy()
            new_occ[i] += 1
            new_occ[j] -= 1
            sp_onehot = torch.tensor(
                state_to_onehot(tuple(new_occ), L, Nmax),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            logpsi_sp = psi(sp_onehot).reshape(())
            ratio = torch.exp(logpsi_sp - logpsi_s)
            coeff = -t * np.sqrt((n_i + 1) * n_j)
            E_loc = E_loc + torch.tensor(coeff, dtype=dtype, device=device) * ratio

        # b_j^dag b_i : i -> j
        if n_i > 0 and n_j < Nmax:
            new_occ = occ.copy()
            new_occ[i] -= 1
            new_occ[j] += 1
            sp_onehot = torch.tensor(
                state_to_onehot(tuple(new_occ), L, Nmax),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            logpsi_sp = psi(sp_onehot).reshape(())
            ratio = torch.exp(logpsi_sp - logpsi_s)
            coeff = -t * np.sqrt((n_j + 1) * n_i)
            E_loc = E_loc + torch.tensor(coeff, dtype=dtype, device=device) * ratio

    return E_loc


def compute_log_psi_jacobian_vmap(psi, samples_onehot, device):
    """
    Jacobian of log|Psi| w.r.t. parameters, assuming model output is already log|Psi|.
    """
    x = torch.from_numpy(samples_onehot).float().to(device)
    params_dict = {k: v for k, v in psi.named_parameters() if v.requires_grad}
    param_names = list(params_dict.keys())

    def log_psi_single(params, x_single):
        return functional_call(psi, params, (x_single.unsqueeze(0),)).reshape(())

    jac_fn = vmap(jacrev(log_psi_single), in_dims=(None, 0))
    jac_dict = jac_fn(params_dict, x)

    rows = []
    for name in param_names:
        rows.append(jac_dict[name].reshape(x.shape[0], -1))
    return torch.cat(rows, dim=1)


def sr_gradient(J, E_loc, tau=1e-3):
    J64 = J.detach().cpu().double()
    E64 = E_loc.detach().cpu().double()
    S = J64.T @ J64
    S.diagonal().add_(tau)
    f = J64.T @ E64
    delta = torch.linalg.solve(S, f.unsqueeze(-1)).squeeze(-1)
    return delta.float().to(J.device)


def sr_update(psi, sampled_states, E_tensor, L, Nmax, lr, tau, device):
    s_np = np.stack([state_to_onehot(s, L, Nmax) for s in sampled_states])
    J = compute_log_psi_jacobian_vmap(psi, s_np, device)
    E_loc = E_tensor.detach() - E_tensor.mean().detach()
    delta_theta = sr_gradient(J, E_loc, tau=tau)

    max_update = 0.0
    with torch.no_grad():
        offset = 0
        for p in psi.parameters():
            if p.requires_grad:
                numel = p.numel()
                update = lr * delta_theta[offset:offset + numel].reshape(p.shape)
                p.data -= update
                offset += numel
                max_update = max(max_update, update.abs().max().item())
    #print(f"learning rate: {lr}, tau: {tau}, max parameter update: {max_update:.4e}")


def Obtain_Sampling_batch(
    psi,
    L,
    Nmax,
    initial_states,
    n_steps,
    burnin=False,
    print_rate=False,
    Sampler=GlobalSampler,
    t=1.0,
    U=2.0,
    mu=0.0,
    pbc=True,
    device="cpu",
):
    """
    Metropolis sampling on |Psi|^2 where network output is log|Psi|.
    Returns:
        LogPsis_t: sampled log|Psi| values (flattened over walkers and steps)
        Elocs_t: local energies (after burnin)
        states: final walker states
        Sampled_states: visited states (flattened)
    """
    n_walkers = len(initial_states)
    states = initial_states.copy()
    LogPsis = []
    Elocs = []
    Sampled_states = []
    accept_count = 0

    for _ in range(n_steps):
        proposed = [Sampler(s, L=L, Nmax=Nmax, pbc=pbc) for s in states]

        s_np = np.stack([state_to_onehot(s, L, Nmax) for s in states])
        new_np = np.stack([state_to_onehot(s, L, Nmax) for s in proposed])
        s_batch = torch.from_numpy(s_np).float().to(device)
        new_batch = torch.from_numpy(new_np).float().to(device)

        logpsi_vals = psi(s_batch).reshape(-1)
        logpsi_new_vals = psi(new_batch).reshape(-1)

        log_ratio_prob = 2.0 * (logpsi_new_vals - logpsi_vals)
        accept_prob = torch.exp(torch.clamp(log_ratio_prob, max=60.0)).clamp(max=1.0)

        rand = torch.rand(n_walkers, device=device)
        accept = (rand < accept_prob).cpu().numpy()

        for i in range(n_walkers):
            if accept[i]:
                states[i] = proposed[i]
                LogPsis.append(logpsi_new_vals[i])
                accept_count += 1
            else:
                LogPsis.append(logpsi_vals[i])
            Sampled_states.append(states[i])

        if burnin:
            continue

        for s in states:
            E_loc = local_energy_on_the_fly(
                s, psi, L=L, Nmax=Nmax, t=t, U=U, mu=mu, pbc=pbc, device=device
            )
            Elocs.append(E_loc)

    if print_rate:
        print(f"Acceptance rate (batched): {accept_count / (n_steps * n_walkers):.4f}")

    LogPsis_t = torch.stack(LogPsis).squeeze() if len(LogPsis) > 0 else torch.tensor([], device=device)
    Elocs_t = torch.stack(Elocs).squeeze() if len(Elocs) > 0 else torch.tensor([], device=device)

    return LogPsis_t, Elocs_t, states, Sampled_states


def sr_update_optimizer(psi, sampled_states, E_tensor, L, Nmax, optimizer, tau, device,
                        adaptive_lr=False, lr=1e-3):
    s_np = np.stack([state_to_onehot(s, L, Nmax) for s in sampled_states])
    J = compute_log_psi_jacobian_vmap(psi, s_np, device)
    E_loc = E_tensor.detach() - E_tensor.mean().detach()
    delta_theta = sr_gradient(J, E_loc, tau=tau)
    print(f"tau: {tau}, max SR gradient: {delta_theta.abs().max():.4e}")

    optimizer.zero_grad()

    offset = 0
    for p in psi.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.grad = delta_theta[offset:offset + numel].reshape(p.shape).clone()
            offset += numel

    if adaptive_lr:
        scale = delta_theta.abs().max().item() + 1e-5
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr / scale
            print(f"Adaptive learning rate set to: {param_group['lr']:.4e}")

    optimizer.step()


def cleanup_memory(free_vars: Optional[list] = None, optimizer: Optional[torch.optim.Optimizer] = None):
    if free_vars:
        for v in free_vars:
            try:
                if hasattr(v, "clear"):
                    v.clear()
                else:
                    del v
            except Exception:
                pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if optimizer is not None:
        for k in list(optimizer.state.keys()):
            state = optimizer.state[k]
            for sk in list(state.keys()):
                if torch.is_tensor(state[sk]):
                    state.pop(sk, None)

