#!/usr/bin/env python3
"""
Train NQS on the full basis (all particle sectors) in logpsi space.

Target:
- if sum_i n_i == train_Ntot: y = log(|psi_ED| + eps)
- else:                       y = log_zero  (default -10.0)

Training loss:
- globally normalized log-space MSE across ALL states (all sectors),
  with optional in/out-sector weighting.
"""

import argparse
import numpy as np
import torch

from BoseHubbardED import exact_ground_state, basis_to_one_hot
from NeuralNetworkClass import FCNet, JastrowNet, Conv1DNet, RBMNet
from utils import pick_best_device


ARCH_BUILDERS = {
    "FCNet": lambda L, Ch, hd: FCNet(n_sites=L, channels=Ch, hidden_dim=hd),
    "JastrowNet": lambda L, Ch, hd: JastrowNet(n_sites=L, channels=Ch),
    "Conv1DNet": lambda L, Ch, hd: Conv1DNet(n_sites=L, channels=Ch, hidden_dim=hd),
    "RBMNet": lambda L, Ch, hd: RBMNet(n_sites=L, channels=Ch, n_hidden=hd),
}


def make_sector_log_target_map(L, train_Ntot, Nmax, t, U, mu=0.0, pbc=True, eps=1e-12):
    energy, psi0, basis, _ = exact_ground_state(
        L=L, Ntot=train_Ntot, Nmax=Nmax, t=t, U=U, mu=mu, pbc=pbc
    )
    target_map = {state: float(np.log(np.abs(amp) + eps)) for state, amp in zip(basis, psi0)}
    return target_map, energy, basis, psi0


def indices_to_states(indices, L, base):
    idx = indices.astype(np.int64, copy=True)
    out = np.empty((len(indices), L), dtype=np.int16)
    for site in range(L - 1, -1, -1):
        out[:, site] = idx % base
        idx //= base
    return out


def build_targets_log(states, target_map, train_Ntot, log_zero=-10.0):
    y = np.full(states.shape[0], fill_value=log_zero, dtype=np.float32)
    ntot = states.sum(axis=1)
    in_sector = ntot == train_Ntot

    idx_in = np.where(in_sector)[0]
    for k in idx_in:
        key = tuple(int(v) for v in states[k])
        y[k] = target_map.get(key, log_zero)

    return y, in_sector


def normalize_logpsi(logpsi):
    return logpsi - 0.5 * torch.logsumexp(2.0 * logpsi, dim=0)


def build_full_space_tensors(L, Nmax, train_Ntot, target_map, log_zero, device):
    base = Nmax + 1
    total_states = base ** L

    idx_all = np.arange(total_states, dtype=np.int64)
    states_all = indices_to_states(idx_all, L=L, base=base)

    x_np = basis_to_one_hot(states_all, n_sites=L, nmax=Nmax)
    y_np, in_sector_np = build_targets_log(
        states=states_all,
        target_map=target_map,
        train_Ntot=train_Ntot,
        log_zero=log_zero,
    )

    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    in_sector_mask = torch.as_tensor(in_sector_np, dtype=torch.bool, device=device)
    y_norm = normalize_logpsi(y)

    return states_all, x, y, y_norm, in_sector_mask


def normalized_weighted_mse_fullspace(pred_log, target_norm, in_sector_mask, inside_weight=1.0, outside_weight=0.2):
    pred_norm = normalize_logpsi(pred_log)
    sq = (pred_norm - target_norm) ** 2
    w = torch.where(
        in_sector_mask,
        torch.full_like(sq, inside_weight),
        torch.full_like(sq, outside_weight),
    )
    return torch.mean(w * sq)


def train_model_on_full_basis_log(
    model,
    x,
    y_norm,
    in_sector_mask,
    lr=1e-3,
    epochs=300,
    inside_weight=1.0,
    outside_weight=0.2,
    print_every=10,
):
    model = model.to(x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        pred_log = model(x).reshape(-1)
        loss = normalized_weighted_mse_fullspace(
            pred_log=pred_log,
            target_norm=y_norm,
            in_sector_mask=in_sector_mask,
            inside_weight=inside_weight,
            outside_weight=outside_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch % print_every == 0) or (epoch == epochs - 1):
            print(f"epoch={epoch:4d} loss={loss.item():.6e}")

    return model


@torch.no_grad()
def evaluate_fullspace_log_overlap(model, x, y):
    model.eval()
    pred_log = model(x).reshape(-1)

    pred_norm = normalize_logpsi(pred_log)
    true_norm = normalize_logpsi(y)

    psi_pred = torch.exp(pred_norm)
    psi_true = torch.exp(true_norm)

    overlap = torch.abs(torch.dot(psi_pred, psi_true)).item()
    mse = torch.mean((pred_norm - true_norm) ** 2).item()

    print(f"[full-space eval] normalized overlap = {overlap:.8e}")
    print(f"[full-space eval] normalized MSE     = {mse:.8e}")
    return overlap, mse


@torch.no_grad()
def evaluate_in_sector_overlap(model, L, Nmax, basis_sector, psi0_sector, device="cpu", eps=1e-12):
    model.eval()
    x_np = basis_to_one_hot(basis_sector, n_sites=L, nmax=Nmax)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    pred_log = model(x).reshape(-1)

    true_log = torch.as_tensor(np.log(np.abs(psi0_sector) + eps), dtype=torch.float32, device=device).reshape(-1)

    pred_norm = normalize_logpsi(pred_log)
    true_norm = normalize_logpsi(true_log)

    psi_pred = torch.exp(pred_norm)
    psi_true = torch.exp(true_norm)

    overlap = torch.abs(torch.dot(psi_pred, psi_true)).item()
    mse = torch.mean((pred_norm - true_norm) ** 2).item()

    print(f"[sector-only eval] normalized overlap = {overlap:.8e}")
    print(f"[sector-only eval] normalized MSE     = {mse:.8e}")
    return overlap, mse


@torch.no_grad()
def full_outside_log_check(model, x, states_all, train_Ntot, log_zero=-10.0):
    model.eval()
    pred = model(x).reshape(-1).detach().cpu().numpy()

    ntot = states_all.sum(axis=1)
    outside = ntot != train_Ntot

    if np.any(outside):
        outside_vals = pred[outside]
        print(f"[outside/full] mean(logpsi) = {outside_vals.mean():.6f}")
        print(f"[outside/full] std(logpsi)  = {outside_vals.std():.6f}")
        print(
            f"[outside/full] mean error to log_zero({log_zero}) = "
            f"{np.mean(np.abs(outside_vals - log_zero)):.6e}"
        )
    else:
        print("[outside/full] no outside states found")


def main():
    parser = argparse.ArgumentParser(
        description="Full-basis logpsi training with global normalization over all sectors."
    )
    parser.add_argument("--arch", type=str, default="FCNet", choices=["FCNet", "JastrowNet", "Conv1DNet", "RBMNet"])
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--Nmax", type=int, default=4)
    parser.add_argument("--train_Ntot", type=int, default=4)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--U", type=float, default=2.0)
    parser.add_argument("--mu", type=float, default=0.0)

    parser.add_argument("--pbc", dest="pbc", action="store_true")
    parser.add_argument("--no-pbc", dest="pbc", action="store_false")
    parser.set_defaults(pbc=True)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--inside_weight", type=float, default=1.0)
    parser.add_argument("--outside_weight", type=float, default=0.2)
    parser.add_argument("--log_zero", type=float, default=-3.0)
    parser.add_argument("--save_model", type=str, default="")

    args = parser.parse_args()

    device = pick_best_device()
    print(f"Using device: {device}")

    target_map, energy, basis_sector, psi0_sector = make_sector_log_target_map(
        L=args.L,
        train_Ntot=args.train_Ntot,
        Nmax=args.Nmax,
        t=args.t,
        U=args.U,
        mu=args.mu,
        pbc=args.pbc,
        eps=1e-12,
    )
    print(f"ED energy (N={args.train_Ntot}): {energy:.8f}")
    print(f"Sector basis size: {len(basis_sector)}")

    channels = args.Nmax + 1
    model = ARCH_BUILDERS[args.arch](args.L, channels, args.hidden_dim)

    states_all, x, y, y_norm, in_sector_mask = build_full_space_tensors(
        L=args.L,
        Nmax=args.Nmax,
        train_Ntot=args.train_Ntot,
        target_map=target_map,
        log_zero=args.log_zero,
        device=device,
    )
    print(f"Full-space training states: {states_all.shape[0]}")

    model = train_model_on_full_basis_log(
        model=model,
        x=x,
        y_norm=y_norm,
        in_sector_mask=in_sector_mask,
        lr=args.lr,
        epochs=args.epochs,
        inside_weight=args.inside_weight,
        outside_weight=args.outside_weight,
        print_every=max(1, args.epochs // 10),
    )

    evaluate_fullspace_log_overlap(model=model, x=x, y=y)
    evaluate_in_sector_overlap(
        model=model,
        L=args.L,
        Nmax=args.Nmax,
        basis_sector=basis_sector,
        psi0_sector=psi0_sector,
        device=device,
    )
    full_outside_log_check(
        model=model,
        x=x,
        states_all=states_all,
        train_Ntot=args.train_Ntot,
        log_zero=args.log_zero,
    )

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"Saved model to {args.save_model}")


if __name__ == "__main__":
    main()