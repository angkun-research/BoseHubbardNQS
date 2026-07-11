#!/usr/bin/env python3
import argparse
import numpy as np
import torch

from BoseHubbardED import exact_ground_state
from NeuralNetworkClass import FCNet, JastrowNet, Conv1DNet, RBMNet
from utils import pick_best_device
from vmc_utils import (
    generate_initial_state,
    Obtain_Sampling_batch,
    sr_update,
    state_to_onehot,
)

ARCH_BUILDERS = {
    "FCNet": lambda L, Ch, hd: FCNet(n_sites=L, channels=Ch, hidden_dim=hd),
    "JastrowNet": lambda L, Ch, hd: JastrowNet(n_sites=L, channels=Ch),
    "Conv1DNet": lambda L, Ch, hd: Conv1DNet(n_sites=L, channels=Ch, hidden_dim=hd),
    "RBMNet": lambda L, Ch, hd: RBMNet(n_sites=L, channels=Ch, n_hidden=hd),
}

def native_vmc_update(model, sampled_states, E_tensor, L, Nmax, optimizer, device):
    """
    Native VMC gradient step for real log-amplitude ansatz:
    grad E = 2 * < (E_loc - <E_loc>) * grad log|psi| >
    """
    s_np = np.stack([state_to_onehot(s, L, Nmax) for s in sampled_states])
    s_batch = torch.from_numpy(s_np).float().to(device)

    logpsi = model(s_batch).reshape(-1)
    E_centered = (E_tensor - E_tensor.mean()).detach()

    loss = 2.0 * torch.mean(E_centered * logpsi)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def main():
    parser = argparse.ArgumentParser(description="VMC for 1D Bose-Hubbard with SR")
    parser.add_argument("--arch", type=str, default="FCNet", choices=list(ARCH_BUILDERS.keys()))
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--Nmax", type=int, default=8)
    parser.add_argument("--Ntot", type=int, default=8)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--U", type=float, default=2.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--pbc", action="store_true", default=True)
    parser.add_argument("--no-pbc", dest="pbc", action="store_false")

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_walkers", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--burnin_steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--update_method", type=str, default="sr", choices=["sr", "native"])
    parser.add_argument("--native_opt", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--native_lr", type=float, default=None)
    args = parser.parse_args()

    device = pick_best_device()
    print(f"Using device: {device}")

    model = ARCH_BUILDERS[args.arch](args.L, args.Nmax + 1, args.hidden_dim).to(device)

    native_optimizer = None
    if args.update_method == "native":
        native_lr = args.native_lr if args.native_lr is not None else args.lr
        if args.native_opt == "adam":
            native_optimizer = torch.optim.Adam(model.parameters(), lr=native_lr)
        else:
            native_optimizer = torch.optim.SGD(model.parameters(), lr=native_lr, momentum=0.9)
        print(f"Using native optimizer: {args.native_opt}, lr={native_lr}")
    else:
        print(f"Using SR optimizer: lr={args.lr}, tau={args.tau}")

    # ED reference
    ed_energy, _, basis, _ = exact_ground_state(
        L=args.L,
        Ntot=args.Ntot,
        Nmax=args.Nmax,
        t=args.t,
        U=args.U,
        mu=args.mu,
        pbc=args.pbc,
    )
    print(f"ED ground-state energy: {ed_energy:.10f} (dim={len(basis)})")

    # Initialize walkers
    states = [generate_initial_state(args.L, args.Ntot, args.Nmax) for _ in range(args.n_walkers)]

    # Before VMC burn-in
    _, _, states, _ = Obtain_Sampling_batch(
        psi=model,
        L=args.L,
        Nmax=args.Nmax,
        initial_states=states,
        n_steps=args.burnin_steps,
        burnin=True,
        print_rate=False,
        t=args.t,
        U=args.U,
        mu=args.mu,
        pbc=args.pbc,
        device=device,
    )

    for epoch in range(1, args.epochs + 1):
        # Burn-in
        _, _, states, _ = Obtain_Sampling_batch(
            psi=model,
            L=args.L,
            Nmax=args.Nmax,
            initial_states=states,
            n_steps=args.burnin_steps,
            burnin=True,
            print_rate=False,
            t=args.t,
            U=args.U,
            mu=args.mu,
            pbc=args.pbc,
            device=device,
        )

        # Production sampling
        _, Elocs, states, sampled_states = Obtain_Sampling_batch(
            psi=model,
            L=args.L,
            Nmax=args.Nmax,
            initial_states=states,
            n_steps=args.n_steps,
            burnin=False,
            print_rate=(epoch % args.eval_every == 0),
            t=args.t,
            U=args.U,
            mu=args.mu,
            pbc=args.pbc,
            device=device,
        )

        # Energy estimate
        Elocs = Elocs.reshape(-1)
        E_mean = Elocs.mean()
        E_std = Elocs.std(unbiased=False)

        if args.update_method == "sr":
            sr_update(
                psi=model,
                sampled_states=sampled_states,
                E_tensor=Elocs,
                L=args.L,
                Nmax=args.Nmax,
                lr=args.lr,
                tau=args.tau,
                device=device,
            )
            update_info = "sr"
        else:
            native_loss = native_vmc_update(
                model=model,
                sampled_states=sampled_states,
                E_tensor=Elocs,
                L=args.L,
                Nmax=args.Nmax,
                optimizer=native_optimizer,
                device=device,
            )
            update_info = f"native_loss={native_loss:.4e}"

        if epoch % args.eval_every == 0 or epoch == 1:
            delta = E_mean.item() - ed_energy
            print(
                f"epoch={epoch:4d}  "
                f"E_vmc={E_mean.item(): .8f}  "
                f"sigma={E_std.item():.4e}  "
                f"delta_to_ED={delta: .4e} "
                f"{update_info}"
            )

if __name__ == "__main__":
    main()