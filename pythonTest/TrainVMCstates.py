#!/usr/bin/env python3
"""Train a Bose-Hubbard NQS with batched VMC and save an inference checkpoint.

Example (from the repository root):
    python pythonTest/TrainVMCstates.py --L 4 --Ntot 4 --Nmax 4 --epochs 100
"""

import argparse
from pathlib import Path
import random

import numpy as np
import torch

from NeuralNetworkClass import FCNet, JastrowNet, Conv1DNet, RBMNet, save_model
from utils import pick_best_device
from vmc_utils_gpu import (
    generate_initial_state, Obtain_Sampling_batch_gpu, native_vmc_update, sr_update_optimizer,
)


ARCH_BUILDERS = {
    "FCNet": lambda L, C, h: FCNet(n_sites=L, channels=C, hidden_dim=h),
    "JastrowNet": lambda L, C, h: JastrowNet(n_sites=L, channels=C),
    "Conv1DNet": lambda L, C, h: Conv1DNet(n_sites=L, channels=C, hidden_dim=h),
    "RBMNet": lambda L, C, h: RBMNet(n_sites=L, channels=C, n_hidden=h),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=list(ARCH_BUILDERS), default="FCNet")
    for name, default in (("L", 4), ("Ntot", 4), ("Nmax", 4), ("hidden_dim", 64),
                          ("epochs", 100), ("n_walkers", 64), ("n_steps", 32),
                          ("burnin_steps", 32), ("eval_every", 10), ("seed", 123)):
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("t", 1.0), ("U", 2.0), ("mu", 0.0), ("lr", 5e-3), ("tau", 1e-3)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--pbc", action="store_true", default=True)
    parser.add_argument("--no-pbc", dest="pbc", action="store_false")
    parser.add_argument("--update_method", choices=["sr", "native"], default="sr")
    parser.add_argument("--optimizer", "--native_opt", choices=["adam", "sgd"],
                        help="optimizer for either update method (default: sr=sgd, native=adam)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda[:index], or mps")
    parser.add_argument("--evaluation_batch_size", "--evaluation-batch-size", type=int)
    parser.add_argument("--ed-reference", action="store_true", help="compute dense ED for comparison")
    parser.add_argument("--output", type=Path, default=(
        Path(__file__).resolve().parents[1] / "data" / "bose_hubbard_vmc_nqs.pt"
    ))
    args = parser.parse_args()
    if (min(args.L, args.hidden_dim, args.epochs, args.n_walkers, args.n_steps, args.eval_every) <= 0
            or args.Nmax < 0 or not 0 <= args.Ntot <= args.L * args.Nmax
            or args.burnin_steps < 0 or args.lr <= 0 or args.tau <= 0
            or (args.evaluation_batch_size is not None and args.evaluation_batch_size <= 0)):
        parser.error("invalid lattice, particle count, training count, learning rate, tau, or chunk size")

    if args.optimizer is None:
        args.optimizer = "sgd" if args.update_method == "sr" else "adam"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_best_device() if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}; arch={args.arch}; update={args.update_method}; "
          f"optimizer={args.optimizer}; seed={args.seed}")
    model = ARCH_BUILDERS[args.arch](args.L, args.Nmax + 1, args.hidden_dim).to(device)
    # The supported models are deterministic; eval also keeps future dropout disabled.
    model.eval()
    optimizer = (torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == "adam"
                 else torch.optim.SGD(model.parameters(), lr=args.lr))

    ed_energy = None
    if args.ed_reference:
        if args.L == 1 or args.Ntot in (0, args.L * args.Nmax):
            state = generate_initial_state(args.L, args.Ntot, args.Nmax)
            ed_energy = sum(args.U * n * (n - 1) - args.mu * n for n in state)
        else:
            from BoseHubbardED import exact_ground_state

            ed_energy, _, _, _ = exact_ground_state(
                args.L, args.Ntot, args.Nmax, args.t, args.U, args.mu, args.pbc
            )
        ed_energy = float(ed_energy)
        print(f"ED ground-state energy: {ed_energy:.10f}")

    states = [generate_initial_state(args.L, args.Ntot, args.Nmax) for _ in range(args.n_walkers)]
    sampling = dict(psi=model, L=args.L, Nmax=args.Nmax, t=args.t, U=args.U,
                    mu=args.mu, pbc=args.pbc, evaluation_batch_size=args.evaluation_batch_size)
    history = []
    for epoch in range(1, args.epochs + 1):
        _, _, states, _ = Obtain_Sampling_batch_gpu(
            initial_states=states, n_steps=args.burnin_steps, burnin=True, **sampling
        )
        logs, energies, states, samples = Obtain_Sampling_batch_gpu(
            initial_states=states, n_steps=args.n_steps, **sampling
        )
        mean, std = energies.mean().item(), energies.std(unbiased=False).item()
        if not np.isfinite(mean) or not np.isfinite(std):
            raise RuntimeError(f"Non-finite sampled energy at epoch {epoch}")
        history.append({"epoch": epoch, "energy": mean, "std": std})
        if args.update_method == "sr":
            del logs
            sr_update_optimizer(model, samples, energies, args.L, args.Nmax, optimizer, args.tau,
                                evaluation_batch_size=args.evaluation_batch_size)
        else:
            native_vmc_update(model, samples, energies, args.L, args.Nmax, optimizer, logpsis=logs)
            del logs
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            delta = "" if ed_energy is None else f"  delta_to_ED={mean - ed_energy:.4e}"
            print(f"epoch={epoch:4d}  E_vmc={mean:.8f}  sigma={std:.4e}{delta}")

    # Measure once more after the last update, so this energy describes the saved model.
    _, _, states, _ = Obtain_Sampling_batch_gpu(
        initial_states=states, n_steps=args.burnin_steps, burnin=True, **sampling
    )
    logs, energies, _, _ = Obtain_Sampling_batch_gpu(
        initial_states=states, n_steps=args.n_steps, **sampling
    )
    del logs
    final_energy, final_std = energies.mean().item(), energies.std(unbiased=False).item()
    if (not np.isfinite(final_energy) or not np.isfinite(final_std)
            or not all(torch.isfinite(p).all().item() for p in model.parameters())):
        raise RuntimeError("Non-finite final model or energy; checkpoint not saved")
    print(f"Final model: E_vmc={final_energy:.8f}  sigma={final_std:.4e}")
    settings = {**vars(args), "output": str(args.output), "device": str(device)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_model(args.output, model.cpu().eval(), args.arch, args.L, args.Nmax, args.hidden_dim,
               extra={"training_method": "vmc", "target": "logabs", "settings": settings,
                      "energy_history": history, "final_energy": final_energy,
                      "final_energy_std": final_std, "ed_energy": ed_energy})


if __name__ == "__main__":
    main()
