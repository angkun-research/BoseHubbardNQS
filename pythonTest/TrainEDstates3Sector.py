import numpy as np
import torch
import torch.nn as nn
import argparse

from BoseHubbardED import exact_ground_state, basis_to_one_hot
from NeuralNetworkClass import FCNet, JastrowNet, Conv1DNet, RBMNet
from utils import pick_best_device

def make_single_sector_dataset(L, Ntot, Nmax, t, U, mu=0.0, pbc=True, target="logabs", eps=1e-12):
    energy, psi0, basis, H = exact_ground_state(
        L=L, Ntot=Ntot, Nmax=Nmax, t=t, U=U, mu=mu, pbc=pbc
    )
    x = basis_to_one_hot(basis, n_sites=L, nmax=Nmax)

    if target == "amplitude":
        y = np.asarray(psi0, dtype=np.float32)
    elif target == "logabs":
        y = np.log(np.abs(psi0) + eps).astype(np.float32)
    else:
        raise ValueError(f"unknown target mode: {target}")

    return x, y, energy, basis, psi0

def train_model_on_combined_data(
    model,
    x,
    y,
    target="logabs",
    lr=1e-3,
    epochs=10000,
    device="cpu",
):
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()

        if target == "logabs":
            logpsi = model(x_tensor)
            # Local normalization for the current batch (the whole dataset here)
            pred = logpsi - 0.5 * torch.logsumexp(2.0 * logpsi, dim=0)
            y_norm = y_tensor - 0.5 * torch.logsumexp(2.0 * y_tensor, dim=0)
            loss = loss_fn(pred, y_norm)
        else:
            pred = model(x_tensor)
            loss = loss_fn(pred, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"epoch={epoch:4d}  loss={loss.item():.6e}")

    return model

@torch.no_grad()
def compute_sector_fidelity(model, x, y, device="cpu"):
    """
    Compute fidelity (overlap) between the NN and ED ground state
    within a specific sector using local normalization.
    """
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device).reshape(-1)

    # NN prediction for this sector
    logpsi_raw = model(x_tensor).reshape(-1)

    # Local normalization: psi_norm = exp(logpsi - 0.5 * logsumexp(2*logpsi))
    logZ_nn = torch.logsumexp(2.0 * logpsi_raw, dim=0)
    psi_pred = torch.exp(logpsi_raw - 0.5 * logZ_nn)

    # ED normalization: phi_norm = exp(y - 0.5 * logsumexp(2*y))
    logZ_ed = torch.logsumexp(2.0 * y_tensor, dim=0)
    psi_true = torch.exp(y_tensor - 0.5 * logZ_ed)

    overlap = torch.abs(torch.dot(psi_pred, psi_true))
    return overlap.item()

def main():
    parser = argparse.ArgumentParser(description="Train NQS on three sectors (N-1, N, N+1)")
    parser.add_argument("--arch", type=str, default="FCNet", choices=["FCNet", "JastrowNet", "Conv1DNet", "RBMNet"])
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--Nmax", type=int, default=8)
    parser.add_argument("--Ntot", type=int, default=8)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--U", type=float, default=2.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    args = parser.parse_args()

    DEVICE = pick_best_device()
    print(f"Using device: {DEVICE}")

    sectors = [args.Ntot - 1, args.Ntot, args.Ntot + 1]
    all_x = []
    all_y = []
    sector_data = {}

    print("Generating ED data for three sectors...")
    for n in sectors:
        if n < 0: continue
        x, y, energy, basis, psi0 = make_single_sector_dataset(
            L=args.L, Ntot=n, Nmax=args.Nmax, t=args.t, U=args.U, mu=args.mu
        )
        all_x.append(x)
        all_y.append(y)
        sector_data[n] = {"x": x, "y": y}
        print(f"Sector N={n}: dim={len(basis)}, energy={energy:.6f}")

    # Combine all sectors into one training set
    X_train = np.concatenate(all_x, axis=0)
    Y_train = np.concatenate(all_y, axis=0)
    print(f"Combined training set size: {len(X_train)}")

    # Model selection
    arch_builders = {
        "FCNet":       lambda: FCNet(n_sites=args.L, channels=args.Nmax + 1, hidden_dim=args.hidden_dim),
        "JastrowNet":  lambda: JastrowNet(n_sites=args.L, channels=args.Nmax + 1),
        "Conv1DNet":   lambda: Conv1DNet(n_sites=args.L, channels=args.Nmax + 1, hidden_dim=args.hidden_dim),
        "RBMNet":      lambda: RBMNet(n_sites=args.L, channels=args.Nmax + 1, n_hidden=args.hidden_dim),
    }

    print(f"Using architecture: {args.arch}")
    model = arch_builders[args.arch]()

    # Train on the combined dataset
    model = train_model_on_combined_data(
        model, X_train, Y_train, target="logabs", lr=args.lr, epochs=args.epochs, device=DEVICE
    )

    print("\n--- Final Evaluation (Local Normalization) ---")
    print(f"{'Sector':<10} | {'Fidelity':<15}")
    print("-" * 27)
    for n in sectors:
        if n < 0: continue
        data = sector_data[n]
        fidelity = compute_sector_fidelity(model, data["x"], data["y"], device=DEVICE)
        print(f"N={n:<8} | {fidelity:.8f}")

if __name__ == "__main__":
    main()
