"""Run a tiny A/B experiment comparing boundary penalties."""

import math
from typing import Tuple

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:  # pragma: no cover - friendly exit when torch missing
    print("PyTorch is required to run the boundary penalty demo. Please install torch.")
    raise SystemExit(0)

from simplicial_tensors.nn import BoundaryConfig, train_with_boundary


def make_synthetic_dataset(n: int = 1024, noise: float = 0.1) -> Tuple[TensorDataset, TensorDataset]:
    torch.manual_seed(0)
    x = torch.randn(n, 2)
    y = torch.sign(x[:, 0] * x[:, 1]).unsqueeze(-1)
    y[y == 0] = 1
    y = (y > 0).float()

    x += noise * torch.randn_like(x)
    split = n // 5
    val_x, val_y = x[:split], y[:split]
    train_x, train_y = x[split:], y[split:]
    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    return train_ds, val_ds


def make_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def run_job(mode: str, coboundary: str = "zero", lambda1: float = 1e-3, epochs: int = 10) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds = make_synthetic_dataset()
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = make_model(seed=42)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.BCEWithLogitsLoss()

    cfg = BoundaryConfig(mode=mode, coboundary=coboundary, lambda1=lambda1)
    history = train_with_boundary(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs,
        cfg,
    )

    final_loss = history["val_loss"][-1]
    final_boundary = history["val_boundary"][-1]
    print(f"Mode={mode!r} coboundary={coboundary!r}: val_loss={final_loss:.4f} boundary={final_boundary:.4f}")


def main() -> None:
    print("=== Boundary penalty A/B demo ===")
    run_job(mode="incidence", lambda1=5e-4, epochs=8)
    run_job(mode="principal", coboundary="zero", lambda1=5e-4, epochs=8)


if __name__ == "__main__":
    main()
