"""A/B experiments comparing incidence and principal boundary penalties on MNIST-like data."""

from __future__ import annotations

import argparse
from typing import Tuple

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:  # pragma: no cover - friendly message when torch missing
    raise SystemExit("PyTorch is required for the experiments scaffold.") from exc

try:  # pragma: no cover - torchvision is optional
    from torchvision import datasets, transforms
    HAS_TORCHVISION = True
except ModuleNotFoundError:
    HAS_TORCHVISION = False

from simplicial_tensors.nn import BoundaryConfig, train_with_boundary


def make_synthetic_dataset(num_samples: int = 1024) -> Tuple[TensorDataset, TensorDataset]:
    torch.manual_seed(0)
    features = torch.randn(num_samples, 1, 28, 28)
    labels = ((features.sum(dim=(1, 2, 3)) > 0).long())
    split = num_samples // 5
    val = TensorDataset(features[:split], labels[:split])
    train = TensorDataset(features[split:], labels[split:])
    return train, val


def build_loaders(batch_size: int, use_torchvision: bool = HAS_TORCHVISION):
    if use_torchvision:
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST("./data", train=True, download=True, transform=transform)
        val = datasets.MNIST("./data", train=False, download=True, transform=transform)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=batch_size)
        return train_loader, val_loader
    train_ds, val_ds = make_synthetic_dataset()
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size)


def make_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def run_once(cfg: BoundaryConfig, epochs: int, batch_size: int, lr: float) -> Tuple[float, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = build_loaders(batch_size)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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
    return history["val_loss"][-1], history["val_boundary"][-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST A/B boundary penalty experiment")
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs for each run")
    parser.add_argument("--batch-size", type=int, default=128, help="mini-batch size")
    parser.add_argument("--lambda1", type=float, default=1e-3, help="regularisation weight")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    args = parser.parse_args()

    cfg_incidence = BoundaryConfig(mode="incidence", lambda1=args.lambda1)
    cfg_principal = BoundaryConfig(mode="principal", lambda1=args.lambda1, coboundary="zero")

    val_loss_inc, val_bound_inc = run_once(cfg_incidence, args.epochs, args.batch_size, args.lr)
    print(f"[incidence] epochs={args.epochs} val_loss={val_loss_inc:.4f} val_boundary={val_bound_inc:.4f}")

    val_loss_pr, val_bound_pr = run_once(cfg_principal, args.epochs, args.batch_size, args.lr)
    print(f"[principal] epochs={args.epochs} val_loss={val_loss_pr:.4f} val_boundary={val_bound_pr:.4f}")


if __name__ == "__main__":
    main()
