"""Boundary-aware regularisation utilities for neural models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ModuleNotFoundError:  # pragma: no cover - handled dynamically
    torch = None  # type: ignore
    Tensor = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for simplicial_tensors.nn; install torch to use these utilities."
        )


def _vector_norm(vec: "Tensor", p: float = 2.0, eps: float = 0.0) -> "Tensor":
    vec = vec.flatten()
    if p == 2.0:
        return torch.sqrt(torch.sum(vec * vec) + eps)
    if p == 1.0:
        return torch.sum(vec.abs()) + eps
    if p == float("inf"):
        return vec.abs().max() + eps
    return torch.pow(torch.sum(torch.abs(vec) ** p) + eps, 1.0 / p)


def incidence_boundary_vec(W: "Tensor", eps: float = 0.0) -> "Tensor":
    """Return the concatenated incidence boundary vector for a weight matrix."""

    _require_torch()
    ones_in = torch.ones(W.shape[1], device=W.device, dtype=W.dtype)
    ones_out = torch.ones(W.shape[0], device=W.device, dtype=W.dtype)
    inbound = W @ ones_in
    outbound = -(W.t() @ ones_out)
    if eps:
        inbound = inbound + eps
        outbound = outbound - eps
    return torch.cat([inbound, outbound])


def incidence_boundary_norm(W: "Tensor", p: float = 2.0, eps: float = 0.0) -> "Tensor":
    """Compute the p-norm of the incidence boundary vector."""

    vec = incidence_boundary_vec(W, eps=0.0)
    return _vector_norm(vec, p=p, eps=eps)


def incidence_penalty(
    linears: Sequence["nn.Linear"],
    lambda1: float,
    p: float = 2.0,
    eps: float = 1e-8,
    squared: bool = True,
) -> Tuple["Tensor", Dict[str, List[float]]]:
    """Aggregate incidence penalties for a collection of linear layers."""

    _require_torch()
    if not linears:
        penalty = torch.zeros((), requires_grad=True)
        return penalty, {"incidence_norms": [], "incidence_vec_norm": 0.0}

    device = linears[0].weight.device
    penalty = torch.zeros((), device=device)
    norms: List[float] = []
    for layer in linears:
        norm = incidence_boundary_norm(layer.weight, p=p, eps=eps)
        norms.append(norm.detach().item())
        if squared:
            norm = norm * norm
        penalty = penalty + norm
    penalty = lambda1 * penalty
    return penalty, {"incidence_norms": norms, "incidence_penalty": penalty.detach().item()}


def incidence_laplacian(W: "Tensor") -> "Tensor":
    """Return a Laplacian-style feedback matrix with the same shape as ``W``."""

    _require_torch()
    row_sum = W.abs().sum(dim=1, keepdim=True)
    col_sum = W.abs().sum(dim=0, keepdim=True)
    return row_sum - col_sum


if torch is not None:  # pragma: no cover - executed when torch is present

    class BoundaryFeedbackLinear(nn.Linear):
        """Linear layer that adds a Laplacian feedback to its weights during the forward pass."""

        def __init__(self, *args, beta: float = 0.0, laplacian_fn=incidence_laplacian, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.beta = beta
            self.laplacian_fn = laplacian_fn

        def forward(self, input: "Tensor") -> "Tensor":  # type: ignore[override]
            weight = self.weight
            if self.beta:
                weight = weight + self.beta * self.laplacian_fn(self.weight)
            return F.linear(input, weight, self.bias)

        def extra_repr(self) -> str:
            return super().extra_repr() + f", beta={self.beta}"

else:  # pragma: no cover - placeholder when torch is not installed

    class BoundaryFeedbackLinear:  # type: ignore[override]
        def __init__(self, *_, **__):
            _require_torch()

        def forward(self, *_args, **_kwargs):
            _require_torch()

        def extra_repr(self) -> str:
            return "beta=?"


def principal_boundary(W: "Tensor") -> Tuple["Tensor", "Tensor"]:
    """Compute the alternating sum of principal faces for a square-like weight matrix."""

    _require_torch()
    out_dim, in_dim = W.shape[-2:]
    width = min(out_dim, in_dim)
    faces: List[Tensor] = []
    indices: List[int] = []
    for idx in range(width):
        rows = torch.cat((W[:idx], W[idx + 1 :]), dim=-2)
        face = torch.cat((rows[..., :idx], rows[..., idx + 1 :]), dim=-1)
        faces.append(((-1) ** idx) * face)
        indices.append(idx)
    if not faces:
        empty_shape = (0, max(out_dim - 1, 0), max(in_dim - 1, 0))
        empty_faces = W.new_zeros(empty_shape)
        empty_indices = torch.empty(0, dtype=torch.long, device=W.device)
        return empty_faces, empty_indices
    stacked = torch.stack(faces)
    idx_tensor = torch.tensor(indices, device=W.device, dtype=torch.long)
    return stacked, idx_tensor


def _principal_embed(
    face: "Tensor",
    index: int,
    shape: Tuple[int, int],
    mode: str = "zero",
) -> "Tensor":
    out_dim, in_dim = shape
    result = face.new_zeros((out_dim, in_dim))
    if out_dim == 0 or in_dim == 0:
        return result
    rows = torch.tensor([i for i in range(out_dim) if i != index], device=face.device)
    cols = torch.tensor([j for j in range(in_dim) if j != index], device=face.device)
    if rows.numel() == 0 or cols.numel() == 0:
        return result
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    result.index_put_((rr, cc), face)
    if mode == "degen":
        src_r = index - 1 if index > 0 else 0
        src_c = index - 1 if index > 0 else 0
        result[index, :] = result[src_r, :]
        result[:, index] = result[:, src_c]
    return result


def principal_laplacian(W: "Tensor", coboundary: str = "zero") -> "Tensor":
    """Construct a Laplacian-like feedback for the principal boundary."""

    faces, indices = principal_boundary(W)
    result = torch.zeros_like(W)
    for face, idx in zip(faces, indices.tolist()):
        result = result + _principal_embed(face, idx, W.shape[-2:], mode=coboundary)
    return result


def principal_boundary_norm(
    W: "Tensor",
    p: float = 2.0,
    eps: float = 0.0,
    coboundary: str = "zero",
) -> "Tensor":
    """Compute a norm of the principal boundary returned to the matrix shape."""

    boundary = principal_laplacian(W, coboundary=coboundary)
    return _vector_norm(boundary, p=p, eps=eps)


def principal_penalty(
    linears: Sequence["nn.Linear"],
    lambda1: float,
    p: float = 2.0,
    eps: float = 1e-8,
    squared: bool = True,
    coboundary: str = "zero",
) -> Tuple["Tensor", Dict[str, List[float]]]:
    """Aggregate principal penalties for a collection of linear layers."""

    _require_torch()
    if not linears:
        penalty = torch.zeros((), requires_grad=True)
        return penalty, {"principal_norms": [], "principal_penalty": 0.0}

    device = linears[0].weight.device
    penalty = torch.zeros((), device=device)
    norms: List[float] = []
    for layer in linears:
        norm = principal_boundary_norm(layer.weight, p=p, eps=eps, coboundary=coboundary)
        norms.append(norm.detach().item())
        if squared:
            norm = norm * norm
        penalty = penalty + norm
    penalty = lambda1 * penalty
    return penalty, {"principal_norms": norms, "principal_penalty": penalty.detach().item()}


@dataclass
class BoundaryConfig:
    mode: str = "incidence"
    coboundary: str = "zero"
    lambda1: float = 1e-3
    p: float = 2.0
    eps: float = 1e-8
    squared: bool = True
    residual_beta: float = 0.0


def boundary_penalty(
    linears: Sequence["nn.Linear"],
    cfg: BoundaryConfig,
) -> Tuple["Tensor", Dict[str, List[float]]]:
    """Dispatch to the requested boundary penalty."""

    _require_torch()
    mode = cfg.mode.lower()
    if mode == "incidence":
        penalty, diag = incidence_penalty(
            linears,
            lambda1=cfg.lambda1,
            p=cfg.p,
            eps=cfg.eps,
            squared=cfg.squared,
        )
    elif mode == "principal":
        penalty, diag = principal_penalty(
            linears,
            lambda1=cfg.lambda1,
            p=cfg.p,
            eps=cfg.eps,
            squared=cfg.squared,
            coboundary=cfg.coboundary,
        )
    else:  # pragma: no cover - configuration error
        raise ValueError(f"Unsupported boundary mode: {cfg.mode}")
    return penalty, diag


def _collect_linears(module: "nn.Module") -> List["nn.Linear"]:
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


def train_with_boundary(
    model: "nn.Module",
    train_loader,
    val_loader,
    optimizer: "torch.optim.Optimizer",
    criterion,
    device: str,
    epochs: int,
    cfg: BoundaryConfig,
    project: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Train a model with boundary regularisation and optional W&B logging."""

    _require_torch()
    model = model.to(device)
    history = {"train_loss": [], "train_boundary": [], "val_loss": [], "val_boundary": []}

    wandb_run = None
    if project:
        try:  # pragma: no cover - optional dependency
            import wandb

            wandb_run = wandb.init(project=project, name=run_name, config=cfg.__dict__)
        except Exception as exc:  # pragma: no cover - log but continue
            print(f"wandb unavailable: {exc}")
            wandb_run = None

    global_step = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_penalty = 0.0
        batches = 0
        for inputs, targets in train_loader:
            batches += 1
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            linears = _collect_linears(model)
            penalty, diag = boundary_penalty(linears, cfg)
            total = loss + penalty
            total.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            running_penalty += penalty.detach().item()

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": loss.detach().item(),
                        "train/boundary": penalty.detach().item(),
                    },
                    step=global_step,
                )
            global_step += 1

        epoch_loss = running_loss / max(batches, 1)
        epoch_penalty = running_penalty / max(batches, 1)
        history["train_loss"].append(epoch_loss)
        history["train_boundary"].append(epoch_penalty)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                val_batches += 1
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss = val_loss / max(val_batches, 1)
        history["val_loss"].append(val_loss)
        with torch.no_grad():
            linears = _collect_linears(model)
            val_penalty, _ = boundary_penalty(linears, cfg)
        history["val_boundary"].append(val_penalty.detach().item())

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "val/loss": val_loss,
                    "val/boundary": history["val_boundary"][-1],
                },
                step=global_step,
            )

    if wandb_run is not None:  # pragma: no cover - depends on wandb
        wandb_run.finish()

    return history


__all__ = [
    "BoundaryConfig",
    "BoundaryFeedbackLinear",
    "boundary_penalty",
    "incidence_boundary_norm",
    "incidence_boundary_vec",
    "incidence_laplacian",
    "incidence_penalty",
    "principal_boundary",
    "principal_boundary_norm",
    "principal_penalty",
    "principal_laplacian",
    "train_with_boundary",
]
