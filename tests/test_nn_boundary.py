import pytest

torch = pytest.importorskip("torch")

from simplicial_tensors.nn import (
    BoundaryConfig,
    boundary_penalty,
    incidence_boundary_vec,
    principal_boundary,
    principal_laplacian,
)


def test_incidence_boundary_vec_shape() -> None:
    W = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    vec = incidence_boundary_vec(W)
    assert vec.shape[0] == W.shape[0] + W.shape[1]


def test_principal_boundary_and_laplacian() -> None:
    W = torch.arange(9.0, dtype=torch.float32).reshape(3, 3)
    faces, indices = principal_boundary(W)
    assert faces.shape[0] == indices.numel() == 3
    lap = principal_laplacian(W)
    assert lap.shape == W.shape


def test_boundary_penalty_dispatch() -> None:
    layer = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.eye_(layer.weight)
    penalty, diag = boundary_penalty([layer], BoundaryConfig(mode="incidence", lambda1=1.0))
    assert penalty.requires_grad
    assert "incidence_norms" in diag
