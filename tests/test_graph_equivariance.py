# Revised tests for graph equivariance: termwise boundary check

# tests/test_graph_equivariance.py

import pytest
import random
import itertools

np = pytest.importorskip("numpy")

from simplicial_tensors.tensor_ops import face, bdry

def relabel_matrix(A: np.ndarray, sigma: list[int]) -> np.ndarray:
    """
    Relabel both rows and columns of the adjacency matrix A
    according to the vertex‐permutation sigma.
    """
    return A[np.ix_(sigma, sigma)]

def restrict_sigma(sigma: list[int], i: int) -> list[int]:
    """
    Given a permutation sigma of length n and a deleted index i,
    return the induced permutation on the surviving n-1 vertices.
    """
    n = len(sigma)
    J = [j for j in range(n) if j != i]
    K = [k for k in range(n) if k != sigma[i]]
    return [K.index(sigma[j]) for j in J]

@pytest.mark.parametrize("n,seed", [(4, 0), (5, 1), (6, 42)])
def test_adjacency_face_equivariance(n: int, seed: int) -> None:
    """
    For a random adjacency matrix A and random permutation sigma,
    verify d_i(A^σ) = (d_{σ(i)} A)^{σ_{(i)}} for all i.
    """
    rng = np.random.default_rng(seed)
    # random symmetric adjacency with zero diagonal
    A = np.zeros((n, n), int)
    for u, v in itertools.combinations(range(n), 2):
        if rng.random() < 0.5:
            A[u, v] = A[v, u] = 1

    rng2 = random.Random(seed)
    sigma = list(range(n))
    rng2.shuffle(sigma)

    Aσ = relabel_matrix(A, sigma)

    for i in range(n):
        lhs = face(Aσ, i)
        orig_face = face(A, sigma[i])
        sigma_i = restrict_sigma(sigma, i)
        rhs = relabel_matrix(orig_face, sigma_i)
        np.testing.assert_array_equal(
            lhs, rhs,
            f"face‐equivariance failed for n={n}, seed={seed}, i={i}"
        )

@pytest.mark.parametrize("n,seed", [(4, 0), (5, 1), (6, 42)])
def test_adjacency_boundary_equivariance(n: int, seed: int) -> None:
    """
    Verify ∂(A^σ) = σ⋅∂(A)⋅σ^{-1} term by term:
      ∂(A^σ) == Σ_i (-1)^i ⋅ (d_{σ(i)}A)^{σ_{(i)}}.
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), int)
    for u, v in itertools.combinations(range(n), 2):
        if rng.random() < 0.5:
            A[u, v] = A[v, u] = 1

    rng2 = random.Random(seed)
    sigma = list(range(n))
    rng2.shuffle(sigma)

    # compute boundary after relabeling
    Aσ = relabel_matrix(A, sigma)
    lhs = bdry(Aσ)

    # build rhs termwise
    rhs = np.zeros_like(lhs)
    for i in range(n):
        # face at σ(i), then relabel by restricted sigma
        term = face(A, sigma[i])
        term_relabel = relabel_matrix(term, restrict_sigma(sigma, i))
        rhs += ((-1)**i) * term_relabel

    np.testing.assert_array_equal(
        lhs, rhs,
        f"boundary‐equivariance failed for n={n}, seed={seed}"
    )


