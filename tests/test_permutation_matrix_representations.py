# test_permutation_matrix_representations.py

import random
import numpy as np

def kronecker_delta(a: int, b: int) -> int:
    """Kronecker delta: 1 if a == b, else 0."""
    return int(a == b)

def make_hom_matrix(perm):
    """Homomorphic representation A^σ with A[i,j]=δ_{i,σ(j)}."""
    n = len(perm)
    A = np.zeros((n, n), dtype=int)
    for j, sigma_j in enumerate(perm):
        A[sigma_j, j] = 1
    return A

def make_anti_matrix(perm):
    """Anti-homomorphic representation B^σ with B[i,j]=δ_{σ(i),j}."""
    n = len(perm)
    B = np.zeros((n, n), dtype=int)
    for i, sigma_i in enumerate(perm):
        B[i, sigma_i] = 1
    return B

# This is a tautological test, not a unit test.
def test_kronecker_delta_symmetry():
    n = random.randint(2, 50)
    perm = list(range(n))
    random.shuffle(perm)

    def sigma(i): return perm[i]

    # verify δ(j,σ(i)) == δ(σ(i),j)
    for i in range(n):
        for j in range(n):
            assert kronecker_delta(j, sigma(i)) == kronecker_delta(sigma(i), j)

def test_homomorphic_action_and_composition():
    n = random.randint(2, 20)
    perm = list(range(n))
    random.shuffle(perm)
    def sigma(i): return perm[i]

    A = make_hom_matrix(perm)
    I = np.eye(n, dtype=int)

    # action: A e_i = e_{σ(i)}
    for i in range(n):
        e_i = I[:, i]
        e_sigma = I[:, sigma(i)]
        assert np.array_equal(A.dot(e_i), e_sigma)

    # composition: A^{σ∘τ} = A^σ A^τ
    perm_tau = list(range(n))
    random.shuffle(perm_tau)
    def tau(i): return perm_tau[i]

    A_sigma = A
    A_tau = make_hom_matrix(perm_tau)
    composed = [sigma(tau(i)) for i in range(n)]
    A_comp = make_hom_matrix(composed)

    assert np.array_equal(A_sigma.dot(A_tau), A_comp)

def test_anti_homomorphic_action_and_composition_and_transpose():
    n = random.randint(2, 20)
    perm = list(range(n))
    random.shuffle(perm)
    def sigma(i): return perm[i]

    A = make_hom_matrix(perm)
    B = make_anti_matrix(perm)
    I = np.eye(n, dtype=int)

    # relation: B = A^T
    assert np.array_equal(B, A.T)

    # action: e_i^T B = e_{σ(i)}^T
    for i in range(n):
        row = B[i, :]
        e_sigma_row = I[sigma(i), :]
        assert np.array_equal(row, e_sigma_row)

    # composition: B^{σ∘τ} = B^τ B^σ
    perm_tau = list(range(n))
    random.shuffle(perm_tau)
    def tau(i): return perm_tau[i]

    B_sigma = B
    B_tau = make_anti_matrix(perm_tau)
    composed = [sigma(tau(i)) for i in range(n)]
    B_comp = make_anti_matrix(composed)

    assert np.array_equal(B_tau.dot(B_sigma), B_comp)
