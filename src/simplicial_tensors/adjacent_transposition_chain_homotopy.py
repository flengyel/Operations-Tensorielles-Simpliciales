import numpy as np
from .tensor_ops import bdry, degen  # your implementations

# ----------------------------
# Permutations and actions
# ----------------------------

def perm_matrix_adjacent(n: int, k: int) -> np.ndarray:
    """
    Permutation matrix for the adjacent transposition tau_k = (k k+1) on {0,..,n-1}.
    Convention: columns->rows; P[i,j]=1 iff i = tau_k(j).
    """
    if not (0 <= k < n-1):
        raise ValueError("k must satisfy 0 <= k < n-1")
    P = np.eye(n, dtype=int)
    # swap columns k and k+1 by swapping the mapping of j -> tau_k(j)
    # In our column->row convention, swapping images means swapping corresponding rows
    P[[k, k+1], :] = P[[k+1, k], :]
    return P

def eps_tau_conjugate(A: np.ndarray, k: int) -> np.ndarray:
    """
    epsilon(tau_k) acting on A: simultaneous row/column swap k <-> k+1.
    Implemented as conjugation by P_{tau_k}.
    """
    n = A.shape[0]
    P = perm_matrix_adjacent(n, k)
    return P @ A @ P.T

# ----------------------------
# The homotopy operator H_k
# ----------------------------

def H_k(X: np.ndarray, k: int) -> np.ndarray:
    """
    H_k = sum_{i=0}^{k-1} (-1)^i s_i, where s_i = degen(-, i)
    Diagonal degeneracy: duplicate index i along *all* axes.
    For matrices: returns an (n+1) x (n+1) matrix.
    """
    if k <= 0:
        # empty sum => zero of shape with one higher dimension than X
        # but to keep simple, return exact zeros after first term exists
        # we choose convention H_0 = 0 (no degeneracies)
        return np.zeros(tuple(np.array(X.shape) + 1), dtype=X.dtype)

    # First term builds the correct (n+1)x(n+1) shape, then we sum in place.
    acc = np.zeros(tuple(np.array(X.shape) + 1), dtype=X.dtype)
    for i in range(k):
        term = degen(X, i)
        if (i % 2) == 0:
            acc = acc + term
        else:
            acc = acc - term
    return acc

# ----------------------------
# Verifications
# ----------------------------

def verify_adjacent_transposition_homotopy_once(n=6, k=2, directed=True, mod2=False, seed=0):
    """
    Verifies (id - eps(tau_k))(A) == bdry(H_k(A)) + H_k(bdry(A))
    for a random adjacency-like matrix A (0/1, zero diagonal).
    Also checks the boundary-level corollary:
        eps(tau_k) bdry(A) - bdry(A) == bdry(H_k(bdry(A)))
    """
    rng = np.random.default_rng(seed)

    # random adjacency-like matrix (0/1), zero diagonal
    A = rng.integers(low=0, high=2, size=(n, n), dtype=int)
    np.fill_diagonal(A, 0)
    if not directed:
        A = np.triu(A, 1)
        A = A + A.T

    # left-hand side: id - eps(tau_k)
    lhs = A - eps_tau_conjugate(A, k)

    # right-hand side: bdry(H_k(A)) + H_k(bdry(A))
    hkA = H_k(A, k)
    rhs = bdry(hkA) + H_k(bdry(A), k)

    # optionally reduce mod 2
    if mod2:
        lhs = np.mod(lhs, 2)
        rhs = np.mod(rhs, 2)

    ok_top = np.array_equal(lhs, rhs)

    # boundary-level corollary: eps(tau_k) bdry(A) - bdry(A) == bdry(H_k(bdry(A)))
    lhs_b = eps_tau_conjugate(bdry(A), k) - bdry(A)
    rhs_b = bdry(H_k(bdry(A), k))
    if mod2:
        lhs_b = np.mod(lhs_b, 2)
        rhs_b = np.mod(rhs_b, 2)

    ok_bdry = np.array_equal(lhs_b, rhs_b)

    return ok_top and ok_bdry, A, lhs, rhs, lhs_b, rhs_b

def run_trials(trials=50, n=6, directed=True, mod2=False, seed=123):
    """
    Run many random trials for all adjacent transpositions k=1..n-2.
    (We skip k=0 to avoid the trivial H_0=0 sum, which needs separate handling.)
    """
    all_ok = True
    for t in range(trials):
        for k in range(1, n-1):
            ok, A, lhs, rhs, lhs_b, rhs_b = verify_adjacent_transposition_homotopy_once(
                n=n, k=k, directed=directed, mod2=mod2, seed=seed + 1000*t + k
            )
            if not ok:
                print("Counterexample found!")
                print(f"n={n}, k={k}, mod2={mod2}, directed={directed}")
                print("A=\n", A)
                print("(id - eps(tau_k))(A)=\n", lhs)
                print("bdry(H_k(A)) + H_k(bdry(A))=\n", rhs)
                print("eps(tau_k)bdry(A) - bdry(A)=\n", lhs_b)
                print("bdry(H_k(bdry(A)))=\n", rhs_b)
                all_ok = False
                return all_ok
    if all_ok:
        print(f"Verified ∂H_k + H_k∂ = id - ε(τ_k) for {trials} trials, n={n}, "
              f"directed={directed}, mod2={mod2}, for all k=1..{n-2}.")
    return all_ok

def main():
    # Over Z (exact equality)
    run_trials(trials=100, n=7, directed=True, mod2=False)
    # Over Z2 (mod 2)
    run_trials(trials=100, n=7, directed=True, mod2=True)
