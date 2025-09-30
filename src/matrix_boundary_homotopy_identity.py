import numpy as np
from tensor_ops import bdry, face  # your library

def perm_matrix_from_sigma(sigma):
    """P_sigma with (P)[i,j]=1 iff i = sigma(j)."""
    n = len(sigma)
    P = np.zeros((n, n), dtype=int)
    for j in range(n):
        P[sigma[j], j] = 1
    return P

def conjugate_by_sigma(A, sigma):
    """Relabel adjacency by sigma: B = P_sigma A P_sigma^T."""
    P = perm_matrix_from_sigma(sigma)
    return P @ A @ P.T

def induced_subperm_matrix(sigma, i):
    """
    Induced (n-1)x(n-1) permutation on indices after removing i.
    Implements the bijection j -> sigma^{-1}(j) on {0..n-1}\\{i}, then reindexes to 0..n-2.
    Returns the (n-1)x(n-1) permutation matrix Q_i.
    """
    n = len(sigma)
    sigma_inv = np.empty(n, dtype=int)
    for j in range(n):
        sigma_inv[sigma[j]] = j

    # Domain indices after removing i, in increasing order:
    dom = [j for j in range(n) if j != i]
    # Their images under sigma^{-1}:
    img = [sigma_inv[j] for j in dom]
    # Codomain standard order is {0..n-1}\{sigma^{-1}(i)} in increasing order:
    cod = [j for j in range(n) if j != sigma_inv[i]]

    # Build the (n-1)x(n-1) permutation Q mapping 'cod' -> 'img' orders.
    pos_in_img = {val: t for t, val in enumerate(img)}  # value -> position in 'img'
    Q = np.zeros((n-1, n-1), dtype=int)
    for s, val in enumerate(cod):         # 'cod' source position s holding value 'val'
        t = pos_in_img[val]               # target position in 'img'
        Q[t, s] = 1
    return Q

def principal_face(A, i):
    """Principal submatrix d_i(A) removing row/col i (uses your face() wrapper)."""
    return face(A, i)

def boundary_by_faces_via_sigma(A, sigma):
    """
    Right-hand side of the boxed identity:
    sum_i (-1)^i Q_i^T d_{sigma^{-1}(i)}(A) Q_i.
    """
    n = A.shape[0]
    acc = np.zeros((n-1, n-1), dtype=int)
    for i in range(n):
        Q = induced_subperm_matrix(sigma, i)
        di = principal_face(A, inverse_image(sigma, i))  # d_{sigma^{-1}(i)}(A)
        term = Q.T @ di @ Q
        if (i % 2) == 0:
            acc = acc + term
        else:
            acc = acc - term
    return acc

def inverse_image(sigma, i):
    """Return sigma^{-1}(i)."""
    for j, sj in enumerate(sigma):
        if sj == i:
            return j
    raise ValueError("Not a permutation.")

def verify_boundary_equivariance_once(n=6, directed=True, mod2=False, seed=0):
    rng = np.random.default_rng(seed)
    # random permutation
    sigma = np.arange(n)
    rng.shuffle(sigma)

    # random adjacency (0/1), zero diagonal
    A = rng.integers(low=0, high=2, size=(n, n), dtype=int)
    np.fill_diagonal(A, 0)
    if not directed:
        A = np.triu(A, 1)
        A = A + A.T  # undirected simple graph

    # compute LHS = bdry(B) with B = P A P^T
    B = conjugate_by_sigma(A, sigma)
    lhs = bdry(B).astype(int)

    # compute RHS = sum_i (-1)^i Q_i^T d_{sigma^{-1}(i)}(A) Q_i
    rhs = boundary_by_faces_via_sigma(A, sigma)

    if mod2:
        lhs = np.mod(lhs, 2)
        rhs = np.mod(rhs, 2)

    ok = np.array_equal(lhs, rhs)
    return ok, A, sigma, lhs, rhs

def run_many_trials(trials=50, n=6, directed=True, mod2=False, seed=123):
    ok_all = True
    for t in range(trials):
        ok, A, sigma, lhs, rhs = verify_boundary_equivariance_once(
            n=n, directed=directed, mod2=mod2, seed=seed + t
        )
        if not ok:
            print("Counterexample found!")
            print("sigma =", sigma)
            print("A =\n", A)
            print("bdry(P A P^T) =\n", lhs)
            print("sum_i (-1)^i Q_i^T d_{sigma^{-1}(i)}(A) Q_i =\n", rhs)
            ok_all = False
            break
    if ok_all:
        print(f"Verified identity in {trials} random trials (n={n}, directed={directed}, mod2={mod2}).")

if __name__ == "__main__":
    # Over Z (with signs):
    run_many_trials(trials=100, n=7, directed=True, mod2=False)
    # Over Z2 (mod 2):
    run_many_trials(trials=100, n=7, directed=True, mod2=True)
