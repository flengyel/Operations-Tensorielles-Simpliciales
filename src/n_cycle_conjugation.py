# n_cycle_conjugation.py

from sympy import simplify
from symbolic_tensor_ops import SymbolicTensor
import numpy as np
from collections import Counter


def permutation_cycle_partition(perm):
    """
    Return the partition (sorted list of cycle lengths) of a permutation.
    Knuth's algorithm: Loop over unvisited elements, tracing out cycles.
    """
    n = len(perm)
    seen = [False] * n
    partition = []

    for i in range(n):
        if not seen[i]:
            length = 0
            j = i
            while not seen[j]:
                seen[j] = True
                j = perm[j]
                length += 1
            if length > 0:
                partition.append(length)

    return tuple(sorted(partition, reverse=True))


def generate_n_cycle(n):
    """Return a single n-cycle permutation."""
    return [(i + 1) % n for i in range(n)]


def test_conjugation_by_n_cycle(n):
    A = SymbolicTensor((n, n))
    perm = generate_n_cycle(n)
    P = np.eye(n)[perm]
    P = np.array(P, dtype=int)

    a_tensor = A.tensor
    b_tensor = P.T @ a_tensor @ P
    B = SymbolicTensor((n, n), tensor=b_tensor)

    boundary_a = A.bdry()
    boundary_b = B.bdry()

    diff_tensor = boundary_a.tensor - boundary_b.tensor
    print(f"difference tensor {diff_tensor}")

    # Dummy values for result, actual, theoretical
    result = np.all(diff_tensor == 0)
    actual = np.count_nonzero(diff_tensor)
    theoretical = (n - 1) ** 2
    return result, actual, theoretical
    


def run_n_cycle_tests(min_n: int = 3, max_n: int = 15):
    print("Testing conjugation by n-cycle permutations:")
    for n in range(min_n, max_n + 1):
        result, actual, theoretical = test_conjugation_by_n_cycle(n)
        status = "PASS" if result else "FAIL"
        print(f"n={n:2d}: {status}   Nonzero cocycles: {actual:4d} / {(n - 1):2d}^2 = {theoretical:4d}")


if __name__ == "__main__":
    run_n_cycle_tests(3, 15)
