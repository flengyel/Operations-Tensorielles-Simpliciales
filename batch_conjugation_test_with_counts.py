from sympy import simplify
<<<<<<< HEAD
from chain_complex import ChainComplex
=======
from cochain_complex_memoized_mod2 import CochainComplex
>>>>>>> ddc767ecb3cb817612baea2d1f4e82f9752866d5
from symbolic_tensor_ops import SymbolicTensor
import numpy as np
from collections import Counter

<<<<<<< HEAD
# This code is a stub for the actual implementation of the permutation cycle partition function.
# It relies on SageMath routines for chain and cochain complex computations
=======
>>>>>>> ddc767ecb3cb817612baea2d1f4e82f9752866d5

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


def test_boundary_conjugation(n: int, seed: int):
    A = SymbolicTensor((n, n))
    rng = np.random.Generator(np.random.PCG64(seed))
    perm = rng.permutation(n)
    P = np.eye(n)[perm]
    P = np.array(P, dtype=int)

    a_tensor = A.tensor
    b_tensor = P.T @ a_tensor @ P
    B = SymbolicTensor((n, n), tensor=b_tensor)

    boundary_a = A.bdry()
    boundary_b = B.bdry()

    diff_tensor = boundary_a.tensor - boundary_b.tensor
<<<<<<< HEAD

    cochain_complex = ChainComplex(shape=(n - 1, n - 1))
=======
    cochain_complex = CochainComplex(shape=(n - 1, n - 1))
>>>>>>> ddc767ecb3cb817612baea2d1f4e82f9752866d5

    nonzero_cocycles = 0
    all_pass = True
    for i in range(n - 1):
        for j in range(n - 1):
            expr = simplify(diff_tensor[i, j])
            if expr != 0:
                nonzero_cocycles += 1
                if not cochain_complex.is_cocycle(expr):
                    all_pass = False
                    break
        if not all_pass:
            break

    partition = permutation_cycle_partition(perm)
    return all_pass, nonzero_cocycles, partition


def run_batch_tests(n: int = 10, num_tests: int = 20):
    print(f"Testing boundary conjugation for {num_tests} random seeds (n={n}):")
    results = []
    for seed in range(num_tests):
        result, count, partition = test_boundary_conjugation(n, seed)
        results.append((seed, result, count, partition))
        print(f"Seed {seed:02d}: {'PASS' if result else 'FAIL'}   Nonzero cocycles: {count}   Partition: {partition}")
    total_pass = sum(1 for _, r, _, _ in results if r)
    print(f"Summary: {total_pass}/{num_tests} tests passed.")


if __name__ == "__main__":
    for n in range(3, 11):
        run_batch_tests(n, 50)
    for n in [23,31, 43]:
        run_batch_tests(n, 50)
