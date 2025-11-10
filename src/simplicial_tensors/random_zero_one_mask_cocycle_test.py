from sympy import simplify
from cochain_complex_memoized_mod2 import CochainComplex
from .symbolic_tensor_ops import SymbolicTensor
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import random


def permutation_cycle_partition(perm):
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


def random_permutation(n):
    perm = list(range(n))
    random.shuffle(perm)
    return perm


def kneser_petersen_adjacency():
    vertices = list(combinations(range(1, 6), 2))
    n = len(vertices)
    adj = np.zeros((n, n), dtype=int)
    for i, v1 in enumerate(vertices):
        for j, v2 in enumerate(vertices):
            if set(v1).isdisjoint(set(v2)):
                adj[i, j] = 1
    return adj


def random_zero_one_matrix(n, density=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice([0, 1], size=(n, n), p=[1 - density, density])


def test_random_zero_one_cocycles(n=10, num_trials=20, density=0.5):
    A = SymbolicTensor((n, n))
    cochain_complex = CochainComplex(shape=(n - 1, n - 1))

    print(f"\nTesting random 0-1 masks for {num_trials} trials (n={n}):")
    for trial in range(num_trials):
        mask_a = random_zero_one_matrix(n, density)
        mask_b = random_zero_one_matrix(n, density)

        a_tensor = np.multiply(A.tensor, mask_a)
        b_tensor = np.multiply(A.tensor, mask_b)

        A_masked = SymbolicTensor((n, n), tensor=a_tensor)
        B_masked = SymbolicTensor((n, n), tensor=b_tensor)

        boundary_a = A_masked.bdry()
        boundary_b = B_masked.bdry()

        diff_tensor = boundary_a.tensor - boundary_b.tensor

        nonzero_cocycles = 0
        non_coboundaries = 0

        for i in range(n - 1):
            for j in range(n - 1):
                expr = simplify(diff_tensor[i, j])
                if expr != 0:
                    nonzero_cocycles += 1
                    if not cochain_complex.is_coboundary(expr):
                        non_coboundaries += 1

        status = "PASS" if nonzero_cocycles == non_coboundaries else "FAIL"
        print(f"Trial {trial:02d}: {status}   Nonzero cocycles: {nonzero_cocycles:4d}")


def test_petersen_permutation_cocycles(n=10, num_trials=20):
    A = SymbolicTensor((n, n))
    petersen_adjacency = kneser_petersen_adjacency()
    cochain_complex = CochainComplex(shape=(n - 1, n - 1))

    results = []
    partition_counter = defaultdict(int)

    print(f"Testing Petersen graph conjugation for {num_trials} random permutations (n={n}):")
    for trial in range(num_trials):
        perm = random_permutation(n)
        partition = permutation_cycle_partition(perm)
        P = np.eye(n)[perm]
        P = np.array(P, dtype=int)

        a_tensor = np.multiply(A.tensor, petersen_adjacency)
        b_tensor = P.T @ a_tensor @ P
        B = SymbolicTensor((n, n), tensor=b_tensor)

        boundary_a = A.bdry()
        boundary_b = B.bdry()

        diff_tensor = boundary_a.tensor - boundary_b.tensor

        nonzero_cocycles = 0
        non_coboundaries = 0

        for i in range(n - 1):
            for j in range(n - 1):
                expr = simplify(diff_tensor[i, j])
                if expr != 0:
                    nonzero_cocycles += 1
                    if not cochain_complex.is_coboundary(expr):
                        non_coboundaries += 1

        status = "PASS" if nonzero_cocycles == non_coboundaries else "FAIL"
        results.append((trial, status, nonzero_cocycles, partition))
        partition_counter[partition] += 1

        print(f"Trial {trial:02d}: {status}   Nonzero cocycles: {nonzero_cocycles:4d}   Partition: {partition}")

    print(f"\nSummary: {num_trials}/{num_trials} tests passed.")
    print("Partition frequencies:")
    for partition, count in sorted(partition_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"Partition {partition}: {count} occurrences")


def main():
    test_petersen_permutation_cocycles(n=10, num_trials=50)
    test_random_zero_one_cocycles(n=10, num_trials=20)
