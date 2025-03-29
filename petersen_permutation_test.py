import sympy as sp
from sympy import simplify
from symbolic_tensor_ops import SymbolicTensor
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import random

# saved for SageMath 

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
    rng = np.random.default_rng(seed)
    return rng.choice([0, 1], size=(n, n), p=[1 - density, density])


def test_random_zero_one_cocycles(n=10, num_trials=20, density=0.5):

    print(f"\nTesting random 0-1 masks for {num_trials} trials (n={n}):")
    for _ in range(num_trials):
        A = SymbolicTensor((n, n))
        mask_a = random_zero_one_matrix(n, density)
        mask_b = random_zero_one_matrix(n, density)

        a_tensor = np.multiply(A.tensor, mask_a)
        b_tensor = np.multiply(A.tensor, mask_b)

        a_masked = SymbolicTensor((n, n), tensor=a_tensor)
        b_masked = SymbolicTensor((n, n), tensor=b_tensor)

        boundary_a = a_masked.bdry()
        boundary_b = b_masked.bdry()

        diff_tensor = boundary_a.tensor - boundary_b.tensor
        sp.pprint(f"diff_tensor:\n{diff_tensor}")    


def test_petersen_permutation_cocycles(n=10, num_trials=20):
    petersen_adjacency = kneser_petersen_adjacency()
    
    # defunct cochain_complex = CochainComplex(shape=(n - 1, n - 1))

    results = []
    partition_counter = defaultdict(int)

    print(f"Testing Petersen graph conjugation for {num_trials} random permutations (n={n}):")
    for trial in range(num_trials):
        A = SymbolicTensor((n, n))
        perm = random_permutation(n)
        partition = permutation_cycle_partition(perm)
        P = np.eye(n)[perm]
        P = np.array(P, dtype=int)

        a_tensor = np.multiply(A.tensor, petersen_adjacency)
        a = SymbolicTensor((n, n), tensor=a_tensor)
        b_tensor = P.T @ a.tensor @ P
        b = SymbolicTensor((n, n), tensor=b_tensor)


        boundary_a = a.bdry()
        boundary_b = b.bdry()

        diff_tensor = boundary_a.tensor - boundary_b.tensor

        results.append((trial, partition))
        partition_counter[partition] += 1

        print(f"Trial {trial:02d}:  Partition: {partition} diff_tensor:\n{diff_tensor}")

    print("Partition frequencies:")
    for partition, count in sorted(partition_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"Partition {partition}: {count} occurrences")


if __name__ == "__main__":
    test_petersen_permutation_cocycles(n=10, num_trials=10)
    test_random_zero_one_cocycles(n=10, num_trials=10)

