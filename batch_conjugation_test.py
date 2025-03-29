
from sympy import simplify
from cochain_complex import CochainComplex
from symbolic_tensor_ops import SymbolicTensor
import numpy as np

def test_boundary_conjugation(n: int, seed: int) -> bool:
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
    cochain_complex = CochainComplex(shape=(n - 1, n - 1))

    for i in range(n - 1):
        for j in range(n - 1):
            expr = simplify(diff_tensor[i, j])
            if expr != 0 and not cochain_complex.is_cocycle(expr):
                return False
    return True

def run_batch_tests(n: int = 10, num_tests: int = 20):
    print(f"Testing boundary conjugation for {num_tests} random seeds (n={n}):")
    results = []
    for seed in range(num_tests):
        result = test_boundary_conjugation(n, seed)
        results.append((seed, result))
        print(f"Seed {seed:02d}: {'PASS' if result else 'FAIL'}")
    total_pass = sum(1 for _, r in results if r)
    print(f"Summary: {total_pass}/{num_tests} tests passed.")

if __name__ == "__main__":
    run_batch_tests(10,100)
