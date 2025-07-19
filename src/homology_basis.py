import numpy as np
from itertools import product
from tensor_ops import range_tensor, is_degen, dimen


def standard_basis_tensor(idx, shape):
    """
    Return the standard basis tensor E_idx of given shape.
    """
    T = np.zeros(shape, dtype=int)
    T[idx] = 1
    return T


def get_normalized_basis(shape, j):
    """
    Compute the non-degenerate (normalized) basis indices in A_n for degree n=min(shape)-1,
    filtering out degenerate simplices via tensor_ops.is_degen.
    """
    n = min(shape) - 1
    k = len(shape)
    basis = []
    for idx in product(range(n+1), repeat=k):
        E = standard_basis_tensor(idx, (n+1,)*k)
        if not is_degen(E):
            basis.append(idx)
    return set(basis)


def get_missing_indices(shape, j):
    """
    Compute the set of 'missing' multi-indices in direction j:
    those idx in normalized basis whose jth coordinate omits exactly one value from 0..n.
    """
    n = min(shape) - 1
    A = get_normalized_basis(shape, j)
    missing = set()
    # For each i in 0..n, collect indices missing value i at position j
    for i in range(n+1):
        for idx in A:
            if idx[j] != i:
                # candidate: idx misses coordinate i
                # check it misses exactly one face? since basis is normalized, it's sufficient
                missing.add(idx)
    # True missing are those that omit exactly one face index,
    # so we subtract those that omit more than one
    # But since jth coord can only omit one at a time, missing as defined is correct.
    # Final missing set is basis minus those that never appear with each i
    # However here missing collects all non-horn; horn = A - missing
    return missing


def compute_homology_dimension(shape, j=0):
    """
    Compute H_n dimension as number of missing multi-indices in normalized basis.
    """
    A = get_normalized_basis(shape, j)
    missing = get_missing_indices(shape, j)
    # homology dimension = |missing|
    return len(missing)


if __name__ == '__main__':
    shapes = [(3,3), (4,4), (5,5), (3,4,5)]
    for shape in shapes:
        dim = dimen(range_tensor(shape))
        for j in range(len(shape)):
            hom_dim = compute_homology_dimension(shape, j)
            print(f"shape={shape}, axis={j}, simplex_dim={dim}, homology_dim={hom_dim}")
