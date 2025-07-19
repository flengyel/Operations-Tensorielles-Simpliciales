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


from horn_map_reduce import compute_missing_indices_dask

def get_missing_indices(shape, j):
    """
    Delegate missing multi-indices computation to the Dask-based implementation.
    """
    # Use compute_missing_indices_dask to get the raw set
    raw = compute_missing_indices_dask(shape, j)
    # Intersect with normalized basis to ensure non-degenerate
    A = get_normalized_basis(shape, j)
    return set(raw).intersection(A)


def compute_homology_dimension(shape, j=0):
    """
    Compute H_n dimension as number of missing multi-indices in normalized basis.
    """
    A = get_normalized_basis(shape, j)
    missing = get_missing_indices(shape, j)
    # homology dimension = |missing|
    return len(missing)


if __name__ == '__main__':
    shapes = [(3,3), (4,4), (5,5), (3,3,3),(3,3,4),(3,4,4), (4,4,4),(4,4,5),(4,5,5), (5,5,5),(4,4,4,4)]
    for shape in shapes:
        dim = dimen(range_tensor(shape))
        for j in range(len(shape)):
            hom_dim = compute_homology_dimension(shape, j)
            print(f"shape={shape}, axis={j}, simplex_dim={dim}, homology_dim={hom_dim}")
