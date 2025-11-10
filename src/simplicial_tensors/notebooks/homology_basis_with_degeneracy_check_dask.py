import numpy as np
from itertools import product
# These functions are imported from your tensor_ops.py file.
from ..tensor_ops import range_tensor, dimen, is_degen

# We import the robust function for finding missing indices from your other script.
from ..horn_map_reduce import compute_missing_indices_dask


def standard_basis_tensor(idx, shape):
    """
    Return the standard basis tensor E_idx of given shape.
    """
    T = np.zeros(shape, dtype=int)
    T[idx] = 1
    return T


def get_non_degenerate_indices(shape):
    """
    Computes the set of all indices for a given shape that correspond to
    non-degenerate standard basis tensors.
    """
    non_degenerate_indices = set()
    # Iterate through ALL possible indices for the given tensor shape.
    for idx in product(*(range(s) for s in shape)):
        # Create the standard basis tensor for this index.
        E_idx = standard_basis_tensor(idx, shape)
        # Check if this tensor is non-degenerate and add its index to the set.
        if not is_degen(E_idx):
            non_degenerate_indices.add(idx)
    return non_degenerate_indices


def demonstrate_degeneracy(shape):
    """
    Counts the total, degenerate, and non-degenerate standard basis
    tensors for a given shape to show that normalization is non-trivial.
    """
    total_tensors = np.prod(shape)
    non_degenerate_indices = get_non_degenerate_indices(shape)
    non_degenerate_count = len(non_degenerate_indices)
    degenerate_count = total_tensors - non_degenerate_count

    print(f"\n--- Degeneracy Analysis for shape={shape} ---")
    print(f"Total standard basis tensors: {total_tensors}")
    print(f"Degenerate tensors: {degenerate_count}")
    print(f"Non-degenerate tensors (survive normalization): {non_degenerate_count}")
    if degenerate_count > 0:
        print("Conclusion: Normalization is a non-trivial filtering process.")
    else:
        print("Conclusion: All basis tensors are non-degenerate for this shape.")


def compute_homology_dimension(shape, j=0):
    """
    Computes the dimension of the relative homology group H_n(V, L)
    by finding the intersection of "missing indices" and indices that
    correspond to non-degenerate standard basis tensors.
    """
    # 1. Get the set of indices predicted to be the generators.
    missing_indices = compute_missing_indices_dask(shape, j)

    # 2. Get the set of all indices that survive normalization (are non-degenerate).
    surviving_indices = get_non_degenerate_indices(shape)

    # 3. The true basis for the relative homology corresponds to the indices
    #    that are BOTH missing from the horn AND non-degenerate.
    homology_basis_indices = missing_indices.intersection(surviving_indices)

    return len(homology_basis_indices)


def main():
    # A list of shapes to test.
    shapes_to_test = [
        (3, 3),
        (3, 3, 3),
        (4, 4, 4),
    ]
    
    print("--- Demonstrating that Normalization is Non-Trivial ---")
    for shape in shapes_to_test:
        demonstrate_degeneracy(shape)
    
    print("\n\n--- Running Predicted Homology Dimension Calculation ---")
    # A comprehensive list of shapes to test the corrected logic.
    shapes_to_test_full = [
        (3, 3), (4, 4), (5, 5), (3, 3, 3), (3, 3, 4), (3, 4, 4),
        (4, 4, 4), (4, 4, 5), (4, 5, 5), (5, 5, 5), (4, 4, 4, 4),
        (4, 4, 4, 5), (4, 4, 5, 5), (4, 5, 5, 5), (5, 5, 5, 5)
    ]
    for shape in shapes_to_test_full:
        try:
            n_dim = dimen(range_tensor(shape))
            j_horn = 0
            hom_dim = compute_homology_dimension(shape, j_horn)
            print(f"shape={shape}, n_dim={n_dim}, j={j_horn} -> predicted_homology_dim={hom_dim}")
        except Exception as e:
            print(f"Could not compute for shape={shape}. Error: {e}")
