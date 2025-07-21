import numpy as np
import itertools


# We will use functions from your existing scripts.
# Ensure tensor_ops.py and horn_map_reduce.py are in the same directory.
try:
    from tensor_ops import face
    from horn_map_reduce import compute_missing_indices_dask
except ImportError as e:
    print(f"Error: Could not import required functions. {e}")
    print("Please ensure tensor_ops.py and horn_map_reduce.py are in the same directory.")
    # Define dummy functions to avoid crashing the script
    def face(m, i): return np.array([])
    def dimen(t): return 0
    def compute_missing_indices_dask(shape, horn_j): return set()


def standard_basis_tensor(idx, shape):
    """
    Returns the standard basis tensor E_idx of a given shape.
    """
    T = np.zeros(shape, dtype=int)
    T[idx] = 1
    return T

def verify_constraints(shape, horn_j):
    """
    Verifies that the set of "missing indices" is identical to the set of
    indices `m` where d_i(E_m) is zero for all faces `i` in the horn.
    This provides a computational test of the proof's core mechanism.
    """
    print(f"\n--- Verifying Constraints for shape={shape}, horn_j={horn_j} ---")
    
    n = min(shape) - 1
    if not (0 <= horn_j <= n):
        print(f"Skipping: horn_j={horn_j} is out of bounds for n={n}.")
        return

    # 1. Calculate the set of missing indices using the combinatorial definition.
    # This is the set we expect to find.
    expected_indices = compute_missing_indices_dask(shape, horn_j)
    print(f"Combinatorial method (Dask) found {len(expected_indices)} missing indices.")

    # 2. Calculate the set of indices that are "killed" by every face map in the horn.
    horn_face_indices = [i for i in range(n + 1) if i != horn_j]
    
    # Start with the set of all possible indices. We will pare it down.
    all_indices = set(itertools.product(*(range(s) for s in shape)))
    indices_surviving_constraints = all_indices

    # For each face in the horn, find the indices that are sent to zero
    # and intersect our running set with them.
    for i in horn_face_indices:
        Z_i = set() # The set of indices m where d_i(E_m) = 0
        for m in all_indices:
            E_m = standard_basis_tensor(m, shape)
            face_of_Em = face(E_m, i)
            if not np.any(face_of_Em): # Check if the result is the zero tensor
                Z_i.add(m)
        
        # Update the set of survivors by intersecting with the current zero set
        indices_surviving_constraints = indices_surviving_constraints.intersection(Z_i)

    print(f"Constraint method found {len(indices_surviving_constraints)} indices surviving all constraints.")

    # 3. Compare the two sets. They must be identical for the proof to hold.
    if indices_surviving_constraints == expected_indices:
        print("Result: PASSED. The two sets are identical.")
        return True
    else:
        print("Result: FAILED. The sets are not identical.")
        # To help debug, show the differences
        missed_by_constraint = expected_indices - indices_surviving_constraints
        extra_in_constraint = indices_surviving_constraints - expected_indices
        if missed_by_constraint:
            print(f"  - Missing indices not found by constraint method: {missed_by_constraint}")
        if extra_in_constraint:
            print(f"  - Indices found by constraint method that are not missing: {extra_in_constraint}")
        return False


if __name__ == '__main__':
    # Test cases
    shapes_to_test = [
        (3, 3),
        (4, 4, 4),
        (3, 4, 5),
        (4, 4, 4, 4)
    ]

    all_passed = True
    for shape in shapes_to_test:
        n_dim = min(shape) - 1
        # Test for a representative horn index, e.g., j=0
        if not verify_constraints(shape=shape, horn_j=0):
            all_passed = False
        # Test for another horn index to be thorough
        if n_dim > 0:
            if not verify_constraints(shape=shape, horn_j=1):
                all_passed = False
    
    print("\n-------------------------------------------")
    if all_passed:
        print("✅ All tested shapes passed the constraint verification.")
    else:
        print("❌ Some shapes failed the constraint verification.")
