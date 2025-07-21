import numpy as np
import itertools

# We will use functions from your existing scripts.
# Ensure tensor_ops.py and horn_map_reduce.py are in the same directory.
try:
    from tensor_ops import face, dimen
    from horn_map_reduce import compute_missing_indices_dask
except ImportError as e:
    print(f"Error: Could not import required functions. {e}")
    print("Please ensure tensor_ops.py and horn_map_reduce.py are in the same directory.")
    # Define dummy functions to avoid crashing the script
    def face(m, i): return np.array([])
    def dimen(t): return min(t.shape) - 1 if t.shape else -1
    # CORRECTED DUMMY FUNCTION SIGNATURE:
    def compute_missing_indices_dask(shape, horn_j): return set()


def standard_basis_tensor(idx, shape):
    """
    Returns the standard basis tensor E_idx of a given shape.
    """
    T = np.zeros(shape, dtype=int)
    T[idx] = 1
    return T

def test_face_map_linear_independence(shape, face_index):
    """
    Tests the claim that for a fixed face index `i`, the set of all
    non-zero images {d_i(E_m)} is linearly independent.
    """
    print(f"\n--- Testing Linear Independence for shape={shape}, face_index={face_index} ---")

    # 1. Generate all non-zero faces d_i(E_m)
    non_zero_faces = []
    # Iterate over all multi-indices for the given shape
    for m in itertools.product(*(range(dim) for dim in shape)):
        E_m = standard_basis_tensor(m, shape)
        face_of_Em = face(E_m, face_index)
        if np.any(face_of_Em):
            non_zero_faces.append(face_of_Em.flatten())

    if not non_zero_faces:
        print("Result: Vacuously TRUE (no non-zero faces were produced).")
        return True

    # 2. Check for linear independence by computing the matrix rank
    matrix_of_faces = np.array(non_zero_faces)
    num_vectors = matrix_of_faces.shape[0]
    matrix_rank = np.linalg.matrix_rank(matrix_of_faces)

    print(f"Number of non-zero face tensors generated: {num_vectors}")
    print(f"Rank of the matrix formed by these tensors: {matrix_rank}")

    is_independent = (matrix_rank == num_vectors)
    
    if is_independent:
        print("Result: PASSED. The set of non-zero face tensors is linearly independent.")
    else:
        print("Result: FAILED. The set is linearly dependent.")
        
    return is_independent

def verify_constraints(shape, horn_j):
    """
    Verifies that the set of "missing indices" is identical to the set of
    indices `m` where d_i(E_m) is zero for all faces `i` in the horn.
    """
    print(f"\n--- Verifying Constraints for shape={shape}, horn_j={horn_j} ---")
    
    # Create a dummy tensor to use the imported dimen function
    dummy_tensor = np.empty(shape)
    n = dimen(dummy_tensor)

    if not (0 <= horn_j <= n):
        print(f"Skipping: horn_j={horn_j} is out of bounds for n={n}.")
        return True # Return True to not fail the overall check

    # 1. Calculate the set of missing indices using the combinatorial definition.
    expected_indices = compute_missing_indices_dask(shape, horn_j)
    print(f"Combinatorial method (Dask) found {len(expected_indices)} missing indices.")

    # 2. Calculate the set of indices that are "killed" by every face map in the horn.
    horn_face_indices = [i for i in range(n + 1) if i != horn_j]
    
    # Iterate over all multi-indices for the given shape
    all_indices = set(itertools.product(*(range(dim) for dim in shape)))
    indices_surviving_constraints = all_indices

    for i in horn_face_indices:
        Z_i = set() # The set of indices m where d_i(E_m) = 0
        for m in all_indices:
            E_m = standard_basis_tensor(m, shape)
            face_of_Em = face(E_m, i)
            if not np.any(face_of_Em): # Check if the result is the zero tensor
                Z_i.add(m)
        
        indices_surviving_constraints = indices_surviving_constraints.intersection(Z_i)

    print(f"Constraint method found {len(indices_surviving_constraints)} indices surviving all constraints.")

    # 3. Compare the two sets.
    if indices_surviving_constraints == expected_indices:
        print("Result: PASSED. The two sets are identical.")
        return True
    else:
        print("Result: FAILED. The sets are not identical.")
        return False


if __name__ == '__main__':
    # A representative set of shapes for testing
    shapes_to_test = [
        (3, 3),
        (4, 4, 4),
        (3, 4, 5),
        (4, 4, 4, 4),
        (2,2)
    ]

    print("======================================================")
    print(" STEP 1: VERIFYING THE LINEAR INDEPENDENCE PREMISE ")
    print("======================================================")
    li_passed = True
    for shape in shapes_to_test:
        # Create a dummy tensor to use the imported dimen function
        dummy_tensor = np.empty(shape)
        n_dim = dimen(dummy_tensor)
        for i in range(n_dim + 1):
            if not test_face_map_linear_independence(shape=shape, face_index=i):
                li_passed = False
    
    print("\n-------------------------------------------")
    if li_passed:
        print("✅ Linear independence premise holds for all tested cases.")
    else:
        print("❌ Linear independence premise FAILED for one or more cases.")


    print("\n\n======================================================")
    print(" STEP 2: VERIFYING THE CONSEQUENCE (CONSTRAINT CHECK) ")
    print("======================================================")
    constraints_passed = True
    for shape in shapes_to_test:
        # Create a dummy tensor to use the imported dimen function
        dummy_tensor = np.empty(shape)
        n_dim = dimen(dummy_tensor)
        for j in range(n_dim + 1):
            if not verify_constraints(shape=shape, horn_j=j):
                constraints_passed = False
    
    print("\n-------------------------------------------")
    if constraints_passed:
        print("✅ All tested shapes passed the constraint verification.")
    else:
        print("❌ Some shapes failed the constraint verification.")
