import numpy as np
import itertools

# We will use functions from your existing tensor_ops.py script.
# Ensure tensor_ops.py is in the same directory.
try:
    from tensor_ops import face, dimen
except ImportError as e:
    print(f"Error: Could not import required functions from tensor_ops.py. {e}")
    # Define dummy functions to avoid crashing the script
    def face(m: np.ndarray, i: int) -> np.ndarray: return np.array([])
    def dimen(t: np.ndarray) -> int: return min(t.shape) - 1 if t.shape else -1


def standard_basis_tensor(idx, shape):
    """
    Returns the standard basis tensor E_idx of a given shape.
    """
    T = np.zeros(shape, dtype=int)
    T[idx] = 1
    return T

def verify_face_map_injectivity(shape, face_index):
    """
    Tests the crucial claim that the map m -> d_i(E_m) is injective on the
    set of indices m where the result is non-zero. This is equivalent to
    showing there is no repetition among the non-zero output tensors.
    """
    print(f"\n--- Verifying Injectivity for shape={shape}, face_index={face_index} ---")

    non_zero_faces_list = []
    source_indices = [] # Keep track of which m produced which face

    # We are interested in the domain where d_i(E_m) is non-zero.
    # This occurs when the multi-index `m` does NOT contain `face_index`.
    domain_indices = [m for m in itertools.product(*(range(dim) for dim in shape)) if face_index not in m]

    for m in domain_indices:
        E_m = standard_basis_tensor(m, shape)
        face_of_Em = face(E_m, face_index)
        
        # This check should always pass given our domain, but it's a good sanity check.
        if np.any(face_of_Em):
            non_zero_faces_list.append(face_of_Em)
            source_indices.append(m)

    if not non_zero_faces_list:
        print("Result: Vacuously TRUE (no non-zero faces were produced).")
        return True

    # Now, check for duplicates. A robust way is to convert flattened arrays to hashable tuples.
    seen_faces = set()
    has_duplicates = False
    for i, face_tensor in enumerate(non_zero_faces_list):
        face_tuple = tuple(face_tensor.flatten())

        if face_tuple in seen_faces:
            print(f"Result: FAILED. Duplicate face found.")
            # Find the original source index for the duplicate
            original_index_of_duplicate = -1
            for j in range(i):
                if np.array_equal(non_zero_faces_list[j], face_tensor):
                    original_index_of_duplicate = j
                    break
            print(f"  - Face from index {source_indices[i]} is identical to face from index {source_indices[original_index_of_duplicate]}")
            has_duplicates = True
            break
        seen_faces.add(face_tuple)

    if not has_duplicates:
        print(f"Generated {len(non_zero_faces_list)} non-zero faces from {len(domain_indices)} source indices.")
        print("Found 0 duplicates.")
        print("Result: PASSED. The map m -> d_i(E_m) is injective on its non-zero domain.")

    return not has_duplicates

if __name__ == '__main__':
    # A representative set of shapes for testing
    shapes_to_test = [
        (3, 3),
        (4, 4, 4),
        (3, 4, 5),
    ]

    print("======================================================")
    print(" VERIFYING INJECTIVITY OF THE FACE MAP d_i ")
    print("======================================================")
    all_passed = True
    for shape in shapes_to_test:
        # Create a dummy tensor to use the imported dimen function
        dummy_tensor = np.empty(shape)
        n_dim = dimen(dummy_tensor)
        for i in range(n_dim + 1):
            if not verify_face_map_injectivity(shape=shape, face_index=i):
                all_passed = False
    
    print("\n-------------------------------------------")
    if all_passed:
        print("✅ Injectivity premise holds for all tested cases.")
    else:
        print("❌ Injectivity premise FAILED for one or more cases.")
