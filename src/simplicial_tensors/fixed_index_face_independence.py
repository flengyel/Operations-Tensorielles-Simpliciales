import numpy as np
import itertools

# We will use the 'face' function from your tensor_ops.py file.
# Ensure tensor_ops.py is in the same directory as this script.
try:
    from .tensor_ops import face
except ImportError:
    print("Error: Could not import 'face' from tensor_ops.py.")
    print("Please ensure tensor_ops.py is in the same directory.")
    # Define a dummy function to avoid crashing the script
    def face(m, i):
        print("DUMMY FACE FUNCTION: tensor_ops.py not found.")
        return np.array([])

def standard_basis_tensor(idx, shape):
    """
    Returns the standard basis tensor E_idx of a given shape.
    This is a tensor with a 1 at the specified index and 0s elsewhere.
    """
    T = np.zeros(shape, dtype=int)
    T[idx] = 1
    return T

def test_face_map_linear_independence(shape, face_index):
    """
    Tests the claim that for a fixed face index `i`, the set of all
    non-zero images {d_i(E_m)} is linearly independent.

    Args:
        shape (tuple): The shape of the tensor space to test.
        face_index (int): The fixed face index `i` to apply.
    """
    print(f"\n--- Testing for shape={shape}, face_index={face_index} ---")

    # 1. Generate all non-zero faces d_i(E_m)
    non_zero_faces = []
    for m in itertools.product(*(range(s) for s in shape)):
        # Create the standard basis tensor for this index
        E_m = standard_basis_tensor(m, shape)
        
        # Compute the i-th face
        face_of_Em = face(E_m, face_index)
        
        # Collect the non-zero results
        if np.any(face_of_Em):
            non_zero_faces.append(face_of_Em.flatten())

    if not non_zero_faces:
        print("Result: Vacuously TRUE (no non-zero faces were produced).")
        return True

    # 2. Check for linear independence
    # Create a matrix where each row is a flattened face tensor
    matrix_of_faces = np.array(non_zero_faces)
    
    # The number of vectors in our set
    num_vectors = matrix_of_faces.shape[0]
    
    # Compute the rank of the matrix
    matrix_rank = np.linalg.matrix_rank(matrix_of_faces)

    print(f"Number of non-zero face tensors generated: {num_vectors}")
    print(f"Rank of the matrix formed by these tensors: {matrix_rank}")

    # 3. The set is linearly independent if the rank equals the number of vectors
    is_independent = (matrix_rank == num_vectors)
    
    if is_independent:
        print("Result: PASSED. The set of non-zero face tensors is linearly independent.")
    else:
        print("Result: FAILED. The set is linearly dependent.")
        
    return is_independent

def main():
    # Test Case 1: A simple 3x3 matrix
    test_face_map_linear_independence(shape=(3, 3), face_index=0)
    test_face_map_linear_independence(shape=(3, 3), face_index=1)
    
    # Test Case 2: A 3rd-order tensor
    # This is a more robust test of the principle
    test_face_map_linear_independence(shape=(4, 4, 4), face_index=0)
    test_face_map_linear_independence(shape=(4, 4, 4), face_index=2)
    
    # Test Case 3: A non-constant shape tensor
    test_face_map_linear_independence(shape=(3, 4, 5), face_index=1)
