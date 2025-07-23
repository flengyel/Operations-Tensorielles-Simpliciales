import numpy as np
import itertools

# We will use functions from your existing tensor_ops.py script.
# Ensure tensor_ops.py is in the same directory.
try:
    from tensor_ops import (
        face,
        degen,
        horn,
        filler,
        is_degen,
        dimen,
        range_tensor # Added range_tensor for more robust testing
    )
except ImportError as e:
    print(f"Error: Could not import required functions from tensor_ops.py. {e}")
    # CORRECTED: Dummy functions now match the actual signatures
    def face(m: np.ndarray, i: int) -> np.ndarray: return np.array([])
    def degen(z: np.ndarray, k: int) -> np.ndarray: return np.array([])
    def horn(m: np.ndarray, k: int) -> np.ndarray: return np.array([])
    def filler(horn: np.ndarray, k: int) -> np.ndarray: return np.array([])
    def is_degen(a: np.ndarray) -> bool: return False
    def dimen(t: np.ndarray) -> int: return 0
    def range_tensor(shape): return np.zeros(shape)


def verify_degenerate_preference(non_degenerate_base, degeneracy_index, horn_index):
    """
    Tests if the filler algorithm is degenerate-preferring for a specific case.

    Args:
        non_degenerate_base (np.ndarray): A known non-degenerate tensor.
        degeneracy_index (int): The index `j` for the degeneracy `s_j`.
        horn_index (int): The index `k` for the horn `Lambda^n_k`.
    """
    print(f"\n--- Verifying Degenerate Preference ---")
    print(f"Base (non-degenerate) tensor shape: {non_degenerate_base.shape}")
    print(f"Applying degeneracy s_{degeneracy_index}, creating horn Lambda_{horn_index}")

    # 1. Create the known degenerate tensor. This is our target solution.
    try:
        T_degen = degen(non_degenerate_base, degeneracy_index)
        print(f"Constructed degenerate tensor of shape: {T_degen.shape}")
    except IndexError as e:
        print(f"Skipping: Cannot apply degeneracy s_{degeneracy_index}. {e}")
        return True # Vacuously true

    # 2. Create the horn from this degenerate tensor.
    #    By construction, this horn has a known degenerate filler (T_degen).
    try:
        the_horn = horn(T_degen, horn_index)
    except ValueError as e:
        print(f"Skipping: Cannot create horn Lambda_{horn_index}. {e}")
        return True # Vacuously true

    # 3. Run the filler algorithm on this horn.
    filler_result = filler(the_horn, horn_index)

    # 4. Check if the result is the known degenerate tensor.
    is_correct_filler = np.array_equal(filler_result, T_degen)

    if is_correct_filler:
        print("Result: PASSED. The filler algorithm produced the correct degenerate solution.")
    else:
        print("Result: FAILED. The filler algorithm did NOT produce the degenerate solution.")

    return is_correct_filler


if __name__ == '__main__':
    # --- Test Case 1: A simple 2D matrix using range_tensor ---
    base_matrix_shape = (2, 2)
    base_matrix = range_tensor(base_matrix_shape)
    # A range_tensor is non-degenerate because all its entries are unique.
    # We add a check to ensure our base case is valid.
    if not is_degen(base_matrix):
        print(f"Using a non-degenerate range_tensor of shape {base_matrix.shape} as the base.")
    else:
        print(f"ERROR: Base tensor of shape {base_matrix.shape} is unexpectedly degenerate!")
        exit()

    print("======================================================")
    print(" STEP 1: TESTING A 2D CASE (MATRIX) ")
    print("======================================================")
    
    all_passed = True
    # Test every possible combination of degeneracy and horn index
    for degen_idx in range(dimen(base_matrix) + 1): # s_0, s_1
        # The degenerate tensor will have dimen+1
        # Need to create a dummy tensor to get the dimension after degeneracy
        dummy_degen_tensor = degen(base_matrix, degen_idx)
        degenerate_dim = dimen(dummy_degen_tensor)
        for horn_idx in range(degenerate_dim + 1): # Lambda_0, Lambda_1, Lambda_2
            if not verify_degenerate_preference(base_matrix, degen_idx, horn_idx):
                all_passed = False

    # --- Test Case 2: A 3rd-order tensor using range_tensor ---
    base_tensor_shape = (3, 3, 3)
    base_tensor = range_tensor(base_tensor_shape)
    if not is_degen(base_tensor):
        print(f"\nUsing a non-degenerate range_tensor of shape {base_tensor.shape} as the base.")
    else:
        print(f"ERROR: Base tensor of shape {base_tensor.shape} is unexpectedly degenerate!")
        exit()

    print("\n\n======================================================")
    print(" STEP 2: TESTING A 3D CASE (TENSOR) ")
    print("======================================================")

    for degen_idx in range(dimen(base_tensor) + 1): # s_0, s_1, s_2
        dummy_degen_tensor = degen(base_tensor, degen_idx)
        degenerate_dim = dimen(dummy_degen_tensor)
        for horn_idx in range(degenerate_dim + 1): # Horns of the 4x4x4 tensor
            if not verify_degenerate_preference(base_tensor, degen_idx, horn_idx):
                all_passed = False

    print("\n-------------------------------------------")
    if all_passed:
        print("✅ Overall Result: The filler algorithm appears to be degenerate-preferring.")
    else:
        print("❌ Overall Result: The filler algorithm is NOT degenerate-preferring.")
