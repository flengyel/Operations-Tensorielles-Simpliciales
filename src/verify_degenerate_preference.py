import numpy as np
import itertools
from typing import Tuple, List, Union, Any
from tensor_ops import (
        degen,
        horn,
        filler,
        dimen,
        standard_basis_matrix # Using the 2D version for a clear example
    )



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
        # Optional: show the difference for debugging
        # print("Expected (Degenerate Tensor):\n", T_degen)
        # print("Actual (Filler Result):\n", filler_result)

    return is_correct_filler


if __name__ == '__main__':
    # --- Test Case 1: A simple 2D matrix ---
    # Start with a non-degenerate 2x2 matrix (a 1-simplex)
    base_matrix = standard_basis_matrix(2, 2, 0, 1) # E_(0,1) is non-degenerate
    
    # Let's make a degenerate 3x3 matrix from it by applying s_0
    # The result should be a 3x3 matrix. Its dimension n is 2.
    # We can test any horn from k=0 to 2.
    
    print("======================================================")
    print(" STEP 1: TESTING A 2D CASE (MATRIX) ")
    print("======================================================")
    
    all_passed = True
    # Test every possible combination of degeneracy and horn index
    for degen_idx in range(dimen(base_matrix) + 1): # s_0, s_1
        # The degenerate tensor will have dimen+1
        degenerate_dim = dimen(degen(base_matrix, degen_idx))
        for horn_idx in range(degenerate_dim + 1): # Lambda_0, Lambda_1, Lambda_2
            if not verify_degenerate_preference(base_matrix, degen_idx, horn_idx):
                all_passed = False

    # --- Test Case 2: A 3rd-order tensor ---
    # We need a general standard_basis_tensor function for this
    def standard_basis_tensor(idx, shape):
        T = np.zeros(shape, dtype=int)
        T[idx] = 1
        return T

    print("\n\n======================================================")
    print(" STEP 2: TESTING A 3D CASE (TENSOR) ")
    print("======================================================")

    # Start with a non-degenerate 3x3x3 tensor (a 2-simplex)
    # E_(0,1,2) is non-degenerate
    base_tensor = standard_basis_tensor((0, 1, 2), (3, 3, 3))

    for degen_idx in range(dimen(base_tensor) + 1): # s_0, s_1, s_2
        degenerate_dim = dimen(degen(base_tensor, degen_idx))
        for horn_idx in range(degenerate_dim + 1): # Horns of the 4x4x4 tensor
            if not verify_degenerate_preference(base_tensor, degen_idx, horn_idx):
                all_passed = False

    print("\n-------------------------------------------")
    if all_passed:
        print("✅ Overall Result: The filler algorithm appears to be degenerate-preferring.")
    else:
        print("❌ Overall Result: The filler algorithm is NOT degenerate-preferring.")
