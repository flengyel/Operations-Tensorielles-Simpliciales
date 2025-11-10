import numpy as np
from itertools import product
# These functions are imported from your tensor_ops.py file.
# is_degen is crucial for the verification step.
from ..tensor_ops import range_tensor, dimen, is_degen

# We import the robust function for finding missing indices.
from ..horn_map_reduce import compute_missing_indices_dask


def standard_basis_tensor(idx, shape):
    """
    Return the standard basis tensor E_idx of given shape.
    This is a tensor with a 1 at the specified index and 0s elsewhere.
    """
    T = np.zeros(shape, dtype=int)
    T[idx] = 1
    return T


def full_homology_verification(shape, j=0):
    """
    Computes the relative homology dimension by performing a full verification:
    1. Checks that basis tensors for "missing indices" are non-degenerate.
    2. Checks that ALL OTHER standard basis tensors are degenerate.
    This provides a rigorous computational test of the conjecture.
    """
    # 1. Get the set of indices predicted by the conjecture to form the basis.
    predicted_basis_indices = compute_missing_indices_dask(shape, j)
    
    found_homology_generators = 0
    unexpected_survivors = 0

    # 2. Iterate through ALL possible indices for the given tensor shape.
    for idx in product(*(range(s) for s in shape)):
        # Create the standard basis tensor for this index.
        E_idx = standard_basis_tensor(idx, shape)

        # 3. Check if this tensor survives normalization (is non-degenerate).
        if not is_degen(E_idx):
            # The tensor is non-degenerate. Now we check if the theory predicted this.
            if idx in predicted_basis_indices:
                # This is expected. It's a non-degenerate tensor whose index is "missing".
                # This corresponds to a true homology generator.
                found_homology_generators += 1
            else:
                # This is a failure of the conjecture. We found a non-degenerate
                # tensor that should not exist in the normalized complex's basis.
                unexpected_survivors += 1
                if unexpected_survivors < 5: # Report first few errors
                    print(f"FATAL ERROR for shape {shape}: Found unexpected non-degenerate tensor at index {idx}.")

    # 4. Final validation checks.
    if unexpected_survivors > 0:
        print(f"--> VERIFICATION FAILED for shape {shape}: Found {unexpected_survivors} unexpected non-degenerate basis tensors.")
        return None # Indicates failure

    if found_homology_generators != len(predicted_basis_indices):
        print(f"--> VERIFICATION FAILED for shape {shape}: Expected {len(predicted_basis_indices)} non-degenerate generators, but found {found_homology_generators}.")
        return None # Indicates failure

    # If we pass all checks, the dimension is the number of generators we found.
    return found_homology_generators


def main():
    # A comprehensive list of shapes to test the corrected logic.
    shapes_to_test = [
        (3, 3),
        (4, 4),
        (3, 3, 3),
        (3, 3, 4),
        (4, 4, 4),
        (4, 4, 4, 4),
        # Add more complex shapes to be thorough
        (3, 4, 5),
        (4, 4, 5),
    ]
    
    print("--- Running Homology Dimension Calculation with Full Verification ---")
    all_passed = True
    for shape in shapes_to_test:
        try:
            n_dim = dimen(range_tensor(shape))
            # The number of horns is n_dim + 1. We test for j=0 as a representative case.
            j_horn = 0
            print(f"\nVerifying shape={shape} (n_dim={n_dim}, j={j_horn})...")
            hom_dim = full_homology_verification(shape, j_horn)
    
            if hom_dim is not None:
                print(f"--> VERIFICATION PASSED: homology_dim={hom_dim}")
            else:
                all_passed = False
    
        except Exception as e:
            print(f"Could not compute for shape={shape}. Error: {e}")
            all_passed = False
    
    print("\n-------------------------------------------")
    if all_passed:
        print("✅ All tested shapes passed the full verification.")
    else:
        print("❌ Some shapes failed the full verification.")
    
