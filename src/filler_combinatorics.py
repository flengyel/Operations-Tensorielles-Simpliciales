#    Opérations Tensorielles Simpliciales
#    Simplicial Operations on Matrices and Hypermatrices
#    filler_combinatorics.py
#    
#    Copyright (C) 2021-2025 Florian Lengyel
#    Email: florian.lengyel at cuny edu, florian.lengyel at gmail
#    Website: https://github.com/flengyel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import sympy as sp
import numpy as np
import itertools
import math # For math.factorial
from symbolic_tensor_ops import SymbolicTensor
from tensor_ops import n_hypergroupoid_conjecture


# Helper, can be standalone or a method of SymbolicTensor
def get_all_local_indices_for_shape(shape_tuple: tuple):
    if not shape_tuple:
        return iter(())
    iterators = [range(s) for s in shape_tuple]
    return itertools.product(*iterators)

# --- Corrected Test Function 1 ---
def validate_filler_conjecture(shape: tuple):
    """
    Validates parts of the n-Hypergroupoid Conjecture by comparing the prediction
    of the conjecture function with the observed behavior of the filler.
    Focuses on uniqueness, missing indices, and properties of (t - t_prime).
    """
    print(f"\n--- Validating Filler Conjecture for Shape {shape} ---")
    t = SymbolicTensor(shape=shape, init_type='range') 
    N = t.dimen()
    k_order = t.order()

    print(f"Tensor Info: Order k={k_order}, Dimension N={N}")

    # Get the conjecture's prediction for this shape
    # Assuming n_hypergroupoid_conjecture is imported and takes shape
    conjecture_predicts_unique = n_hypergroupoid_conjecture(t.shape, verbose=False)
    print(f"Conjecture function prediction for unique filler: {conjecture_predicts_unique}")

    if N < 1: 
        print(f"Skipping filler-based validation for shape {shape} as dimension N={N} < 1.")
        # For N=0, even if filler doesn't apply, s_indices can be checked
        if N == 0:
            all_t_indices_n0 = set(np.ndindex(t.shape))
            for j_missing_n0 in range(N + 1): # j_missing_n0 will be 0
                print(f"  For N=0, Horn j_missing={j_missing_n0}:")
                s_indices_n0 = all_t_indices_n0 # Horn is empty, all original entries are "missing"
                print(f"    s_indices (entries of t not covered by empty horn): {len(s_indices_n0)} entries.")
                
                is_prop_case_n0 = (k_order == N) and all(s == (N + 1) for s in t.shape)
                if is_prop_case_n0: # e.g. N=0, k_order=0, shape=(1) - check if this case makes sense for prop.
                    expected_missing_count_n0 = math.factorial(N if N >=0 else 0) 
                    if len(s_indices_n0) == expected_missing_count_n0:
                         print(f"    OK: len(s_indices)={len(s_indices_n0)} matches N!={N}! = {expected_missing_count_n0} (Prop. case N=0).")
                    else:
                         print(f"    FAIL (Prop. N! for N=0): Expected {expected_missing_count_n0}, Got {len(s_indices_n0)}.")
            # If conjecture_predicts_unique for N=0 (based on k_order < N),
            # and no filler is run, we can't compare observed.
            # The main test logic below handles N >= 1.
        return True # Successfully processed/skipped N < 1 based on current logic.

    all_t_indices = set(np.ndindex(t.shape))
    all_tests_passed_for_shape = True # Will be set to False on direct contradictions
    original_axes_t = t._dims() 

    for j_missing in range(N + 1): 
        print(f"\n  Testing Horn j_missing={j_missing}:")
        
        horn_faces_list = t.horn(j_missing)

        indices_covered_by_horn_faces = set()
        for p_face_val, face_tensor_in_horn in enumerate(horn_faces_list):
            if p_face_val == j_missing: 
                continue 
            
            temp_original_axes_for_face_construction = [
                np.delete(original_axes_t[ax_idx], p_face_val) for ax_idx in range(t.order())
            ]
            for face_local_idx in np.ndindex(face_tensor_in_horn.shape):
                original_idx_tuple = tuple(
                    temp_original_axes_for_face_construction[dim_idx][face_local_idx[dim_idx]]
                    for dim_idx in range(t.order())
                )
                indices_covered_by_horn_faces.add(original_idx_tuple)
        
        s_indices = all_t_indices - indices_covered_by_horn_faces
        print(f"    s_indices (indices of t not covered by horn faces): {len(s_indices)} entries.")

        # Check N! for the specific "Proposition case"
        # Proposition: k_order == N and shape is (N+1, ..., N+1)
        is_prop_case = (k_order == N) and all(s_dim == (N + 1) for s_dim in t.shape)
        if is_prop_case:
            expected_N_factorial = math.factorial(N)
            if len(s_indices) == expected_N_factorial:
                print(f"    OK (Proposition N!): len(s_indices)={len(s_indices)} matches N!={N}! = {expected_N_factorial}.")
            else:
                print(f"    FAIL (Proposition N!): For this case (k_order=N, shape=(N+1,...)), expected {expected_N_factorial} differing terms, Got {len(s_indices)}.")
                # This check is on s_indices. If filler behaves, len(nonzero_diff_indices) will also be N!
                # For now, this part of "conjecture" is about s_indices count for this specific shape.
                # all_tests_passed_for_shape = False # Not necessarily a failure of filler if s_indices itself is the issue.

        t_prime = t.filler(horn_faces_list, j_missing) 
        diff_tensor = t - t_prime

        nonzero_diff_indices = set()
        for idx in np.ndindex(diff_tensor.shape):
            val_expr = diff_tensor.tensor[idx] 
            if sp.simplify(val_expr) != 0:
                nonzero_diff_indices.add(idx)

        print(f"    Non-zero entries in (t - t_prime): {nonzero_diff_indices}")
        # Check if the conjecture's prediction on uniqueness matches the observed behavior         
        print(f"    Number of non-zero entries in (t - t_prime): {len(nonzero_diff_indices)}")
        observed_is_unique = not nonzero_diff_indices # True if diff_tensor is zero

        # Main check: Does observation match conjecture's prediction?
        if conjecture_predicts_unique == observed_is_unique:
            print(f"    OK: Observation on uniqueness matches conjecture's prediction.")
            if conjecture_predicts_unique: # Both predict unique, and observed unique (diff is zero)
                print(f"        Details: Predicted UNIQUE, Observed UNIQUE (t - t_prime is zero).")
            else: # Both predict non-unique, and observed non-unique (diff is non-zero)
                print(f"        Details: Predicted NON-UNIQUE, Observed NON-UNIQUE (t - t_prime is non-zero).")
                # Further checks for non-unique case from conjecture:
                # 1. Are all non-zero diffs located at s_indices?
                if nonzero_diff_indices == s_indices:
                    print(f"        OK: Set of non-zero indices in (t - t_prime) matches s_indices.")
                else:
                    print(f"        FAIL: Set of non-zero indices in (t - t_prime) ({len(nonzero_diff_indices)}) != s_indices ({len(s_indices)}).")
                    print(f"          Non-zero in diff: {nonzero_diff_indices}")
                    print(f"          Expected (s_indices): {s_indices}")
                    all_tests_passed_for_shape = False
                
                # 2. If it's the proposition case, did len(nonzero_diff_indices) also match N!?
                if is_prop_case:
                    expected_N_factorial = math.factorial(N)
                    if len(nonzero_diff_indices) != expected_N_factorial:
                         print(f"        NOTE (Proposition N! for non-zero diffs): Expected {expected_N_factorial} non-zero diffs, Got {len(nonzero_diff_indices)}.")
                         # This could be a failure if the N! applies to count of non-zero diffs too.
                         # If len(s_indices) was already N!, then this is covered by the check above.


        else: # Mismatch between prediction and observation
            print(f"    FAIL: Observation on uniqueness DOES NOT match conjecture's prediction.")
            all_tests_passed_for_shape = False
            if conjecture_predicts_unique: # Conjecture: Unique, but Observed: Non-Unique
                print(f"        Details: Predicted UNIQUE, but Observed NON-UNIQUE (t - t_prime is non-zero).")
            else: # Conjecture: Non-Unique, but Observed: Unique
                print(f"        Details: Predicted NON-UNIQUE, but Observed UNIQUE (t - t_prime is zero).")
                if s_indices: # If there were s_indices, non-unique was expected to manifest as T'!=T there.
                     print(f"        This is unexpected if s_indices was non-empty ({len(s_indices)} entries).")


    print(f"--- Overall for Shape {shape}: {'PASS (core conjecture validated)' if all_tests_passed_for_shape else 'ISSUES NOTED (check details)'} ---")
    return all_tests_passed_for_shape

# --- Test Function 2 (show_filler_formula) ---
# This function's core logic remains largely the same as it's about inspecting
# the formula from t.filler() and checking the 2^N monomial count for (t-t_prime)[idx].
# It will use t.order() as well.

def show_filler_formula(shape: tuple, j_missing: int, target_index_to_analyze: tuple):
    """
    Shows the symbolic formula for a single entry in t_prime
    and in the difference (t - t_prime), and checks monomial count.
    """
    print(f"\n--- Showing Filler Formula for Shape {shape}, Horn j_missing={j_missing}, Target Index {target_index_to_analyze} ---")
    t = SymbolicTensor(shape=shape, init_type='range')
    N = t.dimen()
    k_order = t.order() # Using instance method

    if N < 1:
        print(f"    Skipping filler formula analysis as N={N} < 1.")
        return

    is_valid_target = True
    if len(target_index_to_analyze) != t.order(): is_valid_target = False
    if is_valid_target:
        for i, dim_size in enumerate(t.shape):
            if not (0 <= target_index_to_analyze[i] < dim_size):
                is_valid_target = False; break
    if not is_valid_target:
        print(f"    ERROR: Target index {target_index_to_analyze} is not valid for shape {shape}.")
        return

    horn_faces_list = t.horn(j_missing)
    t_prime = t.filler(horn_faces_list, j_missing)
    
    t_original_entry_expr = sp.sympify(t.tensor[target_index_to_analyze])
    t_prime_entry_expr = sp.sympify(t_prime.tensor[target_index_to_analyze])
    
    print(f"    t[{target_index_to_analyze}] (original) = {t_original_entry_expr}")
    print(f"    t_prime[{target_index_to_analyze}] (filler) = {t_prime_entry_expr}")

    diff_at_target_expr = sp.simplify(t_original_entry_expr - t_prime_entry_expr)
    print(f"    (t - t_prime)[{target_index_to_analyze}] = {diff_at_target_expr}")

    contributing_symbols_in_diff = diff_at_target_expr.atoms(sp.Symbol)
    print(f"    Symbols in (t - t_prime)[{target_index_to_analyze}]: {sorted([str(s) for s in contributing_symbols_in_diff])}")

    num_monomials_in_diff = 0
    expanded_diff = sp.expand(diff_at_target_expr) 
    if isinstance(expanded_diff, sp.Add):
        num_monomials_in_diff = len(expanded_diff.args)
    elif expanded_diff != 0: 
        num_monomials_in_diff = 1
    
    print(f"    Monomials in (t - t_prime)[{target_index_to_analyze}] (expanded): {num_monomials_in_diff}")

    # Determine if target_index_to_analyze is in s_indices for this horn
    all_t_indices = set(np.ndindex(t.shape))
    original_axes_t = t._dims()
    indices_covered_by_horn_faces = set()
    for p_face_val, face_tensor_in_horn in enumerate(horn_faces_list):
        if p_face_val == j_missing: continue
        temp_original_axes_for_face_construction = [np.delete(original_axes_t[ax_idx], p_face_val) for ax_idx in range(t.order())]
        for face_local_idx in np.ndindex(face_tensor_in_horn.shape):
            original_idx_tuple = tuple(
                temp_original_axes_for_face_construction[dim_idx][face_local_idx[dim_idx]]
                for dim_idx in range(t.order())
            )
            indices_covered_by_horn_faces.add(original_idx_tuple)
    s_indices = all_t_indices - indices_covered_by_horn_faces

    if target_index_to_analyze in s_indices:
        print(f"    Target index IS one of s_indices ('missing entry').")
        # Conjecture: non-zero entries in T-T' have 2^N monomials
        expected_monomials_for_diff = 2**N if N >=0 else 0 # 2^0=1
        if num_monomials_in_diff == expected_monomials_for_diff:
            print(f"    OK: Monomial count in diff matches conjecture's 2^N = 2^{N} = {expected_monomials_for_diff}.")
        else:
            print(f"    FAIL (Conjecture 2^N): Expected {expected_monomials_for_diff} monomials, Got {num_monomials_in_diff}.")
    elif sp.simplify(diff_at_target_expr) != 0:
        print(f"    WARNING: Target index NOT in s_indices, but (t-t_prime)[target] non-zero ({diff_at_target_expr}). This implies s_indices calculation might have missed something or filler is incorrect.")
    else: 
        print(f"    INFO: Target index NOT in s_indices, and (t-t_prime)[target] zero, as expected.")

    print(f"--- End of Formula Inspection for Index {target_index_to_analyze} ---")

if __name__ == "__main__":
    shape = (5, 5, 5, 5, 5)
    validate_filler_conjecture(shape)   
    # Show filler formula for a specific case
    j_missing = 2
    target_index_to_analyze = (0, 2, 1, 3, 4)
    show_filler_formula(shape, j_missing, target_index_to_analyze)



    shape = (5, 5, 5, 5, 5, 5)
    validate_filler_conjecture(shape)   
    # Show filler formula for a specific case
    j_missing = 2
    target_index_to_analyze = (0, 2, 1, 3, 4, 0)
    show_filler_formula(shape, j_missing, target_index_to_analyze)


    # Example usage
    shape = (3, 3, 3)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 1
    target_index_to_analyze = (0, 2, 1)
    show_filler_formula(shape, j_missing, target_index_to_analyze) 

    shape = (5, 5, 5)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 1
    target_index_to_analyze = (0, 2, 1)
    show_filler_formula(shape, j_missing, target_index_to_analyze)

    shape = (5, 5, 5, 5)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 1
    target_index_to_analyze = (0, 2, 1, 3)
    show_filler_formula(shape, j_missing, target_index_to_analyze)

    # Show filler formula for a specific case
    j_missing = 4
    target_index_to_analyze = (3, 2, 1, 0)
    show_filler_formula(shape, j_missing, target_index_to_analyze)

    shape = (3, 3, 4)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 1
    target_index_to_analyze = (0, 2, 1)
    show_filler_formula(shape, j_missing, target_index_to_analyze)

    shape = (3, 4, 4)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 1
    target_index_to_analyze = (0, 2, 1)
    show_filler_formula(shape, j_missing, target_index_to_analyze)

    shape = (4, 4, 5)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 3
    target_index_to_analyze = (0, 2, 1)
    show_filler_formula(shape, j_missing, target_index_to_analyze)

    shape = (4, 5, 5)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 3
    target_index_to_analyze = (0, 2, 1)
    show_filler_formula(shape, j_missing, target_index_to_analyze)


    shape = (4, 5, 5, 5)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 3
    target_index_to_analyze = (0, 2, 1, 3)
    show_filler_formula(shape, j_missing, target_index_to_analyze)

    shape = (4, 4, 5, 5)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 3
    target_index_to_analyze = (0, 2, 1, 3)
    show_filler_formula(shape, j_missing, target_index_to_analyze)

    shape = (4, 4, 4, 4)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 3
    target_index_to_analyze = (0, 2, 1, 3)
    show_filler_formula(shape, j_missing, target_index_to_analyze)

    shape = (4, 4, 4, 5)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 3
    target_index_to_analyze = (0, 2, 1, 3)
    show_filler_formula(shape, j_missing, target_index_to_analyze)

    shape = (4, 4, 4, 6)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 3
    target_index_to_analyze = (0, 2, 1, 3)
    show_filler_formula(shape, j_missing, target_index_to_analyze)

    shape = (3, 3, 3, 3)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 2
    target_index_to_analyze = (0, 2, 1, 0)
    show_filler_formula(shape, j_missing, target_index_to_analyze)

    shape = (3, 3, 3, 4)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 2
    target_index_to_analyze = (0, 2, 1, 0)
    show_filler_formula(shape, j_missing, target_index_to_analyze)

    shape = (3, 3, 4, 4)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 2
    target_index_to_analyze = (0, 2, 1, 0)
    show_filler_formula(shape, j_missing, target_index_to_analyze)

    shape = (3, 4, 4, 4)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 2
    target_index_to_analyze = (0, 2, 1, 0)
    show_filler_formula(shape, j_missing, target_index_to_analyze)

    shape = (4, 4, 4, 4, 4)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 2
    target_index_to_analyze = (0, 2, 1, 3, 0)
    show_filler_formula(shape, j_missing, target_index_to_analyze)

    shape = (4, 4, 4, 4, 5)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 2
    target_index_to_analyze = (0, 2, 1, 3, 0)
    show_filler_formula(shape, j_missing, target_index_to_analyze)  

    shape = (4, 4, 4, 5, 5)
    validate_filler_conjecture(shape)

    # Show filler formula for a specific case
    j_missing = 2
    target_index_to_analyze = (0, 2, 1, 3, 0)
    show_filler_formula(shape, j_missing, target_index_to_analyze)  
    
    shape = (4, 4, 5, 5, 5)
    validate_filler_conjecture(shape)   
    # Show filler formula for a specific case
    j_missing = 2   
    target_index_to_analyze = (0, 2, 1, 3, 0)
    show_filler_formula(shape, j_missing, target_index_to_analyze)