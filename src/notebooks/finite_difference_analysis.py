import itertools
import numpy as np
from typing import List, Tuple, Set

# --- Import the efficient Dask function ---
# This assumes that horn_map_reduce.py is in the same directory.
from horn_map_reduce import compute_missing_indices_dask

# Memoization cache for the wrapper
_cache = {}

def calculate_missing_indices_wrapper(shape: Tuple[int, ...], horn_j: int = 0) -> int:
    """
    Wrapper function that calls the imported Dask implementation and handles caching.
    This function returns only the count of missing indices.
    """
    # Use a canonical representation for caching (sorted tuple)
    cache_key = (tuple(sorted(shape)), horn_j)
    if cache_key in _cache:
        return _cache[cache_key]

    # Call the efficient, imported Dask function to get the set of indices
    missing_indices_set = compute_missing_indices_dask(shape, horn_j)
    
    result = len(missing_indices_set)
    _cache[cache_key] = result
    return result

def generate_shape_path(target_shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """
    Generates a canonical path of shapes from the fundamental shape to the target shape.
    The path is built by incrementing one component at a time, from right to left.
    """
    if not target_shape:
        return []

    min_val = min(target_shape)
    k = len(target_shape)
    fundamental_shape = tuple([min_val] * k)

    path = [fundamental_shape]
    current_shape = list(fundamental_shape)
    
    sorted_target = tuple(sorted(target_shape))

    for i in range(k - 1, -1, -1):
        while current_shape[i] < sorted_target[i]:
            current_shape[i] += 1
            path.append(tuple(sorted(current_shape)))
            
    return list(dict.fromkeys(path))

def compute_finite_differences(sequence: List[int]) -> List[List[int]]:
    """
    Computes all orders of finite differences for a given sequence of numbers.
    """
    if not sequence:
        return []
        
    all_diffs = []
    current_seq = sequence
    
    while len(current_seq) > 1:
        next_seq = np.diff(current_seq).tolist()
        all_diffs.append(next_seq)
        current_seq = next_seq
        
    return all_diffs

def analyze_shape_calculus(target_shape: Tuple[int, ...]):
    """
    Main analysis function. It uses the efficient Dask computation to analyze
    the finite difference calculus for a given target shape.
    """
    print(f"--- Analyzing Path to Shape: {target_shape} ---")

    path = generate_shape_path(target_shape)
    if not path:
        print("Invalid target shape.")
        return

    print(f"Fundamental Shape: {path[0]}")
    print(f"Path contains {len(path)} steps.")

    print("\nComputing counts along the path using Dask...")
    # Use the efficient wrapper function which now calls the imported Dask function
    counts = [calculate_missing_indices_wrapper(s) for s in path]
    print("Computation complete.")

    differences = compute_finite_differences(counts)
    
    print("\n--- Finite Difference Calculus Table ---")
    
    # Formatting the output table
    if not path: return
    max_shape_width = max(len(str(s)) for s in path)
    max_count_width = max(len(str(c)) for c in counts) if counts else 0

    header = f"{'Shape':<{max_shape_width}} | {'Count':>{max_count_width}}"
    diff_headers = []
    for i, diff_seq in enumerate(differences):
        if diff_seq:
            max_diff_width = max(max(len(str(d)) for d in diff_seq), len(f"D{i+1}"))
            diff_header = f"D{i+1}".rjust(max_diff_width)
            diff_headers.append(diff_header)
    
    print(header + " | " + " | ".join(diff_headers))
    print("-" * (len(header) + 3 + len(" | ".join(diff_headers))))

    for i, (shape, count) in enumerate(zip(path, counts)):
        row = f"{str(shape):<{max_shape_width}} | {count:>{max_count_width}}"
        diff_entries = []
        for j, diff_seq in enumerate(differences):
            max_diff_width = max(max(len(str(d)) for d in diff_seq), len(f"D{j+1}")) if diff_seq else 0
            if i < len(diff_seq):
                diff_entry = str(diff_seq[i]).rjust(max_diff_width)
                diff_entries.append(diff_entry)
            else:
                diff_entries.append(" " * max_diff_width)
        print(row + " | " + " | ".join(diff_entries))


if __name__ == '__main__':
    # --- Example Analysis ---
    print("The analyzer now uses the imported Dask implementation for all calculations.")
    
    # Analysis for the k=5, N=2 path
    target_shape_k5_n2 = (3, 4, 4, 4, 4)
    analyze_shape_calculus(target_shape_k5_n2)

    print("\n" + "="*50 + "\n")

    # Analysis for a more complex k=6 path
    target_shape_k6 = (3, 3, 3, 4, 4, 4)
    analyze_shape_calculus(target_shape_k6)
