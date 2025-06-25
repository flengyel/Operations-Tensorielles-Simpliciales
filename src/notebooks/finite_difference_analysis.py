import itertools
import numpy as np
from typing import List, Tuple, Set

#memoization cache
_cache = {}

def calculate_missing_indices(shape: Tuple[int, ...], horn_j: int = 0) -> int:
    """
    Calculates the number of missing indices for a given tensor shape and horn.
    This version includes memoization to speed up repeated calculations for the same shape.
    """
    # Use a canonical representation for caching (sorted tuple)
    cache_key = (tuple(sorted(shape)), horn_j)
    if cache_key in _cache:
        return _cache[cache_key]

    order_k = len(shape)
    dim_N = min(shape) - 1

    if dim_N < 0:
        return 0

    if horn_j < 0 or horn_j > dim_N:
        raise ValueError(f"Horn index {horn_j} is out of bounds for dimension {dim_N}")

    all_indices = itertools.product(*[range(s) for s in shape])

    face_indices_in_horn = [i for i in range(dim_N + 1) if i != horn_j]

    if not face_indices_in_horn:
        # This case is tricky with a generator. We must consume it to count.
        # For very large shapes, this could be a memory issue if not handled carefully.
        # However, for N=0, this is unlikely to be a problem.
        count = sum(1 for _ in all_indices)
        _cache[cache_key] = count
        return count

    # The first set E_k is computed from the full generator
    first_k = face_indices_in_horn[0]
    missing_indices = {idx for idx in all_indices if first_k in idx}
    
    # Intersect with the remaining sets E_k
    for k in face_indices_in_horn[1:]:
        # Filter the already reduced set, which is much more efficient
        missing_indices = {idx for idx in missing_indices if k in idx}
    
    result = len(missing_indices)
    _cache[cache_key] = result
    return result

def generate_shape_path(target_shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """
    Generates a canonical path of shapes from the fundamental shape to the target shape.
    The path is built by incrementing one component at a time, from right to left.
    
    Args:
        target_shape: The final shape tuple to reach.
        
    Returns:
        A list of shape tuples representing the path.
    """
    if not target_shape:
        return []

    # 1. Determine the fundamental shape
    min_val = min(target_shape)
    k = len(target_shape)
    fundamental_shape = tuple([min_val] * k)

    path = [fundamental_shape]
    current_shape = list(fundamental_shape)
    
    # Ensure target_shape is sorted to make the path canonical and easier to analyze
    sorted_target = tuple(sorted(target_shape))

    # 4. Increment from right to left to build the path
    for i in range(k - 1, -1, -1):
        # Increment the current component up to its target value
        while current_shape[i] < sorted_target[i]:
            current_shape[i] += 1
            # Add a sorted version to the path to handle symmetries
            path.append(tuple(sorted(current_shape)))
            
    # The path may contain duplicates if the target was not sorted, remove them.
    # We use a dictionary to preserve order while removing duplicates.
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
    Main analysis function. It computes the path from the fundamental shape to the
    target shape, calculates the number of missing indices for each step, and
    tabulates the finite differences of the resulting sequence.
    """
    print(f"--- Analyzing Path to Shape: {target_shape} ---")

    # 1. & 2. Determine fundamental shape and generate the path
    path = generate_shape_path(target_shape)
    if not path:
        print("Invalid target shape.")
        return

    print(f"Fundamental Shape: {path[0]}")
    print(f"Path contains {len(path)} steps.")

    # 3. Compute the number of missing indices for each shape in the path
    print("\nComputing counts along the path...")
    counts = [calculate_missing_indices(s) for s in path]
    print("Computation complete.")

    # 4. Compute the finite differences
    differences = compute_finite_differences(counts)
    
    # 5. Tabulate the results
    print("\n--- Finite Difference Calculus Table ---")
    
    # Determine max width for formatting
    max_shape_width = max(len(str(s)) for s in path)
    max_count_width = max(len(str(c)) for c in counts)

    # Header
    header = f"{'Shape':<{max_shape_width}} | {'Count':>{max_count_width}}"
    diff_headers = []
    for i, diff_seq in enumerate(differences):
        if diff_seq:
            max_diff_width = max(len(str(d)) for d in diff_seq) if diff_seq else 0
            diff_header = f"D{i+1}".rjust(max_diff_width)
            diff_headers.append(diff_header)
    
    print(header + " | " + " | ".join(diff_headers))
    print("-" * (len(header) + 3 + len(" | ".join(diff_headers))))

    # Data rows
    for i, (shape, count) in enumerate(zip(path, counts)):
        row = f"{str(shape):<{max_shape_width}} | {count:>{max_count_width}}"
        diff_entries = []
        for j, diff_seq in enumerate(differences):
            max_diff_width = max(len(str(d)) for d in diff_seq) if diff_seq else 0
            if i < len(diff_seq):
                diff_entry = str(diff_seq[i]).rjust(max_diff_width)
                diff_entries.append(diff_entry)
            else:
                diff_entries.append(" " * max_diff_width)
        print(row + " | " + " | ".join(diff_entries))


if __name__ == '__main__':
    # --- Example Analysis ---
    # This will reproduce the analysis for the k=5, N=2 path we discussed.
    target_shape_k5_n2 = (3, 4, 4, 4, 4)
    analyze_shape_calculus(target_shape_k5_n2)

    print("\n" + "="*50 + "\n")

    # This will analyze the more complex path from the previous turn.
    target_shape_complex = (3, 3, 4, 5)
    analyze_shape_calculus(target_shape_complex)
    
    print("\n" + "="*50 + "\n")

    # A new, more complex example for k=6
    target_shape_k6 = (3, 3, 3, 4, 4, 4)
    analyze_shape_calculus(target_shape_k6)
