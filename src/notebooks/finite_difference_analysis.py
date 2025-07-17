import itertools
import numpy as np
import math
import logging
from datetime import datetime
from typing import List, Tuple, Set
from functools import reduce
import dask.bag as db

# --- Import the primary algorithm from the dedicated module ---
from horn_map_reduce import compute_missing_indices_dask

# --- Caches for memoization ---
_wrapper_cache = {}
_stirling_cache = {}

def setup_logger():
    """Sets up a logger to write to both console and a uniquely named file."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile_name = f"analysis_log_{timestamp}.txt"
    
    logger = logging.getLogger('calculus_analyzer')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        fh = logging.FileHandler(logfile_name)
        fh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
        
    return logger, logfile_name

def stirling2(n: int, k: int) -> int:
    """Computes the Stirling number of the second kind, S2(n, k), with memoization."""
    if (n, k) in _stirling_cache: return _stirling_cache[(n, k)]
    if k > n or k < 0: return 0
    if k == 0 and n == 0: return 1
    if k == 0 and n > 0: return 0
    if k == n: return 1
    if k == 1 and n > 0: return 1
    result = k * stirling2(n - 1, k) + stirling2(n - 1, k - 1)
    _stirling_cache[(n, k)] = result
    return result

def calculate_missing_indices_wrapper(shape: Tuple[int, ...], horn_j: int = 0) -> int:
    """Wrapper function that calls the Dask implementation and handles caching."""
    cache_key = (tuple(sorted(shape)), horn_j)
    if cache_key in _wrapper_cache:
        return _wrapper_cache[cache_key]
    missing_indices_set = compute_missing_indices_dask(shape, horn_j)
    result = len(missing_indices_set)
    _wrapper_cache[cache_key] = result
    return result

def generate_greedy_path(target_shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """Generates a canonical, minimal-step path from the fundamental shape to the target."""
    if not target_shape: return []
    sorted_target = tuple(sorted(target_shape))
    min_val = sorted_target[0]
    k = len(sorted_target)
    current_shape = tuple([min_val] * k)
    path = [current_shape]
    while current_shape != sorted_target:
        candidates = set()
        for i in range(k):
            if current_shape[i] < sorted_target[i]:
                next_shape_list = list(current_shape)
                next_shape_list[i] += 1
                candidates.add(tuple(next_shape_list))
        if not candidates: break
        candidate_counts = {cand: calculate_missing_indices_wrapper(cand) for cand in candidates}
        best_next_shape = min(candidate_counts, key=lambda x: candidate_counts[x])
        current_shape = best_next_shape
        path.append(current_shape)
    return path

def compute_finite_differences(sequence: List[int]) -> List[List[int]]:
    """Computes all orders of finite differences for a given sequence."""
    if not sequence: return []
    all_diffs = []
    current_seq = sequence
    while len(current_seq) > 1:
        next_seq = np.diff(current_seq).astype(int).tolist()
        all_diffs.append(next_seq)
        current_seq = next_seq
    return all_diffs

def analyze_shape_calculus(target_shape: Tuple[int, ...], logger: logging.Logger):
    """Main analysis function using the corrected and integrated algorithms."""
    logger.info(f"--- Analyzing Path to Shape: {target_shape} ---")
    path = generate_greedy_path(target_shape)
    if not path:
        logger.info("Invalid target shape.")
        return
    fundamental_shape = path[0]
    k = len(fundamental_shape)
    N = min(fundamental_shape) - 1
    predicted_count = math.factorial(N) * stirling2(k + 1, N + 1) if k >= N else 0
    
    logger.info(f"Fundamental Shape: {fundamental_shape} (k={k}, N={N})")
    logger.info(f"Predicted Count for Fundamental Shape: {predicted_count}")
    logger.info(f"Path contains {len(path)} steps.")
    logger.info("\nComputing counts along the path using Dask...")
    counts = [calculate_missing_indices_wrapper(s) for s in path]
    logger.info("Computation complete.")
    differences = compute_finite_differences(counts)
    
    log_output = "\n--- Finite Difference Calculus Table ---\n"
    if not path:
        logger.info(log_output)
        return
        
    max_shape_width = max(len(str(s)) for s in path) if path else 0
    max_count_width = max(len(str(c)) for c in counts) if counts else 0
    check_col_width = len(f"(Predicted: {predicted_count}, Matches!)") + 2
    
    diff_widths = [max(max((len(str(d)) for d in diff_seq), default=0), 3) for diff_seq in differences]
    header_parts = [f"{'Shape':<{max_shape_width}}", f"{'Count':>{max_count_width}}", f"{'Prediction Check':<{check_col_width}}"]
    diff_headers = [f"D{i+1}".rjust(w) for i, w in enumerate(diff_widths)]
    header = " | ".join(header_parts + diff_headers)
    log_output += header + "\n" + "-" * len(header) + "\n"

    for i, (shape, count) in enumerate(zip(path, counts)):
        row_parts = [f"{str(shape):<{max_shape_width}}", f"{count:>{max_count_width}}"]
        if i == 0:
            check_str = "Matches!" if count == predicted_count else "MISMATCH!"
            check_msg = f"(Predicted: {predicted_count}, {check_str})"
            row_parts.append(f"{check_msg:<{check_col_width}}")
        else:
            row_parts.append(f"{'':<{check_col_width}}")
        for j, diff_seq in enumerate(differences):
             if i < len(diff_seq):
                row_parts.append(str(diff_seq[i]).rjust(diff_widths[j]))
             else:
                row_parts.append(" " * diff_widths[j])
        log_output += " | ".join(row_parts) + "\n"
    logger.info(log_output)

if __name__ == '__main__':
    logger, logfile_name = setup_logger()
    logger.info(f"Log file for this session: {logfile_name}\n")
    logger.info("Running analysis with the FIXED algorithm (k < n case is now handled).")

    # This shape has k = 3 < n = 16
    target_shape_k3_n16 = (17, 19, 23)
    analyze_shape_calculus(target_shape_k3_n16, logger)

    # This shape has k = 4 >= n = 3
    target_shape_k4_n3 = (4, 4, 5, 7, 9)
    analyze_shape_calculus(target_shape_k4_n3, logger)