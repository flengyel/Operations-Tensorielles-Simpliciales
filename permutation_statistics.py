#!/usr/bin/env python3
"""
Permutation Matrix Analysis Tool

This script analyzes permutation matrices, their faces, and the algebraic properties
of these faces in relation to the discrepancy operator ∂(A²) - ∂(A)².

The key theoretical question explored:
- What is the relationship between the faces of a permutation matrix P and
  the discrepancy ∂(P²) - ∂(P)²?

Key findings:
- Faces where the corresponding diagonal element is 1 tend to be permutation matrices
- Faces where the corresponding diagonal element is 0 show various algebraic behaviors:
  - Some are nilpotent
  - Some are idempotent (either immediately or eventually)
  - Some exhibit cyclic behavior
"""

import numpy as np
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import argparse
import os
from tensor_ops import bdry, face, dimen, random_real_matrix

def compute_discrepancy(a: np.ndarray) -> np.ndarray:
    """
    Compute the discrepancy ∂(A²) - ∂(A)² for a given matrix A.
    
    Args:
        a: Input matrix
        
    Returns:
        The discrepancy matrix
    """
    a_squared = np.matmul(a, a)
    bdry_a_squared = bdry(a_squared)
    bdry_a = bdry(a)
    squared_bdry_a = np.matmul(bdry_a, bdry_a)
    return bdry_a_squared - squared_bdry_a

def compute_face_similarity(a: np.ndarray, discrepancy: np.ndarray) -> List[float]:
    """
    Compute the cosine similarity between the discrepancy and each face of A.
    
    Args:
        a: Input matrix
        discrepancy: The discrepancy matrix
        
    Returns:
        List of cosine similarities for each face
    """
    similarities = []
    dim = dimen(a)
    
    for i in range(dim + 1):
        face_i = face(a, i)
        
        # Flatten both matrices for cosine similarity
        comm_flat = discrepancy.flatten()
        face_flat = face_i.flatten()
        
        # Normalize
        comm_norm = np.linalg.norm(comm_flat)
        face_norm = np.linalg.norm(face_flat)
        
        # Compute cosine similarity if norms are non-zero
        if comm_norm > 0 and face_norm > 0:
            similarity = np.dot(comm_flat, face_flat) / (comm_norm * face_norm)
        else:
            similarity = 0
            
        similarities.append(similarity)
    
    return similarities

def compute_frobenius_ratios(a: np.ndarray, discrepancy: np.ndarray) -> List[float]:
    """
    Compute the ratio of Frobenius norms: ||discrepancy|| / ||face_i||
    
    Args:
        a: Input matrix
        discrepancy: The discrepancy matrix
        
    Returns:
        List of Frobenius norm ratios for each face
    """
    ratios = []
    dim = dimen(a)
    
    comm_norm = np.linalg.norm(discrepancy)
    
    for i in range(dim + 1):
        face_i = face(a, i)
        face_norm = np.linalg.norm(face_i)
        
        if face_norm > 0:
            ratio = comm_norm / face_norm
        else:
            ratio = np.inf
            
        ratios.append(ratio)
    
    return ratios

def is_permutation_matrix(matrix: np.ndarray) -> bool:
    """
    Check if a matrix is a permutation matrix.
    A permutation matrix has exactly one 1 in each row and column, with all other elements being 0.
    
    Args:
        matrix: Input matrix
        
    Returns:
        True if the matrix is a permutation matrix, False otherwise
    """
    # Check if all elements are 0 or 1
    if not np.all(np.logical_or(matrix == 0, matrix == 1)):
        return False
    
    # Check if each row and column has exactly one 1
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    
    return np.all(row_sums == 1) and np.all(col_sums == 1)

def is_nilpotent(matrix: np.ndarray, max_power: int = None, tolerance: float = 1e-10) -> Tuple[bool, int]:
    """
    Check if a matrix is nilpotent (some power of the matrix is zero).
    
    Args:
        matrix: Input matrix
        max_power: Maximum power to check
        tolerance: Numerical tolerance for zero
        
    Returns:
        Tuple of (is_nilpotent, nilpotent_index)
    """
    if max_power is None:
        max_power = matrix.shape[0]
    
    current = np.copy(matrix)
    for i in range(1, max_power + 1):
        if np.all(np.abs(current) < tolerance):
            return True, i  # Return True and the nilpotent index
        current = np.matmul(current, matrix)
    
    return False, 0

def is_idempotent(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Check if a matrix is idempotent (M² = M).
    
    Args:
        matrix: Input matrix
        tolerance: Numerical tolerance for equality
        
    Returns:
        True if the matrix is idempotent, False otherwise
    """
    squared = np.matmul(matrix, matrix)
    return np.allclose(squared, matrix, atol=tolerance)

def check_cycles(matrix: np.ndarray, max_powers: int = 40, tolerance: float = 1e-7) -> Tuple[bool, int, int]:
    """
    Check if a matrix exhibits cyclic behavior under repeated multiplication.
    
    Args:
        matrix: Input matrix
        max_powers: Maximum number of powers to check
        tolerance: Numerical tolerance for equality
        
    Returns:
        Tuple of (has_cycle, cycle_start, cycle_length)
        If eventually idempotent, returns (True, power_at_idempotence, 0)
    """
    powers = [np.identity(matrix.shape[0]), matrix]  # Start with I and A
    
    for power in range(2, max_powers + 1):
        next_power = np.matmul(powers[-1], matrix)  # A^(n+1) = A^n • A
        
        # Check if this power equals any previous power
        for prev_idx, prev_power in enumerate(powers):
            if np.allclose(next_power, prev_power, atol=tolerance):
                cycle_start = prev_idx
                cycle_length = power - prev_idx
                
                # If A^n = A^n-1, this indicates idempotence, not a cycle
                if cycle_length == 1 and prev_idx == power - 1:
                    return True, prev_idx, 0  # Eventually idempotent at power prev_idx
                else:
                    return True, cycle_start, cycle_length  # True cycle
        
        powers.append(next_power)
    
    return False, 0, 0  # No cycle or idempotence detected within max_powers

def create_permutation_matrix(n: int, seed: int = 42) -> np.ndarray:
    """
    Create a random permutation matrix of size n.
    
    Args:
        n: Size of the matrix
        seed: Random seed
        
    Returns:
        A random permutation matrix
    """
    rng = np.random.default_rng(seed=seed)
    perm = rng.permutation(n)
    P = np.zeros((n, n))
    for i, j in enumerate(perm):
        P[i, j] = 1
    return P

def check_face_cycles(face_i: np.ndarray, is_nilp: bool, is_immediately_idempotent: bool, 
                      verbose: bool = False, max_powers: int = 40, tolerance: float = 1e-7) -> Tuple[bool, int, int, bool]:
    """
    Check if a face exhibits cyclic behavior or idempotence.
    
    Args:
        face_i: Face matrix
        is_nilp: Whether the face is nilpotent
        is_immediately_idempotent: Whether the face is immediately idempotent
        verbose: Whether to print verbose information
        max_powers: Maximum number of powers to check
        tolerance: Numerical tolerance for equality
        
    Returns:
        Tuple of (has_cycle, cycle_start, cycle_length, is_eventually_idempotent)
    """
    has_cycle = False
    cycle_start = 0
    cycle_length = 0
    is_eventually_idempotent = False
    
    if not is_nilp and not is_immediately_idempotent:
        # Call check_cycles with explicit parameters instead of using the global function
        has_cycle, cycle_start, cycle_length = check_cycles(face_i, max_powers, tolerance)
        
        if has_cycle and verbose:
            if cycle_length == 0:
                is_eventually_idempotent = True
                print(f"Eventually idempotent at power {cycle_start}")
                print("Matrix at idempotence:")
                print(np.linalg.matrix_power(face_i, cycle_start))
            else:
                print(f"Cycle detected! Power {cycle_start} repeats at power {cycle_start + cycle_length}")
                print(f"Cycle length: {cycle_length}")
                print(f"Matrix power at cycle start (A^{cycle_start}):")
                print(np.linalg.matrix_power(face_i, cycle_start))
    
    if cycle_length == 0 and has_cycle:
        is_eventually_idempotent = True
    
    return has_cycle, cycle_start, cycle_length, is_eventually_idempotent

def analyze_face(p: np.ndarray, i: int, similarities: List[float], 
                 diagonal_entries: np.ndarray, verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze a face of a permutation matrix to identify its algebraic properties.
    
    Args:
        p: Permutation matrix
        i: Face index
        similarities: List of similarities with discrepancy
        diagonal_entries: Diagonal entries of the permutation matrix
        verbose: Whether to print verbose information
        
    Returns:
        Dictionary with analysis results
    """
    face_i = face(p, i)
    is_perm = is_permutation_matrix(face_i)
    is_nilp, nilp_index = is_nilpotent(face_i)
    is_immediately_idempotent = is_idempotent(face_i)
    
    has_cycle, cycle_start, cycle_length, is_eventually_idempotent = check_face_cycles(
        face_i, is_nilp, is_immediately_idempotent, verbose)
    
    if verbose:
        print(f"Diagonal entry P[{i},{i}] = {diagonal_entries[i]}")
        print(f"Face {i} is {'a permutation matrix' if is_perm else 'not a permutation matrix'}")
        print(f"Face {i} is {'nilpotent (power ' + str(nilp_index) + ')' if is_nilp else 'not nilpotent'}")
        print(f"Face {i} is {'immediately idempotent' if is_immediately_idempotent else 'not immediately idempotent'}")
        print(f"Face {i} is {'eventually idempotent' if is_eventually_idempotent else 'not eventually idempotent'}")
        print(f"Face {i} is {'cyclic' if (has_cycle and cycle_length > 0) else 'not cyclic'}")
        print(f"Similarity with discrepancy: {similarities[i]:.6f}")
    
    return {
        'face_index': i,
        'diagonal_entry': diagonal_entries[i],
        'is_perm': is_perm,
        'is_nilp': is_nilp,
        'nilp_index': nilp_index if is_nilp else 0,
        'is_immediately_idempotent': is_immediately_idempotent,
        'is_eventually_idempotent': is_eventually_idempotent,
        'has_cycle': has_cycle and cycle_length > 0,
        'cycle_start': cycle_start,
        'cycle_length': cycle_length,
        'similarity': similarities[i]
    }

def summarize_matrix_results(results: List[Dict[str, Any]], verbose: bool = False) -> None:
    """
    Summarize the results for a single permutation matrix.
    
    Args:
        results: List of analysis results
        verbose: Whether to print verbose information
    """
    total = len(results)
    perm_count = sum(1 for r in results if r['is_perm'])
    nilp_count = sum(1 for r in results if r['is_nilp'])
    immediate_idemp_count = sum(1 for r in results if r['is_immediately_idempotent'])
    eventual_idemp_count = sum(1 for r in results if r['is_eventually_idempotent'])
    cycle_count = sum(1 for r in results if r['has_cycle'])
    
    print("\nSummary for this permutation matrix:")
    print(f"Total faces: {total}")
    print(f"Permutation matrices: {perm_count}")
    print(f"Nilpotent matrices: {nilp_count}")
    print(f"Immediately idempotent matrices: {immediate_idemp_count}")
    print(f"Eventually idempotent matrices: {eventual_idemp_count}")
    print(f"Cyclic matrices: {cycle_count}")
    
    if results and verbose:
        max_sim_idx = np.argmax([r['similarity'] for r in results])
        max_sim = results[max_sim_idx]['similarity']
        face_idx = results[max_sim_idx]['face_index']
        
        print(f"\nFace most similar to discrepancy: Face {face_idx} (similarity: {max_sim:.6f})")
        print("Properties:", end=" ")
        if results[max_sim_idx]['is_perm']:
            print("permutation", end=" ")
        if results[max_sim_idx]['is_nilp']:
            print("nilpotent", end=" ")
        if results[max_sim_idx]['is_immediately_idempotent']:
            print("immediately idempotent", end=" ")
        if results[max_sim_idx]['is_eventually_idempotent']:
            print("eventually idempotent", end=" ")
        if results[max_sim_idx]['has_cycle']:
            print("cyclic", end=" ")
        print()

def print_overall_statistics(all_results: List[Dict[str, Any]]) -> None:
    """
    Print overall statistics from all analyzed permutation matrices.
    
    Args:
        all_results: List of all analysis results
    """
    if not all_results:
        print("No results to analyze.")
        return
    
    total = len(all_results)
    
    diag_0_count = sum(1 for r in all_results if r['diagonal_entry'] == 0)
    diag_1_count = sum(1 for r in all_results if r['diagonal_entry'] == 1)
    
    perm_count = sum(1 for r in all_results if r['is_perm'])
    nilp_count = sum(1 for r in all_results if r['is_nilp'])
    immediate_idemp_count = sum(1 for r in all_results if r['is_immediately_idempotent'])
    eventual_idemp_count = sum(1 for r in all_results if r['is_eventually_idempotent'])
    cycle_count = sum(1 for r in all_results if r['has_cycle'])
    
    print("\n" + "="*70)
    print("OVERALL STATISTICS FOR PERMUTATION MATRIX FACES")
    print("="*70)
    print(f"Total faces analyzed: {total}")
    print(f"Faces with diagonal entry 0: {diag_0_count} ({diag_0_count/total*100:.1f}%)")
    print(f"Faces with diagonal entry 1: {diag_1_count} ({diag_1_count/total*100:.1f}%)")
    print("\nProperty distribution:")
    print(f"Permutation matrices: {perm_count} ({perm_count/total*100:.1f}%)")
    print(f"Nilpotent matrices: {nilp_count} ({nilp_count/total*100:.1f}%)")
    print(f"Immediately idempotent matrices: {immediate_idemp_count} ({immediate_idemp_count/total*100:.1f}%)")
    print(f"Eventually idempotent matrices: {eventual_idemp_count} ({eventual_idemp_count/total*100:.1f}%)")
    print(f"Cyclic matrices: {cycle_count} ({cycle_count/total*100:.1f}%)")
    
    print("\nConditional probabilities:")
    if diag_0_count:
        print(f"P(Permutation | Diagonal=0): {sum(1 for r in all_results if r['diagonal_entry'] == 0 and r['is_perm'])/diag_0_count*100:.1f}%")
        print(f"P(Nilpotent | Diagonal=0): {sum(1 for r in all_results if r['diagonal_entry'] == 0 and r['is_nilp'])/diag_0_count*100:.1f}%")
        print(f"P(Immediately idempotent | Diagonal=0): {sum(1 for r in all_results if r['diagonal_entry'] == 0 and r['is_immediately_idempotent'])/diag_0_count*100:.1f}%")
        print(f"P(Eventually idempotent | Diagonal=0): {sum(1 for r in all_results if r['diagonal_entry'] == 0 and r['is_eventually_idempotent'])/diag_0_count*100:.1f}%")
        print(f"P(Cyclic | Diagonal=0): {sum(1 for r in all_results if r['diagonal_entry'] == 0 and r['has_cycle'])/diag_0_count*100:.1f}%")
    
    if diag_1_count:
        print(f"P(Permutation | Diagonal=1): {sum(1 for r in all_results if r['diagonal_entry'] == 1 and r['is_perm'])/diag_1_count*100:.1f}%")

def generate_visualizations(all_results: List[Dict[str, Any]], output_dir: str = "plots") -> None:
    """
    Generate visualizations from the analysis results.
    
    Args:
        all_results: List of all analysis results
        output_dir: Directory to save plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert to numpy arrays for easier analysis
    diag_entries = np.array([r['diagonal_entry'] for r in all_results])
    is_perm = np.array([r['is_perm'] for r in all_results], dtype=int)
    is_nilp = np.array([r['is_nilp'] for r in all_results], dtype=int)
    is_imm_idemp = np.array([r['is_immediately_idempotent'] for r in all_results], dtype=int)
    is_ev_idemp = np.array([r['is_eventually_idempotent'] for r in all_results], dtype=int)
    has_cycle = np.array([r['has_cycle'] for r in all_results], dtype=int)
    similarities = np.array([r['similarity'] for r in all_results])
    matrix_sizes = np.array([r['matrix_size'] for r in all_results])
    
    # Plot 1: Diagonal entry vs properties
    plt.figure(figsize=(10, 6))
    labels = ['Permutation', 'Nilpotent', 'Immediately\nIdempotent', 'Eventually\nIdempotent', 'Cyclic']
    diag0_props = [
        sum(is_perm[diag_entries == 0]) / max(sum(diag_entries == 0), 1) * 100,
        sum(is_nilp[diag_entries == 0]) / max(sum(diag_entries == 0), 1) * 100,
        sum(is_imm_idemp[diag_entries == 0]) / max(sum(diag_entries == 0), 1) * 100,
        sum(is_ev_idemp[diag_entries == 0]) / max(sum(diag_entries == 0), 1) * 100,
        sum(has_cycle[diag_entries == 0]) / max(sum(diag_entries == 0), 1) * 100
    ]
    diag1_props = [
        sum(is_perm[diag_entries == 1]) / max(sum(diag_entries == 1), 1) * 100,
        sum(is_nilp[diag_entries == 1]) / max(sum(diag_entries == 1), 1) * 100,
        sum(is_imm_idemp[diag_entries == 1]) / max(sum(diag_entries == 1), 1) * 100,
        sum(is_ev_idemp[diag_entries == 1]) / max(sum(diag_entries == 1), 1) * 100,
        sum(has_cycle[diag_entries == 1]) / max(sum(diag_entries == 1), 1) * 100
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, diag0_props, width, label='Diagonal=0')
    rects2 = ax.bar(x + width/2, diag1_props, width, label='Diagonal=1')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Properties of Faces Based on Diagonal Entry')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagonal_vs_properties.png'))
    
    # Plot 2: Matrix size vs percentage of property types
    unique_sizes = np.unique(matrix_sizes)
    perm_by_size = []
    nilp_by_size = []
    imm_idemp_by_size = []
    ev_idemp_by_size = []
    cycle_by_size = []
    
    for size in unique_sizes:
        size_mask = matrix_sizes == size
        perm_by_size.append(sum(is_perm[size_mask]) / sum(size_mask) * 100)
        nilp_by_size.append(sum(is_nilp[size_mask]) / sum(size_mask) * 100)
        imm_idemp_by_size.append(sum(is_imm_idemp[size_mask]) / sum(size_mask) * 100)
        ev_idemp_by_size.append(sum(is_ev_idemp[size_mask]) / sum(size_mask) * 100)
        cycle_by_size.append(sum(has_cycle[size_mask]) / sum(size_mask) * 100)
    
    plt.figure(figsize=(12, 7))
    plt.plot(unique_sizes, perm_by_size, 'o-', label='Permutation')
    plt.plot(unique_sizes, nilp_by_size, 's-', label='Nilpotent')
    plt.plot(unique_sizes, imm_idemp_by_size, '^-', label='Immediately Idempotent')
    plt.plot(unique_sizes, ev_idemp_by_size, 'd-', label='Eventually Idempotent')
    plt.plot(unique_sizes, cycle_by_size, 'x-', label='Cyclic')
    plt.xlabel('Matrix Size')
    plt.ylabel('Percentage (%)')
    plt.title('Properties by Matrix Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'properties_by_size.png'))
    
    # Plot 3: Similarity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=np.mean(similarities), color='r', linestyle='--', label=f'Mean: {np.mean(similarities):.4f}')
    plt.xlabel('Cosine Similarity with Discrepancy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarities between Faces and Discrepancy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'))
    
    # Plot 4: Face properties vs similarity (boxplot)
    plt.figure(figsize=(12, 8))
    
    # Group similarities
    perm_sims = similarities[is_perm == 1]
    nilp_sims = similarities[is_nilp == 1]
    imm_idemp_sims = similarities[is_imm_idemp == 1]
    ev_idemp_sims = similarities[is_ev_idemp == 1]
    cycle_sims = similarities[has_cycle == 1]
    
    data = [perm_sims, nilp_sims, imm_idemp_sims, ev_idemp_sims, cycle_sims]
    plt.boxplot(data, labels=labels)
    plt.ylabel('Cosine Similarity with Discrepancy')
    plt.title('Face Properties vs Similarity')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'properties_vs_similarity.png'))
    
    print(f"\nVisualizations saved to {output_dir} directory")

def test_permutation_matrices(sizes: List[int], verbose: bool = False, seed: int = 42, 
                           max_powers: int = 40, tolerance: float = 1e-7) -> List[Dict[str, Any]]:
    """
    Test our theory about faces of permutation matrices and their relationship with the discrepancy.
    
    Args:
        sizes: List of permutation matrix sizes to analyze
        verbose: Whether to print verbose information
        seed: Random seed for reproducibility
        max_powers: Maximum number of matrix powers to check for cycles
        tolerance: Numerical tolerance for matrix comparisons
        
    Returns:
        List of all analysis results
    """
    all_results = []
    
    for n in sizes:
        print(f"\n{'='*50}")
        print(f"ANALYZING PERMUTATION MATRIX OF SIZE {n}")
        print(f"{'='*50}")
        
        p = create_permutation_matrix(n, seed=seed)
        discrepancy = compute_discrepancy(p)
        similarities = compute_face_similarity(p, discrepancy)
        diagonal_entries = np.diag(p)
        
        if verbose:
            print(f"Permutation matrix of size {n}:")
            print(p)
            print(f"discrepancy norm: {np.linalg.norm(discrepancy):.6f}")
            print(f"discrepancy:\n{discrepancy}")
        
        matrix_results = analyze_matrix_faces(p, n, similarities, diagonal_entries, verbose, max_powers, tolerance)
        all_results.extend(matrix_results)
        
        summarize_matrix_results(matrix_results, verbose)
    
    print_overall_statistics(all_results)
    
    return all_results

def analyze_matrix_faces(p: np.ndarray, n: int, similarities: List[float], diagonal_entries: np.ndarray, 
                         verbose: bool, max_powers: int, tolerance: float) -> List[Dict[str, Any]]:
    """
    Analyze the faces of a permutation matrix.
    
    Args:
        p: Permutation matrix
        n: Size of the matrix
        similarities: List of similarities with discrepancy
        diagonal_entries: Diagonal entries of the permutation matrix
        verbose: Whether to print verbose information
        max_powers: Maximum number of matrix powers to check for cycles
        tolerance: Numerical tolerance for matrix comparisons
        
    Returns:
        List of analysis results for each face
    """
    matrix_results = []
    
    for i in range(n):
        face_i = face(p, i)
        is_perm = is_permutation_matrix(face_i)
        is_nilp, nilp_index = is_nilpotent(face_i, max_power=n)
        is_immediately_idempotent = is_idempotent(face_i, tolerance=tolerance)
        
        has_cycle, cycle_start, cycle_length, is_eventually_idempotent = check_face_cycles(
            face_i, is_nilp, is_immediately_idempotent, verbose, max_powers, tolerance)
        
        if verbose:
            print(f"Diagonal entry P[{i},{i}] = {diagonal_entries[i]}")
            print(f"Face {i} is {'a permutation matrix' if is_perm else 'not a permutation matrix'}")
            print(f"Face {i} is {'nilpotent (power ' + str(nilp_index) + ')' if is_nilp else 'not nilpotent'}")
            print(f"Face {i} is {'immediately idempotent' if is_immediately_idempotent else 'not immediately idempotent'}")
            print(f"Face {i} is {'eventually idempotent' if is_eventually_idempotent else 'not eventually idempotent'}")
            print(f"Face {i} is {'cyclic' if (has_cycle and cycle_length > 0) else 'not cyclic'}")
            print(f"Similarity with discrepancy: {similarities[i]:.6f}")
        
        result = {
            'face_index': i,
            'diagonal_entry': diagonal_entries[i],
            'is_perm': is_perm,
            'is_nilp': is_nilp,
            'nilp_index': nilp_index if is_nilp else 0,
            'is_immediately_idempotent': is_immediately_idempotent,
            'is_eventually_idempotent': is_eventually_idempotent,
            'has_cycle': has_cycle and cycle_length > 0,
            'cycle_start': cycle_start,
            'cycle_length': cycle_length,
            'similarity': similarities[i],
            'matrix_size': n
        }
        
        matrix_results.append(result)
    
    return matrix_results

def main():
    """
    Main function to execute permutation matrix analysis.
    """
    parser = argparse.ArgumentParser(description='Analyze permutation matrices and their faces.')
    parser.add_argument('--sizes', nargs='+', type=int, default=[3, 5, 7, 11, 13, 17, 19, 20, 21],
                        help='Sizes of permutation matrices to analyze')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--max-powers', type=int, default=220,
                        help='Maximum number of matrix powers to compute when checking for cycles')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output with full matrix details')
    parser.add_argument('--tolerance', type=float, default=1e-7,
                        help='Numerical tolerance for matrix comparisons')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Directory to save visualization plots')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print(f"Analyzing permutation matrices of sizes {args.sizes} with seed {args.seed}")
    if args.verbose:
        print("Verbose mode enabled - showing full details")
    
    # Run the analysis with explicit parameters - avoid global function modification
    all_results = test_permutation_matrices(args.sizes, verbose=args.verbose, seed=args.seed)
    
    # Generate visualizations in the specified output directory
    generate_visualizations(all_results, args.output_dir)

if __name__ == "__main__":
    main()
