import numpy as np
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt

# Import the boundary operator and face functions
from .tensor_ops import (bdry, face, dimen, random_real_matrix)


def compute_discrepancy(a: np.ndarray) -> np.ndarray:
    """
    Compute the discrepancy ∂(A²) - ∂(A)² for a given matrix A.
    """
    a_squared = np.matmul(a, a)
    bdry_a_squared = bdry(a_squared)
    bdry_a = bdry(a)
    squared_bdry_a = np.matmul(bdry_a, bdry_a)
    return bdry_a_squared - squared_bdry_a


def compute_face_similarity(a: np.ndarray, discrepancy: np.ndarray) -> List[float]:
    """
    Compute the cosine similarity between the discrepancy and each face of A.
    """
    similarities = []
    dim = dimen(a)
    
    for i in range(dim + 1):
        face_i = face(a, i)
        
        # Flatten both matrices to vectors for cosine similarity
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


def check_relative_error(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-6) -> bool:
    """
    Check if two matrices are approximately equal within relative error.
    """
    if a.shape != b.shape:
        return False
    
    a_norm = np.linalg.norm(a)
    diff_norm = np.linalg.norm(a - b)
    
    if a_norm < epsilon:  # Handle near-zero a
        return diff_norm < epsilon
    
    return diff_norm / a_norm < epsilon


def is_permutation_matrix(matrix):
    """
    Check if a matrix is a permutation matrix.
    A permutation matrix has exactly one 1 in each row and column, with all other elements being 0.
    """
    # Check if all elements are 0 or 1
    if not np.all(np.logical_or(matrix == 0, matrix == 1)):
        return False
    
    # Check if each row and column has exactly one 1
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    
    return np.all(row_sums == 1) and np.all(col_sums == 1)


def is_nilpotent(matrix, max_power=None, tolerance=1e-10):
    """
    Check if a matrix is nilpotent (some power of the matrix is zero).
    """
    if max_power is None:
        max_power = matrix.shape[0]
    
    current = np.copy(matrix)
    for i in range(1, max_power + 1):
        if np.all(np.abs(current) < tolerance):
            return True, i  # Return True and the nilpotent index
        current = np.matmul(current, matrix)
    
    return False, 0


def is_idempotent(matrix, tolerance=1e-10):
    """
    Check if a matrix is idempotent (M² = M).
    """
    squared = np.matmul(matrix, matrix)
    return np.allclose(squared, matrix, atol=tolerance)


def check_cycles(matrix, max_powers=40, tolerance=1e-7):
    """
    Check if a matrix exhibits cyclic behavior under repeated multiplication.
    
    Returns:
        (bool, int, int): Tuple containing:
            - Whether a cycle was detected
            - The power where the cycle starts
            - The length of the cycle
        OR
        (bool, int, 0): If eventual idempotence is found
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
    """
    rng = np.random.default_rng(seed=seed)
    perm = rng.permutation(n)
    P = np.zeros((n, n))
    for i, j in enumerate(perm):
        P[i, j] = 1
    return P


def analyze_face(p: np.ndarray, i: int, similarities: List[float], 
                 diagonal_entries: np.ndarray) -> Dict[str, Any]:
    """
    Analyze a face of a permutation matrix to identify its algebraic properties.
    
    Args:
        p: The permutation matrix
        i: The index of the face to analyze
        similarities: List of similarities between discrepancy and faces
        diagonal_entries: Diagonal entries of the permutation matrix
    
    Returns:
        Dictionary with analysis results
    """
    face_i = face(p, i)
    is_perm = is_permutation_matrix(face_i)
    is_nilp, nilp_index = is_nilpotent(face_i)
    is_immediately_idempotent = is_idempotent(face_i)
    
    has_cycle, cycle_start, cycle_length, is_eventually_idempotent = check_face_cycles(face_i, is_nilp, is_immediately_idempotent)
    
    print_status_summary(i, diagonal_entries, is_perm, is_nilp, nilp_index, is_immediately_idempotent, is_eventually_idempotent, has_cycle, cycle_length, similarities)
    
    verify_theorem(i, diagonal_entries, is_perm, is_nilp, is_immediately_idempotent, is_eventually_idempotent, has_cycle, cycle_length)
    
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


def print_face_info(i, face_i, face_squared):
    print(f"\nFace {i}:")
    print(face_i)
    print("Face^2 (squared):")
    print(face_squared)


def print_difference_info(difference):
    print("Face^2 - Face (difference):")
    print(difference)
    print(f"Max absolute difference: {np.max(np.abs(difference)):.10f}")


def check_face_cycles(face_i, is_nilp, is_immediately_idempotent):
    has_cycle = False
    cycle_start = 0
    cycle_length = 0
    is_eventually_idempotent = False
    
    if not is_nilp and not is_immediately_idempotent:
        has_cycle, cycle_start, cycle_length = check_cycles(face_i)
        
        if has_cycle:
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
    
    return has_cycle, cycle_start, cycle_length, is_eventually_idempotent


def print_status_summary(i, diagonal_entries, is_perm, is_nilp, nilp_index, is_immediately_idempotent, is_eventually_idempotent, has_cycle, cycle_length, similarities):
    print(f"Diagonal entry P[{i},{i}] = {diagonal_entries[i]}")
    print(f"Face {i} is {'a permutation matrix' if is_perm else 'not a permutation matrix'}")
    print(f"Face {i} is {'nilpotent (power ' + str(nilp_index) + ')' if is_nilp else 'not nilpotent'}")
    print(f"Face {i} is {'immediately idempotent' if is_immediately_idempotent else 'not immediately idempotent'}")
    print(f"Face {i} is {'eventually idempotent' if is_eventually_idempotent else 'not eventually idempotent'}")
    print(f"Face {i} is {'cyclic' if (has_cycle and cycle_length > 0) else 'not cyclic'}")
    print(f"Similarity with discrepancy: {similarities[i]:.6f}")


def verify_theorem(i, diagonal_entries, is_perm, is_nilp, is_immediately_idempotent, is_eventually_idempotent, has_cycle, cycle_length):
    expected_perm = (diagonal_entries[i] == 1)
    expected_special = (diagonal_entries[i] == 0)
    has_special_property = is_nilp or is_immediately_idempotent or is_eventually_idempotent or (has_cycle and cycle_length > 0)
    
    if expected_perm and not is_perm:
        print(f"⚠️ WARNING: Face {i} should be a permutation matrix but isn't")
    elif expected_special and not has_special_property:
        print(f"⚠️ WARNING: Face {i} should have a special property (nilpotent, idempotent, or cyclic) but doesn't")


def test_permutation_matrices():
    """
    Test our theory about faces of permutation matrices and their relationship with the discrepancy.
    """
    sizes = [3, 9, 11, 23]  # Test different sizes of permutation matrices
    all_results = []
    
    for n in sizes:
        print(f"\n{'='*50}")
        print(f"ANALYZING PERMUTATION MATRIX OF SIZE {n}")
        print(f"{'='*50}")
        
        p = create_permutation_matrix(n)
        discrepancy = compute_discrepancy(p)
        similarities = compute_face_similarity(p, discrepancy)
        diagonal_entries = np.diag(p)
        
        print(f"Permutation matrix of size {n}:")
        print(p)
        print(f"discrepancy norm: {np.linalg.norm(discrepancy):.6f}")
        print(f"discrepancy:\n{discrepancy}")
        
        matrix_results = []
        
        for i in range(n):
            result = analyze_face(p, i, similarities, diagonal_entries)
            result['matrix_size'] = n
            matrix_results.append(result)
            all_results.append(result)
        
        # Summarize results for this matrix
        summarize_matrix_results(matrix_results)
    
    # Print overall statistics from all matrices
    print_overall_statistics(all_results)
    
    return all_results


def summarize_matrix_results(results):
    """
    Summarize the results for a single permutation matrix.
    """
    # Count the occurrences of each property
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
    
    # Analyze relationship with discrepancy
    if results:
        # Find face with highest similarity to discrepancy
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


def print_overall_statistics(all_results):
    """
    Print overall statistics from all analyzed permutation matrices.
    """
    if not all_results:
        print("No results to analyze.")
        return
    
    total = len(all_results)
    
    # Count occurrences of each property
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


def test_discrepancy_conjecture():
    """
    Test the conjecture that the discrepancy ∂(A²) - ∂(A)² is similar
    to the faces of A for random matrices. Too bad it's false.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Test different matrix shapes (restrict to square matrices for the discrepancy conjecture)
    shapes = [(3, 3), (9, 9), (11, 11), (23, 23)]
    
    for shape in shapes:
        # Generate random matrix
        a = random_real_matrix(shape)
        
        # Compute discrepancy
        discrepancy = compute_discrepancy(a)
        
        # Compute similarity with each face
        similarities = compute_face_similarity(a, discrepancy)
        
        # Compute ratio of Frobenius norms
        ratios = compute_frobenius_ratios(a, discrepancy)
        
        # Print results
        print(f"\nMatrix shape: {shape}")
        print(f"discrepancy norm: {np.linalg.norm(discrepancy):.6f}")
        print("Face similarities:", [f"{s:.6f}" for s in similarities])
        print("Frobenius ratios:", [f"{r:.6f}" for r in ratios])
        
        # Check if discrepancy is approximately equal to any face or linear combination
        dim = dimen(a)
        for i in range(dim + 1):
            face_i = face(a, i)
            
            # Try to find a scaling factor that makes face_i ≈ discrepancy
            if np.linalg.norm(face_i) > 0 and np.linalg.norm(discrepancy) > 0:
                # Use least squares to find best scaling factor
                scaling = np.dot(face_i.flatten(), discrepancy.flatten()) / np.dot(face_i.flatten(), face_i.flatten())
                scaled_face = scaling * face_i
                
                if check_relative_error(discrepancy, scaled_face, epsilon=0.1):
                    print(f"Face {i} approximates discrepancy with scaling {scaling:.6f}")


def analyze_discrepancy_patterns(num_samples: int = 50, size: int = 5):
    """
    Analyze patterns in the discrepancy across many random matrices.
    """
    shape = (size, size)  # Ensure square matrix
    
    # Metrics to track
    all_similarities = []
    all_ratios = []
    discrepancy_norms = []
    
    for i in range(num_samples):
        # Generate random matrix with a different seed each time
        A = random_real_matrix(shape, seed=i+1000)
        
        # Compute discrepancy
        discrepancy = compute_discrepancy(A)
        discrepancy_norms.append(np.linalg.norm(discrepancy))
        
        # Compute metrics
        similarities = compute_face_similarity(A, discrepancy)
        ratios = compute_frobenius_ratios(A, discrepancy)
        
        all_similarities.append(similarities)
        all_ratios.append(ratios)
    
    # Convert to numpy arrays for analysis
    all_similarities = np.array(all_similarities)
    all_ratios = np.array(all_ratios)
    
    # Calculate statistics
    mean_similarities = np.mean(all_similarities, axis=0)
    std_similarities = np.std(all_similarities, axis=0)
    
    mean_ratios = np.mean(all_ratios, axis=0)
    std_ratios = np.std(all_ratios, axis=0)
    
    # Print results
    print(f"\nAnalysis of {num_samples} random matrices of shape {shape}")
    print(f"Average discrepancy norm: {np.mean(discrepancy_norms):.6f}")
    
    print("\nAverage similarities with each face:")
    for i in range(len(mean_similarities)):
        print(f"Face {i}: {mean_similarities[i]:.6f} ± {std_similarities[i]:.6f}")
    
    print("\nAverage Frobenius norm ratios:")
    for i in range(len(mean_ratios)):
        print(f"Face {i}: {mean_ratios[i]:.6f} ± {std_ratios[i]:.6f}")
    
    # Plot distributions
    plt.figure(figsize=(15, 10))
    
    # Plot similarity distributions
    plt.subplot(2, 1, 1)
    plt.boxplot(all_similarities)
    plt.ylabel('Cosine Similarity')
    plt.title('Distribution of Cosine Similarities between discrepancy and Faces')
    plt.xticks(range(1, shape[0] + 1), [f'Face {i}' for i in range(shape[0])])
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Plot ratio distributions
    plt.subplot(2, 1, 2)
    plt.boxplot(all_ratios)
    plt.ylabel('Ratio of Frobenius Norms')
    plt.title('Distribution of Ratios: ||discrepancy|| / ||Face||')
    plt.xticks(range(1, shape[0] + 1), [f'Face {i}' for i in range(shape[0])])
    plt.axhline(y=1, color='r', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('discrepancy_analysis.png')
    plt.close()


def main():
    print("Testing discrepancy conjecture for random matrices...")
    test_discrepancy_conjecture()
    
    print("\nTesting discrepancy conjecture for permutation matrices...")
    results = test_permutation_matrices()
    
    print("\nPerforming statistical analysis of discrepancy patterns...")
    analyze_discrepancy_patterns(num_samples=100, size=5)
