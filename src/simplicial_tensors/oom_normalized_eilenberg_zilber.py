import numpy as np
import itertools

# ==============================================================================
# 1. Core Simplicial Operators (Axis-wise)
# ==============================================================================

def face(T, axis, i):
    """Removes slice i from the specified axis."""
    n_axis = T.shape[axis] -1
    if not (0 <= i <= n_axis):
        raise IndexError(f"Face index {i} is out of bounds for axis {axis} with size {T.shape[axis]}")
    return np.delete(T, i, axis=axis)

def degen(T, axis, i):
    """Duplicates slice i along the specified axis by inserting it at position i."""
    if not (0 <= i < T.shape[axis]):
        raise IndexError(f"Degen index {i} is out of bounds for axis {axis} with size {T.shape[axis]}")
    return np.insert(T, i, T.take(i, axis=axis), axis=axis)

# ==============================================================================
# 2. Corrected Normalization Projector
# ==============================================================================

def normalize_tensor(M):
    """
    Applies the normalization projector P = product(I - s_i d_i) to a tensor M
    by sequentially killing degeneracies along each axis.
    """
    P = M.copy()
    n = min(P.shape) - 1 if P.shape else -1
    if n < 0: return P

    # For each axis, kill off all degeneracies for that axis
    for axis in range(P.ndim):
        # Apply projectors P_i = (I - s_i d_i) for i=n-1..0 along the current axis
        temp_P_axis = P.copy()
        for i in range(n - 1, -1, -1):
            try:
                # This is P_new = P_old - s_i(d_i(P_old))
                correction = degen(face(temp_P_axis, axis, i), axis, i)
                temp_P_axis = temp_P_axis - correction
            except IndexError:
                # This axis may be too small for this face/degen index, so we skip.
                pass
        P = temp_P_axis # Update P with the result of this axis's normalization
    return P

# ==============================================================================
# 3. Basis Construction Algorithm
# ==============================================================================

def get_missing_indices(shape, j):
    """Computes the set of index tuples that are 'missing' from the j-th horn."""
    n = min(shape) - 1
    if not (0 <= j <= n): raise ValueError(f"Horn index {j} out of bounds")
    horn_face_indices = set(range(n + 1)) - {j}
    missing_indices = [
        idx for idx in itertools.product(*(range(s) for s in shape))
        if horn_face_indices.issubset(set(idx))
    ]
    return missing_indices

def construct_homology_basis(shape, j):
    """
    Constructs a basis for the relative homology group H_n(A, L_j) by
    creating and normalizing an elementary tensor for each missing index.
    """
    n = min(shape) - 1
    missing_indices = get_missing_indices(shape, j)
    basis_vectors = []
    print(f"Found {len(missing_indices)} missing indices for horn Λ^{n}_{j}. Constructing basis...")
    for idx in missing_indices:
        Z = np.zeros(shape, dtype=int)
        Z[idx] = 1
        # Re-insert the normalization step as requested
        normalized_Z = normalize_tensor(Z)
        basis_vectors.append(normalized_Z)
    return basis_vectors

def build_tensor_basis_map(initial_tensors: list, max_dim: int):
    """Generates the basis for the simplicial vector space, organized by dimension."""
    basis = {k: {} for k in range(max_dim + 2)}
    queue = list(initial_tensors)
    # Use a hashable key for the visited set
    visited_keys = { (t.shape, t.tobytes()) for t in queue }

    head = 0
    while head < len(queue):
        s_tensor = queue[head]; head += 1
        k_dim = min(s_tensor.shape) - 1 if s_tensor.shape else -1
        if k_dim < 0: continue
        
        key = (s_tensor.shape, s_tensor.tobytes())
        basis.setdefault(k_dim, {})[key] = s_tensor

        # Generate faces
        if k_dim > 0:
            for i in range(k_dim + 1):
                # Diagonal faces for simplicity in basis generation
                f_tensor = face(s_tensor, 0, i) 
                f_key = (f_tensor.shape, f_tensor.tobytes())
                if min(f_tensor.shape) -1 >= 0 and f_key not in visited_keys:
                    visited_keys.add(f_key); queue.append(f_tensor)
        
        # Generate degeneracies
        if k_dim < max_dim + 1:
            for i in range(k_dim + 1):
                g_tensor = degen(s_tensor, 0, i) # Diagonal degen
                g_key = (g_tensor.shape, g_tensor.tobytes())
                if g_key not in visited_keys:
                    visited_keys.add(g_key); queue.append(g_tensor)
    
    return {k: list(v.values()) for k, v in basis.items() if v}


# ==============================================================================
# 4. Eilenberg-Zilber Chain Complex Builder
# ==============================================================================

def build_ez_chain_complex(basis_map: dict, max_dim: int):
    """
    Builds the chain complex for a tensor product space using the
    Eilenberg-Zilber total differential with the correct Koszul sign.
    """
    differentials = {}
    if not any(basis_map.values()): return differentials
    sample_tensor = list(basis_map.values())[0][0]
    num_axes = len(sample_tensor.shape)

    for k in range(1, max_dim + 2):
        C_k = basis_map.get(k, [])
        C_km1 = basis_map.get(k - 1, [])
        if not C_k or not C_km1: continue
        
        C_km1_indices = { (t.shape, t.tobytes()): i for i, t in enumerate(C_km1) }
        total_d_k = np.zeros((len(C_km1), len(C_k)), dtype=int)

        for axis in range(num_axes):
            d_k_axis = np.zeros((len(C_km1), len(C_k)), dtype=int)
            for j_col, t_k in enumerate(C_k):
                # Corrected face range: 0 to k
                for i_face in range(k + 1):
                    try:
                        face_t_axis = face(t_k, axis, i_face)
                        key_f = (face_t_axis.shape, face_t_axis.tobytes())
                        if key_f in C_km1_indices:
                            row_idx = C_km1_indices[key_f]
                            d_k_axis[row_idx, j_col] += (-1)**i_face
                    except IndexError:
                        continue
            
            sign = (-1)**axis
            total_d_k += sign * d_k_axis

        differentials[k] = total_d_k
        
    return differentials

def compute_homology_rank(differentials, k):
    """Computes dim H_k = dim C_k - rank(d_k) - rank(d_{k+1})"""
    d_k = differentials.get(k, np.array([[]]))
    d_k_plus_1 = differentials.get(k + 1, np.array([[]]))
    
    dim_Ck = d_k.shape[1]
    rank_dk = np.linalg.matrix_rank(d_k) if d_k.size > 0 else 0
    rank_dk_plus_1 = np.linalg.matrix_rank(d_k_plus_1) if d_k_plus_1.size > 0 else 0
    
    return dim_Ck - rank_dk - rank_dk_plus_1

# ==============================================================================
# 5. Test and Verification Suite
# ==============================================================================

def main():
    shape = (3, 3, 3)
    n = min(shape) - 1
    j = 1
    
    print(f"--- Full Homological Test for shape={shape}, horn=Λ^{n}_{j} ---")
    
    # 1. Build the basis maps for the full complex (A) and the horn (L)
    # A is generated by the n-simplex of all 1s
    T_A = np.ones(shape, dtype=int)
    basis_map_A = build_tensor_basis_map([T_A], n)
    
    # L is generated by the faces of T_A
    horn_generators = [face(T_A, 0, i) for i in range(n+1) if i != j]
    basis_map_L = build_tensor_basis_map(horn_generators, n)
    
    # 2. Build the Eilenberg-Zilber differentials for A
    print("\nBuilding differential for C(A)...")
    differentials_A = build_ez_chain_complex(basis_map_A, n)
    
    # 3. Verify d*d=0 for A
    print("Verifying d*d=0 for C(A)...")
    for k in range(1, n + 1):
        d_k = differentials_A.get(k)
        d_kp1 = differentials_A.get(k+1)
        if d_k is not None and d_kp1 is not None:
            prod = d_k @ d_kp1
            if np.any(prod):
                print(f"  ❌ FAILURE at k={k}: d_{k} * d_{k+1} is NOT zero!")
            else:
                print(f"  ✅ SUCCESS at k={k}: d_{k} * d_{k+1} is zero.")
    
    # 4. Compute homology of A (should be trivial)
    print("\nComputing homology of C(A)...")
    for k in range(1, n + 1):
        rank_Hk = compute_homology_rank(differentials_A, k)
        print(f"  Rank H_{k}(A) = {rank_Hk}")
    
    # 5. Compute rank of relative homology H_n(A,L)
    # This requires building the mapping cone, which is a more advanced step.
    # As a proxy, we compare the basis sizes.
    
    # The rank of H_n(A,L) is related to the number of basis elements in C_n(A)
    # that are not in C_n(L).
    
    basis_A_n_keys = set((t.shape, t.tobytes()) for t in basis_map_A.get(n, []))
    basis_L_n_keys = set((t.shape, t.tobytes()) for t in basis_map_L.get(n, []))
    
    relative_basis_count = len(basis_A_n_keys - basis_L_n_keys)
    
    print("\n--- Final Result ---")
    print(f"Direct computation of dim(C_{n}(A)/C_{n}(L)) = {relative_basis_count}")
    
    # Compare with the combinatorial formula
    combinatorial_rank = len(get_missing_indices(shape, j))
    print(f"Combinatorial formula rank C(s,n,j) = {combinatorial_rank}")
    
    if relative_basis_count == combinatorial_rank:
        print("\n✅ SUCCESS: The homological basis size matches the combinatorial rank.")
    else:
        print("\n❌ FAILURE: Discrepancy found between homological model and combinatorial formula.")
    
