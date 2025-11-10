# sagemath_normalized_tensor_homology.py
# Final, definitive version implementing homology with pure Python/SymPy.

import sympy as sp
import numpy as np
from typing import Tuple, List, Union

# ==============================================================================
# 1. STABLE SymbolicTensor CLASS
# ==============================================================================
class SymbolicTensor:
    def __init__(self, shape: Tuple[int, ...], tensor=None, init_type: str = 'range'):
        self.shape = shape
        if tensor is not None:
            self.tensor = np.array(tensor, dtype=object)
        else:
            self.tensor = np.empty(shape, dtype=object)
            for idx in np.ndindex(shape):
                idx_str = ','.join(map(str, idx))
                if init_type == 'range':
                    self.tensor[idx] = sp.Symbol(f'x_{{{idx_str}}}')
                elif init_type == 'zeros':
                    self.tensor[idx] = sp.S.Zero

    def __sub__(self, other: "SymbolicTensor") -> "SymbolicTensor":
        if self.shape != other.shape: raise ValueError("Shape mismatch")
        return SymbolicTensor(self.shape, tensor=(self.tensor - other.tensor))

    def __add__(self, other: "SymbolicTensor") -> "SymbolicTensor":
        if self.shape != other.shape: raise ValueError("Shape mismatch")
        return SymbolicTensor(self.shape, tensor=(self.tensor + other.tensor))

    def dimen(self) -> int:
        if not self.shape: return -1
        return min(self.shape) - 1

    def _dims(self):
        return tuple([np.arange(start=0, stop=dim_size) for dim_size in self.shape])
    
    def face(self, i: int):
        d = self.dimen() + 1
        if not (0 <= i < d): raise IndexError(f"Face index {i} out of bounds")
        axes = self._dims()
        indices = [np.delete(axes[dim], i) for dim in range(len(self.shape))]
        new_shape = tuple(s - 1 for s in self.shape)
        if not all(s > 0 for s in new_shape): return SymbolicTensor(new_shape, init_type='zeros')
        grid = np.ix_(*indices)
        return SymbolicTensor(self.tensor[grid].shape, tensor=self.tensor[grid])

    def degen(self, k: int):
        result = self.tensor
        for axis in range(result.ndim):
            if not (0 <= k < self.shape[axis]): raise IndexError(f"Degeneracy index {k} out of bounds")
            slices: list[Union[int, slice]] = [slice(None)] * result.ndim
            slices[axis] = k
            insert_slice = result[tuple(slices)]
            result = np.insert(result, k, insert_slice, axis=axis)
        return SymbolicTensor(result.shape, tensor=result)
    def __str__(self): return str(self.tensor)

def key_tensor(st: SymbolicTensor) -> tuple:
    return (st.shape, tuple(str(s) for s in st.tensor.flatten()))

# ==============================================================================
# 2. CORRECTED BUILDER AND COMPUTATION FUNCTIONS
# ==============================================================================

def build_tensor_basis(initial_tensors: list, max_dim: int):
    skeleton = {k: {} for k in range(max_dim + 2)}
    queue = list(initial_tensors)
    visited_keys = {key_tensor(t) for t in queue}
    head = 0
    while head < len(queue):
        s_tensor = queue[head]; head += 1
        k_dim = s_tensor.dimen()
        if k_dim < 0: continue
        skeleton.setdefault(k_dim, {})[key_tensor(s_tensor)] = s_tensor
        if k_dim > 0:
            for i in range(k_dim + 1):
                try:
                    f_tensor = s_tensor.face(i)
                    if f_tensor.dimen() >= 0 and key_tensor(f_tensor) not in visited_keys:
                        visited_keys.add(key_tensor(f_tensor)); queue.append(f_tensor)
                except IndexError: continue
    full_basis = {k: v.copy() for k, v in skeleton.items()}
    degen_queue = []
    for k in sorted(skeleton.keys()): degen_queue.extend(skeleton[k].values())
    head = 0
    while head < len(degen_queue):
        s_tensor = degen_queue[head]; head += 1
        k_dim = s_tensor.dimen()
        if k_dim < max_dim + 1:
            for i in range(k_dim + 1):
                try:
                    g_tensor = s_tensor.degen(i)
                    g_key = key_tensor(g_tensor)
                    gk_dim = g_tensor.dimen()
                    if g_key not in full_basis.get(gk_dim, {}):
                        full_basis.setdefault(gk_dim, {})[g_key] = g_tensor
                        degen_queue.append(g_tensor)
                except IndexError: continue
    return {k: list(v.values()) for k, v in full_basis.items() if v}

def build_differential_matrices(basis_map: dict, max_dim: int):
    """Builds the differential matrices for the standard chain complex."""
    differentials = {}
    for k in range(1, max_dim + 2):
        C_k = basis_map.get(k, [])
        C_km1_indices = {key_tensor(t): i for i, t in enumerate(basis_map.get(k - 1, []))}
        if not C_k: continue
        d_k = np.zeros((len(C_km1_indices), len(C_k)), dtype=object)
        for j, t_k in enumerate(C_k):
            for i in range(k + 1):
                try:
                    face_t = t_k.face(i)
                    key_f = key_tensor(face_t)
                    if key_f in C_km1_indices:
                        row_idx = C_km1_indices[key_f]
                        d_k[row_idx, j] += (-1)**i
                except IndexError: continue
        differentials[k] = d_k
    return differentials

def compute_homology_rank(diffs: dict, n: int):
    """Computes the rank of the n-th homology group using pure SymPy."""
    d_n = diffs.get(n)
    d_n_plus_1 = diffs.get(n + 1)
    
    mat_dn = sp.Matrix(d_n) if d_n is not None and d_n.size > 0 else sp.zeros(0, 0)
    mat_dn_plus_1 = sp.Matrix(d_n_plus_1) if d_n_plus_1 is not None and d_n_plus_1.size > 0 else sp.zeros(0, 0)
    
    dim_C_n = mat_dn.cols
    rank_dn = mat_dn.rank()
    rank_dn_plus_1 = mat_dn_plus_1.rank()
    
    dim_ker_dn = dim_C_n - rank_dn
    dim_im_dn_plus_1 = rank_dn_plus_1
    
    return dim_ker_dn - dim_im_dn_plus_1

# ==============================================================================
# 3. DEFINITIVE MAIN FUNCTION
# ==============================================================================

def compute_relative_homology_rank(T_symbolic: SymbolicTensor, j_horn_excluded_face: int):
    """
    Computes the relative homology rank using a pure Python/SymPy implementation.
    This is the definitive, corrected version.
    """
    n = T_symbolic.dimen()
    if n < 0: return 0

    # 1. Build bases and differentials for C(A) and C(L)
    basis_A = build_tensor_basis([T_symbolic], n)
    diffs_A = build_differential_matrices(basis_A, n)

    horn_gens = [T_symbolic.face(i) for i in range(n + 1) if i != j_horn_excluded_face]
    basis_L = build_tensor_basis(horn_gens, n)
    diffs_L = build_differential_matrices(basis_L, n)

    # 2. Pre-calculate the dimensions of all chain groups
    dims_A, dims_L = {}, {}
    for k, d in diffs_A.items():
        dims_A[k] = d.shape[1]
        dims_A[k-1] = d.shape[0]
    for k, d in diffs_L.items():
        dims_L[k] = d.shape[1]
        dims_L[k-1] = d.shape[0]

    # 3. Build the inclusion map i: C(L) -> C(A)
    inclusion_maps = {}
    for k in range(n + 2):
        L_k_basis, A_k_basis = basis_L.get(k, []), basis_A.get(k, [])
        if not L_k_basis: continue
        A_k_indices = {key_tensor(t): i for i, t in enumerate(A_k_basis)}
        incl_k = np.zeros((len(A_k_indices), len(L_k_basis)), dtype=object)
        for i, t in enumerate(L_k_basis):
            key_t = key_tensor(t)
            if key_t in A_k_indices: incl_k[A_k_indices[key_t], i] = 1
        inclusion_maps[k] = incl_k

    # 4. Manually construct the differentials for the mapping cone
    cone_diffs = {}
    max_deg = max(list(dims_A.keys()) + list(dims_L.keys()))
    for k in range(1, max_deg + 2):
        # Get components, defaulting to None
        d_A_k = diffs_A.get(k)
        d_L_km1 = diffs_L.get(k-1)
        i_km1 = inclusion_maps.get(k-1)

        # Get dimensions from our pre-calculated maps
        dim_L_km1 = dims_L.get(k-1, 0)
        dim_L_km2 = dims_L.get(k-2, 0)
        dim_A_k = dims_A.get(k, 0)
        dim_A_km1 = dims_A.get(k-1, 0)

        # Create large zero matrix for the cone differential
        cone_d_k = np.zeros((dim_L_km2 + dim_A_km1, dim_L_km1 + dim_A_k), dtype=object)

        # Paste in the components where they exist
        if d_L_km1 is not None and d_L_km1.size > 0:
            cone_d_k[0:dim_L_km2, 0:dim_L_km1] = -d_L_km1
        if i_km1 is not None and i_km1.size > 0:
            cone_d_k[dim_L_km2:, 0:dim_L_km1] = i_km1
        if d_A_k is not None and d_A_k.size > 0:
            cone_d_k[dim_L_km2:, dim_L_km1:] = d_A_k

        cone_diffs[k] = cone_d_k

    # 5. Compute the homology rank of the cone
    return compute_homology_rank(cone_diffs, n)


