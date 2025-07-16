# sagemath_normalized_tensor_homology.py

# SageMath script to compute H_n(A_•(T), Λ^n_j(T)) for a symbolic tensor T

import numpy as np
import sympy as sp # For SymbolicTensor
from collections import deque

# SageMath imports
from sage.all import matrix, vector, block_matrix, identity_matrix, Matrix, Integer, Hom
from sage.all import QQ # Rational Field, for robust linear algebra
from sage.homology.chain_complex import ChainComplex

# --- SymbolicTensor class (adapted from symbolic_tensor_ops.py) ---

class SymbolicTensor:

    def __init__(self, shape: tuple[int, ...], tensor_data=None, init_type: str = 'range'):
        """Initialize a symbolic tensor."""
        self.shape = shape

        if tensor_data is not None:
            self.tensor = np.array(tensor_data, dtype=object) # Ensure it's a numpy array of objects
            if self.tensor.shape != self.shape:
                raise ValueError(f"Provided tensor_data shape {self.tensor.shape} does not match specified shape {self.shape}")
        else:
            self.tensor = np.empty(shape, dtype=object)
            for idx in np.ndindex(shape):
                idx_str = ','.join(map(str, idx))
                if init_type == 'range':
                    self.tensor[idx] = sp.Symbol(f'x_{{{idx_str}}}')
                elif init_type == 'zeros':
                    self.tensor[idx] = sp.S.Zero
                elif init_type == 'ones':
                    self.tensor[idx] = sp.S.One
                else:
                    raise ValueError(f"Unsupported init_type: {init_type}")

    def __sub__(self, other: "SymbolicTensor") -> "SymbolicTensor":
        if not isinstance(other, SymbolicTensor):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError(f"Cannot subtract tensors of different shapes {self.shape} vs {other.shape}")
        diff_data = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            diff_data[idx] = self.tensor[idx] - other.tensor[idx]
        return SymbolicTensor(self.shape, tensor_data=diff_data)
    
    @staticmethod
    def from_tensor(tensor_data_np_array):
        """Create a SymbolicTensor from an existing numpy array of symbolic expressions"""
        shape = tensor_data_np_array.shape
        return SymbolicTensor(shape, tensor_data=tensor_data_np_array)

    def dimen(self) -> int:
        """Simplicial dimension of the tensor."""
        if not self.shape: return -1 # Empty shape
        return min(self.shape) - 1

    def order(self) -> int:
        """Order (number of axes) of the tensor."""
        return len(self.shape)

    def _dims_ranges(self):
        """Helper to get ranges for each dimension's indices."""
        return tuple([np.arange(start=0, stop=dim_size) for dim_size in self.shape])
    
    def face(self, i: int) -> "SymbolicTensor":
        """Apply the i-th simplicial face operation."""
        # d is the number of possible faces (simplicial dimension + 1)
        # For a tensor of shape (n1, n2, ...), its simplicial dimension is min(ni)-1.
        # The number of faces d_0, ..., d_{min(ni)-1} is min(ni).
        num_faces = min(self.shape) if self.shape else 0
        if not (0 <= i < num_faces): # i is the index to be removed
            raise IndexError(f"Face index {i} out of bounds for {num_faces} possible faces (tensor dim {self.dimen()}). Shape: {self.shape}")

        if any(s <= 1 for s in self.shape): # Cannot take face if any dimension is 1 or less
            # This implies new shape would have a zero dimension.
            # Result is typically an empty tensor or tensor in lower category.
            # For now, let's return a tensor with a zero in its shape if that's the outcome.
            new_shape = tuple(s - 1 for s in self.shape)
            if any(s == 0 for s in new_shape if self.dimen() == 0): # Special case for 0-dim tensor face
                 return SymbolicTensor(new_shape, init_type='zeros') # Or handle as empty
            # Fall through if new_shape is valid (all positive)

        axis_indices_ranges = self._dims_ranges()
        # For each dimension of the tensor, delete the i-th slice/index
        try:
            indices_after_deletion = [np.delete(axis_range, i) for axis_range in axis_indices_ranges]
        except IndexError as e: # If i is too large for a particular axis_range
            # This can happen if i >= size of a particular dimension, but i < min(shape)
            # This implies the definition of face needs careful interpretation for non-cubic tensors.
            # The provided definition T o (delta_i x ... x delta_i) implies 'i' must be valid for all axes.
            # np.delete(arr, i) works if i < len(arr).
            # If min(shape) is used to bound i, then i is valid for all axes.
            raise IndexError(f"Error deleting index {i} from axis ranges. Shape: {self.shape}. Details: {e}")


        # np.ix_ creates a meshgrid for advanced indexing
        grid = np.ix_(*indices_after_deletion)
        face_tensor_data = self.tensor[grid]
        return SymbolicTensor(face_tensor_data.shape, tensor_data=face_tensor_data)

    def degen(self, k: int) -> "SymbolicTensor":
        """Apply k-th simplicial degeneracy."""
        # k is the index to be duplicated. Typically 0 <= k <= self.dimen().
        # The new tensor will have shape components s+1.
        if not self.shape: # Cannot degenerate empty shape tensor
            return SymbolicTensor(self.shape) 

        # Check if k is a valid index to duplicate for all dimensions
        # k must be < current size of dimension for np.insert to work relative to index k
        # and k must be <= current size if k is the position for insertion
        if not all(0 <= k < s for s in self.shape):
             # If k is to be an index that is duplicated, it must exist.
             # If k is an insertion point, it can be up to s.
             # np.insert(arr, k, val) inserts val *before* index k.
             # The slice to duplicate is self.tensor[..., kth_slice_along_axis, ...]
             pass # Let np.insert handle index errors for now, or add more specific checks.


        current_tensor_data = self.tensor
        new_ndim = current_tensor_data.ndim
        
        for axis in range(new_ndim):
            # Check if k is a valid index for slicing this axis
            if not (0 <= k < current_tensor_data.shape[axis]):
                raise IndexError(f"Degeneracy index {k} out of bounds for axis {axis} with size {current_tensor_data.shape[axis]}")

            # Create slice object to extract the k-th slice along the current axis
            slice_obj_parts = [slice(None)] * new_ndim # all elements ':'
            slice_obj_parts[axis] = k 
            slice_to_duplicate = current_tensor_data[tuple(slice_obj_parts)]
            
            # Insert the duplicated slice at position k (which means before original index k)
            current_tensor_data = np.insert(current_tensor_data, k, slice_to_duplicate, axis=axis)
            
        return SymbolicTensor(current_tensor_data.shape, tensor_data=current_tensor_data)

    def __str__(self):
        return str(self.tensor)
    
    def __repr__(self):
        return f"SymbolicTensor(shape={self.shape}, data={self.tensor})"

# --- End of SymbolicTensor class ---

def key_tensor(st: SymbolicTensor) -> tuple:
    """Generates a unique, hashable key for a SymbolicTensor."""
    # Convert sympy expressions to strings to ensure hashability and canonical form for comparison
    # Sorting might be needed if order of elements in flatten() isn't guaranteed or if symbols are complex
    try:
        # Attempt to make symbols consistently stringified for the key
        flat_symbols = tuple(str(sp.sympify(s).expand().simplify()) for s in st.tensor.flatten())
    except Exception: # Fallback if sympify or flatten fails for some reason
        flat_symbols = tuple(str(s) for s in st.tensor.flatten())
    return (st.shape, flat_symbols)


def build_all_tensors(T_init_symbolic: SymbolicTensor, is_horn=False, j_horn_excluded_face=None):
    """
    Generates all unique tensors for a simplicial object (or horn)
    starting from T_init_symbolic.
    T_init_symbolic is assumed to be a simplex of its own simplicial dimension.
    """
    max_s_dim = T_init_symbolic.dimen() # Simplicial dimension of the initial tensor
    # We need to generate tensors up to dimension max_s_dim + 1 for H_{max_s_dim}
    
    all_tensors_by_dim = {k: {} for k in range(max_s_dim + 3)} # Stores key_tensor(t): t
    
    queue = deque()
    initial_simplices_for_queue = []

    if not is_horn:
        key_T = key_tensor(T_init_symbolic)
        all_tensors_by_dim[max_s_dim][key_T] = T_init_symbolic
        initial_simplices_for_queue.append((T_init_symbolic, max_s_dim))
    else: # Constructing a horn Lambda^{max_s_dim}_j (T_init)
        if max_s_dim < 0 : # Cannot form horn of empty or undefined dimension tensor
            print(f"Warning: Cannot form horn for tensor with simplicial dimension {max_s_dim}")
            return {}
        if max_s_dim == 0: # Horn of a 0-simplex (a point)
            # Faces of a 0-simplex: d_0. Lambda^0_0 means d_0 is excluded.
            # The generating set for the horn is empty. The simplicial object is trivial.
            print(f"Warning: Constructing horn Lambda^0_{j_horn_excluded_face} of a 0-simplex. Resulting object is trivial.")
            # No initial simplices to add to queue for generation.
        else: # max_s_dim >= 1
            # Faces of T_init_symbolic are (max_s_dim - 1)-simplices
            num_faces_of_T_init = T_init_symbolic.dimen() + 1 # = max_s_dim + 1
            for i in range(num_faces_of_T_init):
                if i == j_horn_excluded_face:
                    continue
                try:
                    f_i = T_init_symbolic.face(i)
                    # Dimension of f_i is max_s_dim - 1
                    if f_i.dimen() < 0 : # Face resulted in invalid tensor (e.g. shape with 0)
                        # print(f"Skipping face {i} of T_init (dim {f_i.dimen()}) as it's trivial for horn.")
                        continue
                    key_f_i = key_tensor(f_i)
                    if key_f_i not in all_tensors_by_dim[max_s_dim - 1]:
                        all_tensors_by_dim[max_s_dim - 1][key_f_i] = f_i
                        initial_simplices_for_queue.append((f_i, max_s_dim - 1))
                except Exception as e:
                    print(f"Error generating face {i} of T_init for horn: {e}")
                    continue

    visited_for_bfs = set() 
    for t, k_dim in initial_simplices_for_queue:
        queue.append((t, k_dim))
        visited_for_bfs.add(key_tensor(t))

    while queue:
        s_tensor, k_s_dim = queue.popleft() # s_tensor is a SymbolicTensor, k_s_dim is its simplicial dimension

        # Apply face maps (d_i : k_s_dim -> k_s_dim - 1)
        if k_s_dim > 0:
            num_faces_of_s = s_tensor.dimen() + 1
            for i_face_op in range(num_faces_of_s):
                try:
                    f_tensor = s_tensor.face(i_face_op)
                    fk_dim = f_tensor.dimen() 
                    
                    if fk_dim < -1 : continue # Should not happen with current SymbolicTensor.face logic
                                            # A dimen of -1 might mean an empty shape (e.g. (0,) )
                                            # which could be C_{-1}

                    key_f = key_tensor(f_tensor)
                    if fk_dim not in all_tensors_by_dim: all_tensors_by_dim[fk_dim] = {}
                    if key_f not in all_tensors_by_dim[fk_dim]:
                         all_tensors_by_dim[fk_dim][key_f] = f_tensor
                    
                    if key_f not in visited_for_bfs and fk_dim >=0 : # Only explore non-negative dim simplices further
                        visited_for_bfs.add(key_f)
                        queue.append((f_tensor, fk_dim))
                except Exception as e:
                    # print(f"Error applying face {i_face_op} to tensor (key: {key_tensor(s_tensor)}, dim {k_s_dim}): {e}")
                    continue
        
        # Apply degeneracy maps (s_i : k_s_dim -> k_s_dim + 1)
        # We need to generate C_{max_s_dim+1} for H_{max_s_dim}
        if k_s_dim < max_s_dim + 2 : # Allow degeneracies to go up to one dim higher than needed for boundary of top chains
            num_degen_ops_for_s = s_tensor.dimen() + 1 # s_i where i is from 0 to dim(s)
            for i_degen_op in range(num_degen_ops_for_s): 
                try:
                    g_tensor = s_tensor.degen(i_degen_op)
                    gk_dim = g_tensor.dimen()
                    
                    if gk_dim < 0 : continue # Should not happen

                    key_g = key_tensor(g_tensor)
                    if gk_dim not in all_tensors_by_dim: all_tensors_by_dim[gk_dim] = {}
                    if key_g not in all_tensors_by_dim[gk_dim]:
                        all_tensors_by_dim[gk_dim][key_g] = g_tensor

                    if key_g not in visited_for_bfs:
                        visited_for_bfs.add(key_g)
                        queue.append((g_tensor, gk_dim))
                except Exception as e:
                    # print(f"Error applying degen {i_degen_op} to tensor (key: {key_tensor(s_tensor)}, dim {k_s_dim}): {e}")
                    continue
    
    actual_tensors_by_dim_list = {
        k_dim: list(t_map.values()) 
        for k_dim, t_map in all_tensors_by_dim.items() 
        if t_map and k_dim <= max_s_dim + 2 and k_dim >=0 # Consider relevant positive dimensions
    }
    return actual_tensors_by_dim_list


def get_moore_chain_complex(tensors_by_dim_map: dict, max_homology_dim: int, base_ring=QQ):
    """
    Constructs the Moore chain complex.
    Returns a tuple: (ChainComplex_object, M_k_basis_as_cols_in_Ck_map)
    """
    C_k_basis_tensors_list = {} 
    C_k_basis_indices_map = {}  

    # Iterate up to max_homology_dim + 2 for C_k to define M_{max_homology_dim+1}
    # for d_0: M_{max_homology_dim+1} -> M_{max_homology_dim}
    for k_dim in range(max_homology_dim + 3): 
        unique_tensors_at_k = tensors_by_dim_map.get(k_dim, [])
        C_k_basis_tensors_list[k_dim] = unique_tensors_at_k
        C_k_basis_indices_map[k_dim] = {key_tensor(t): i for i, t in enumerate(unique_tensors_at_k)}

    print("\n--- DEBUG: Initial Chain Group Dimensions ---")
    for k, v in sorted(C_k_basis_tensors_list.items()):
        if v: # Only print non-empty
            print(f"  dim(C_{k}) = {len(v)}")
    print("-------------------------------------------\n")

    all_face_op_matrices = {} 
    # Iterate for d_i: C_k -> C_{k-1}
    for k_dim in range(max_homology_dim + 3): 
        if k_dim == 0: continue 
        
        C_k_current_basis = C_k_basis_tensors_list.get(k_dim, [])
        C_km1_current_indices = C_k_basis_indices_map.get(k_dim - 1, {})
        
        if not C_k_current_basis: continue

        num_rows_C_km1 = len(C_km1_current_indices)
        num_cols_C_k = len(C_k_current_basis)

        # Number of face operators d_0, ..., d_k (for C_k which has simplicial dim k)
        num_face_ops_for_Ck = k_dim + 1 # This assumes C_k contains k-simplices.
                                       # The k_dim here is the simplicial dimension.
                                       # So, d_i for i from 0 to k_dim.

        for i_face_op in range(num_face_ops_for_Ck):
            d_i_matrix = matrix(base_ring, num_rows_C_km1, num_cols_C_k, sparse=True)
            if num_rows_C_km1 == 0 and num_cols_C_k == 0:
                 all_face_op_matrices[(k_dim, i_face_op)] = d_i_matrix
                 continue
            if num_rows_C_km1 == 0 and num_cols_C_k > 0:
                 all_face_op_matrices[(k_dim, i_face_op)] = d_i_matrix
                 continue

            for j_col_idx, s_tensor in enumerate(C_k_current_basis): # s_tensor is SymbolicTensor
                try:
                    # Ensure s_tensor actually has simplicial dimension k_dim
                    if s_tensor.dimen() != k_dim:
                        # print(f"Warning: Tensor {key_tensor(s_tensor)} in C_{k_dim} basis has actual dim {s_tensor.dimen()}. Skipping face op for matrix.")
                        continue 
                    
                    face_result_tensor = s_tensor.face(i_face_op)
                    if face_result_tensor.dimen() < -1 : continue 

                    key_face_result = key_tensor(face_result_tensor)
                    if key_face_result in C_km1_current_indices:
                        i_row_idx = C_km1_current_indices[key_face_result]
                        d_i_matrix[i_row_idx, j_col_idx] = 1 
                except IndexError: # face index out of bounds for this specific tensor
                    # print(f"IndexError for face {i_face_op} on tensor {key_tensor(s_tensor)} (dim {s_tensor.dimen()}) in C_{k_dim}. Matrix entry remains 0.")
                    continue
                except Exception as e:
                    # print(f"Error computing face {i_face_op} for tensor in C_{k_dim} (idx {j_col_idx}) for matrix: {e}")
                    continue
            all_face_op_matrices[(k_dim, i_face_op)] = d_i_matrix
            
    M_k_basis_as_cols_in_Ck = {} 
    
    if 0 in C_k_basis_tensors_list and C_k_basis_tensors_list[0]:
        dim_C0 = len(C_k_basis_tensors_list[0])
        M_k_basis_as_cols_in_Ck[0] = identity_matrix(base_ring, dim_C0)
    else: 
        M_k_basis_as_cols_in_Ck[0] = matrix(base_ring, 0, 0)

    # Iterate for M_k up to max_homology_dim + 1
    for k_dim in range(1, max_homology_dim + 2): 
        C_k_current_basis = C_k_basis_tensors_list.get(k_dim, [])
        if not C_k_current_basis: 
            M_k_basis_as_cols_in_Ck[k_dim] = matrix(base_ring, 0, 0) 
            continue

        dim_Ck = len(C_k_current_basis)
        
        face_ops_for_Mk_kernel_matrices = []
        valid_ops_found = True
        # M_k = Intersection_{i=1..k} Ker(d_i: C_k -> C_{k-1})
        # d_i are indexed 0 to k_dim. So for M_k, we need d_1, ..., d_k (k_dim maps)
        num_kernel_face_ops = k_dim # Number of d_i maps (d1 to dk)
        
        for i_face_op_kernel in range(1, num_kernel_face_ops + 1): 
            op_matrix = all_face_op_matrices.get((k_dim, i_face_op_kernel))
            if op_matrix is None:
                if not C_k_basis_tensors_list.get(k_dim-1,[]): 
                    pass 
                else: 
                    print(f"Critical Error: d_{i_face_op_kernel} from C_{k_dim} to C_{k_dim-1} is missing for Moore complex construction.")
                    valid_ops_found = False; break
                continue 
            face_ops_for_Mk_kernel_matrices.append(op_matrix)
        
        if not valid_ops_found:
            M_k_basis_as_cols_in_Ck[k_dim] = matrix(base_ring, dim_Ck, 0) 
            continue

        if not face_ops_for_Mk_kernel_matrices: 
            M_k_basis_as_cols_in_Ck[k_dim] = identity_matrix(base_ring, dim_Ck) 
        # Find the original else block and replace it entirely with this:
        else:
            # Alternative method: Compute kernels individually and intersect them.
            try:
                # Get the list of face maps d_1, ..., d_k
                face_ops = [all_face_op_matrices.get((k_dim, i)) for i in range(1, k_dim + 1)]

                # Start with the full space C_k
                ambient_space = QQ**dim_Ck
                intersection_of_kernels = ambient_space

                # Intersect with the kernel of each face map
                for op_matrix in face_ops:
                    if op_matrix is not None:
                        kernel_subspace = op_matrix.right_kernel()
                        intersection_of_kernels = intersection_of_kernels.intersection(kernel_subspace)

                kernel_basis_matrix_rows = intersection_of_kernels.basis_matrix()
                M_k_basis_as_cols_in_Ck[k_dim] = kernel_basis_matrix_rows.transpose()

            except Exception as e:
                print(f"Error computing kernel for M_{k_dim} via intersection: {e}")
                M_k_basis_as_cols_in_Ck[k_dim] = matrix(base_ring, dim_Ck, 0)


    moore_complex_differential_matrices = {} 
    # Iterate for d_0_Moore: M_k -> M_{k-1}, up to M_{max_homology_dim+1} -> M_{max_homology_dim}
    for k_dim in range(1, max_homology_dim + 2): 
        Inc_Mk_to_Ck = M_k_basis_as_cols_in_Ck.get(k_dim)       
        Basis_Mkm1_in_Ckm1 = M_k_basis_as_cols_in_Ck.get(k_dim-1) 
        d0_from_Ck_to_Ckm1 = all_face_op_matrices.get((k_dim, 0)) 

        if Inc_Mk_to_Ck is None or Inc_Mk_to_Ck.ncols() == 0 or \
           Basis_Mkm1_in_Ckm1 is None or Basis_Mkm1_in_Ckm1.ncols() == 0 or \
           d0_from_Ck_to_Ckm1 is None:
            
            dim_M_k = Inc_Mk_to_Ck.ncols() if Inc_Mk_to_Ck is not None else 0
            dim_M_km1 = Basis_Mkm1_in_Ckm1.ncols() if Basis_Mkm1_in_Ckm1 is not None else 0
            
            # Ensure matrix is created even if one of M_k or M_{k-1} is trivial but not both
            # if (dim_M_k > 0 or dim_M_km1 > 0) and not (dim_M_k == 0 and dim_M_km1 == 0) :
            moore_complex_differential_matrices[k_dim] = matrix(base_ring, dim_M_km1, dim_M_k, 0)
            continue

        print(f"\n--- DEBUG FOR k_dim = {k_dim} ---")
        if d0_from_Ck_to_Ckm1 is not None:
            print(f"  d0_from_Ck_to_Ckm1 dimensions: {d0_from_Ck_to_Ckm1.dimensions()}")
        else:
            print("  d0_from_Ck_to_Ckm1 is None")

        if Inc_Mk_to_Ck is not None:
            print(f"  Inc_Mk_to_Ck dimensions: {Inc_Mk_to_Ck.dimensions()}")
        else:
            print("  Inc_Mk_to_Ck is None")
        print("--------------------------")

        img_d0_Mk_in_Ckm1 = d0_from_Ck_to_Ckm1 * Inc_Mk_to_Ck
        
        try:
            d0_Moore_matrix = Basis_Mkm1_in_Ckm1.solve_left(img_d0_Mk_in_Ckm1)
            moore_complex_differential_matrices[k_dim] = d0_Moore_matrix
        except Exception as e: 
            # print(f"Error computing d_0_Moore for M_{k_dim} -> M_{k_dim-1} via solve_left: {e}")
            dim_M_k = Inc_Mk_to_Ck.ncols()
            dim_M_km1 = Basis_Mkm1_in_Ckm1.ncols()
            moore_complex_differential_matrices[k_dim] = matrix(base_ring, dim_M_km1, dim_M_k, 0)

    # --- Build the final chain complex data with explicit Sage Integers for keys ---
    final_chain_complex_maps = {}
    for k in range(1, max_homology_dim + 2):
        if k in moore_complex_differential_matrices and moore_complex_differential_matrices[k] is not None:
            # Ensure the key `k` is a Sage Integer
            final_chain_complex_maps[Integer(k)] = moore_complex_differential_matrices[k]

    # As requested, add a final debug print
    print("\n--- DEBUG: Final data for ChainComplex constructor ---")
    for k, v in sorted(final_chain_complex_maps.items()):
        print(f"  Map for d_{k}: {v.dimensions()}")
    print("--------------------------------------------------\n")

    # The ChainComplex constructor in this version of Sage prefers a direct dictionary of maps.
    return ChainComplex(final_chain_complex_maps, base_ring=base_ring, degree=-1), M_k_basis_as_cols_in_Ck

def create_mapping_cone_manually(chain_map):
    """
    Manually constructs the mapping cone of a chain map f: C -> D.
    This version is the final solution, deriving all necessary
    information from the internal _diff attribute, bypassing the
    inconsistent public API entirely.
    """
    C = chain_map.domain()
    D = chain_map.codomain()
    
    # Step 1: Manually build dimension dictionaries from matrix shapes.
    c_dims = {}
    for degree, diff_matrix in C._diff.items():
        c_dims[degree] = diff_matrix.ncols()
        c_dims[degree - 1] = diff_matrix.nrows()
        
    d_dims = {}
    for degree, diff_matrix in D._diff.items():
        d_dims[degree] = diff_matrix.ncols()
        d_dims[degree - 1] = diff_matrix.nrows()

    cone_differentials = {}
    all_degs = set(c_dims.keys()) | set(d_dims.keys())

    if not all_degs:
        return ChainComplex({}, base_ring=C.base_ring())

    min_deg = min(all_degs) - 1
    max_deg = max(all_degs)
    
    # Step 2: Build the cone differentials using our new dimension maps.
    for n in range(min_deg, max_deg + 2):
        dC_nm1 = C._diff.get(n - 1)
        dD_n = D._diff.get(n)
        f_nm1 = chain_map._matrix_dictionary.get(n - 1)
        
        if dC_nm1 is None:
            dC_nm1 = matrix(C.base_ring(), c_dims.get(n - 2, 0), c_dims.get(n - 1, 0), 0)

        if dD_n is None:
            dD_n = matrix(D.base_ring(), d_dims.get(n - 1, 0), d_dims.get(n, 0), 0)
            
        if f_nm1 is None:
            f_nm1 = matrix(C.base_ring(), d_dims.get(n - 1, 0), c_dims.get(n - 1, 0), 0)
            
        zero_map = matrix(C.base_ring(), dC_nm1.nrows(), dD_n.ncols(), 0)
        cone_diff_n = block_matrix([[-dC_nm1, zero_map], [-f_nm1, dD_n]])
        
        if not cone_diff_n.is_zero():
            cone_differentials[Integer(n)] = cone_diff_n
            
    return ChainComplex(data=cone_differentials, base_ring=C.base_ring(), degree=-1)


def compute_relative_homology_sage(T_symbolic: SymbolicTensor, j_horn_excluded_face: int, base_ring=QQ):
    """
    Computes H_n(A_•(T), Λ^n_j(T)) using SageMath and Moore Complexes.
    T_symbolic: initial SymbolicTensor.
    j_horn_excluded_face: The 'j' in Λ^n_j. Index of the face excluded.
    base_ring: Field for computations.
    """
    n_homology_dim = T_symbolic.dimen() # n = dim(T)
    if n_homology_dim < 0:
        print(f"Initial tensor has non-positive simplicial dimension {n_homology_dim}. Homology is trivial.")
        return base_ring.zero() # Or appropriate representation of trivial homology group

    print(f"Building all tensors for A_•(T) (T is a {n_homology_dim}-simplex)...")
    tensors_A_map = build_all_tensors(T_symbolic, is_horn=False)
    # print(f"  Simplicial dimensions in A: {{k: len(v) for k,v in tensors_A_map.items() if v}}")

    print(f"Building all tensors for Horn Λ^{n_homology_dim}_{j_horn_excluded_face}(T)...")
    tensors_L_map = build_all_tensors(T_symbolic, is_horn=True, j_horn_excluded_face=j_horn_excluded_face)
    # print(f"  Simplicial dimensions in L: {{k: len(v) for k,v in tensors_L_map.items() if v}}")
    
    print("Constructing Moore chain complex M(A)...")
    Moore_A_obj, M_k_A_basis_map = get_moore_chain_complex(tensors_A_map, n_homology_dim, base_ring)
    
    print("Constructing Moore chain complex M(L)...")
    Moore_L_obj, M_k_L_basis_map = get_moore_chain_complex(tensors_L_map, n_homology_dim, base_ring)
    
    inclusion_chain_maps_M_L_to_M_A = {} 

    # C_k(L) basis tensors:
    C_k_L_basis_tensors_list = {}
    for k_dim_loop in range(n_homology_dim + 2): # Iterate up to n+1 for chain map
        C_k_L_basis_tensors_list[k_dim_loop] = tensors_L_map.get(k_dim_loop, [])
    
    # C_k(A) basis tensors and indices:
    C_k_A_basis_indices_map = {}
    for k_dim_loop in range(n_homology_dim + 2):
        A_tensors_k = tensors_A_map.get(k_dim_loop, [])
        C_k_A_basis_indices_map[k_dim_loop] = {key_tensor(t): i for i, t in enumerate(A_tensors_k)}
    
    for k_map_dim in range(n_homology_dim + 2): # Define map for M_k(L) -> M_k(A) for k from 0 to n+1
        B_Mk_L = M_k_L_basis_map.get(k_map_dim) 
        B_Mk_A = M_k_A_basis_map.get(k_map_dim) 

        dim_Mk_L = B_Mk_L.ncols() if B_Mk_L is not None else 0
        dim_Mk_A = B_Mk_A.ncols() if B_Mk_A is not None else 0

        if dim_Mk_L == 0: # Map from trivial M_k(L)
            inclusion_chain_maps_M_L_to_M_A[k_map_dim] = matrix(base_ring, dim_Mk_A, 0)
            continue
        if dim_Mk_A == 0: # Map to trivial M_k(A) (implies M_k(L) must also be trivial if it's a subcomplex)
            inclusion_chain_maps_M_L_to_M_A[k_map_dim] = matrix(base_ring, 0, dim_Mk_L)
            continue

        # Matrix for J_k: C_k(L) -> C_k(A) (inclusion of ambient chain groups)
        L_basis_k_list = C_k_L_basis_tensors_list.get(k_map_dim, [])
        A_indices_k_map = C_k_A_basis_indices_map.get(k_map_dim, {})
        
        dim_Ck_L = len(L_basis_k_list)
        dim_Ck_A = len(A_indices_k_map)
        
        if dim_Ck_L == 0: # C_k(L) is trivial, so M_k(L) must be. Already handled by dim_Mk_L == 0.
            inclusion_chain_maps_M_L_to_M_A[k_map_dim] = matrix(base_ring, dim_Mk_A, 0)
            continue

        J_k_matrix = matrix(base_ring, dim_Ck_A, dim_Ck_L, 0, sparse=True)
        for col_l_idx, tensor_l in enumerate(L_basis_k_list):
            key_l = key_tensor(tensor_l)
            if key_l in A_indices_k_map:
                row_a_idx = A_indices_k_map[key_l]
                J_k_matrix[row_a_idx, col_l_idx] = 1
        
        # Target matrix for i_k: M_k(L) -> M_k(A) is (B_Mk_A)^pseudo_inv * J_k * B_Mk_L
        # Manually handle the k=0 case, as M_0 = C_0.
        if k_map_dim == 0:
            inclusion_chain_maps_M_L_to_M_A[k_map_dim] = J_k_matrix
        else:
            try:
                # For k > 0, the original logic is fine.
                map_matrix_k = B_Mk_A.solve_left(J_k_matrix * B_Mk_L)
                inclusion_chain_maps_M_L_to_M_A[k_map_dim] = map_matrix_k
            except Exception as e:
                print(f"Error computing inclusion M_{k_map_dim}(L) -> M_{k_map_dim}(A): {e}")
                inclusion_chain_maps_M_L_to_M_A[k_map_dim] = matrix(base_ring, dim_Mk_A, dim_Mk_L, 0)

    chain_map_i_Moore = Hom(Moore_L_obj, Moore_A_obj)(inclusion_chain_maps_M_L_to_M_A)
    #print(type(chain_map_i_Moore))
    #print([name for name in dir(chain_map_i_Moore) if 'cone' in name.lower()])

    
    print(f"\nComputing H_{n_homology_dim}(Cone(i_Moore))...")
    cone_i_Moore = create_mapping_cone_manually(chain_map_i_Moore)
    # The homology group object itself (e.g., Free module of rank k over QQ)
    homology_group_n_object = cone_i_Moore.homology(n_homology_dim, base_ring=base_ring) 
    
    return homology_group_n_object


if __name__ == '__main__':
    print("--- Running Example with SymbolicTensors ---")
    
    # Example 1: T is a 1-simplex (e.g., shape (2,2))
    # Simplicial dimension n = min(2,2)-1 = 1.
    # We compute H_1(A(T), Lambda^1_j(T))
    try:
        print("\n--- Example 1: T is a 1-simplex (shape (2,2)) ---")
        # Create a symbolic 1-simplex (e.g. a 2x2 tensor)
        # T_1simplex_data = [[sp.Symbol('x_00'), sp.Symbol('x_01')],
        #                    [sp.Symbol('x_10'), sp.Symbol('x_11')]]
        # T1 = SymbolicTensor((2,2), tensor_data=T_1simplex_data)
        T1 = SymbolicTensor((2,2), init_type='range') # Uses x_{0,0}, x_{0,1}, ...
        n_homology_T1 = T1.dimen() # Should be 1

        print(f"Initial tensor T1 (dim={n_homology_T1}): {T1}")
        
        # Horns are Lambda^1_0 and Lambda^1_1
        for j_horn in range(n_homology_T1 + 1): 
            print(f"\n--- Computing for T1, H_{n_homology_T1}, Horn Λ^{n_homology_T1}_{j_horn} ---")
            Hn_rel = compute_relative_homology_sage(T1, 
                                                    j_horn_excluded_face=j_horn, 
                                                    base_ring=QQ)
            print(f"H_{n_homology_T1}(A(T1), Λ^{n_homology_T1}_{j_horn}; QQ) = {Hn_rel}")
            print(f"  Rank = {Hn_rel.rank()}")

    except Exception as e:
        print(f"An error occurred during Example 1: {e}")
        import traceback
        traceback.print_exc()

    # Example 2: T is a 2-simplex (e.g., shape (3,3,3))
    # Simplicial dimension n = min(3,3,3)-1 = 2.
    # We compute H_2(A(T), Lambda^2_j(T))
    try:
        print("\n\n--- Example 2: T is a 2-simplex (shape (3,3,3)) ---")
        T2 = SymbolicTensor((3,3,3), init_type='range')
        n_homology_T2 = T2.dimen() # Should be 2
        print(f"Initial tensor T2 (dim={n_homology_T2}), shape {T2.shape}")

        # Horns are Lambda^2_0, Lambda^2_1, Lambda^2_2
        for j_horn_2 in range(n_homology_T2 + 1): 
            print(f"\n--- Computing for T2, H_{n_homology_T2}, Horn Λ^{n_homology_T2}_{j_horn_2} ---")
            Hn_rel_2 = compute_relative_homology_sage(T2, 
                                                      j_horn_excluded_face=j_horn_2, 
                                                      base_ring=QQ)
            print(f"H_{n_homology_T2}(A(T2), Λ^{n_homology_T2}_{j_horn_2}; QQ) = {Hn_rel_2}")
            print(f"  Rank = {Hn_rel_2.rank()}")
            
    except Exception as e:
        print(f"An error occurred during Example 2: {e}")
        import traceback
        traceback.print_exc()

