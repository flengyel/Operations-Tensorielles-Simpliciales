import sympy as sp
import numpy as np
from typing import Tuple, List
from symbolic_tensor_ops import SymbolicTensor
# This file contains the function to construct the relative homology basis
# for a given shape and index of the missing face in the horn.

# To do: use the Dask function compute_missing_indices_dask() instead.

def get_symbols_from_tensor(T: SymbolicTensor) -> set:
    """Helper function to get all unique symbols from a tensor."""
    symbols = set()
    for entry in T.tensor.flatten():
        symbols.update(entry.free_symbols)
    return symbols

def construct_relative_homology_basis(shape: Tuple[int, ...], j: int) -> List[SymbolicTensor]:
    """
    Constructs an explicit basis for the relative homology group H_n(A,L)
    by identifying the missing indices.

    Args:
        shape: The shape of the initial tensor T.
        j: The index of the missing face in the horn Λ^n_j.

    Returns:
        A list of basis tensors, where each tensor corresponds to a
        generator of the relative homology group.
    """
    # 1. Create the original tensor to identify all possible variables
    T_orig = SymbolicTensor(shape, init_type='range')
    all_symbols = get_symbols_from_tensor(T_orig)
    
    # 2. Determine the symbols present in the horn's faces
    n = T_orig.dimen()
    symbols_in_horn_faces = set()
    for i in range(n + 1):
        if i == j:
            continue
        face_i = T_orig.face(i)
        symbols_in_horn_faces.update(get_symbols_from_tensor(face_i))
        
    # 3. The missing symbols correspond to the generators of the homology
    missing_symbols = all_symbols - symbols_in_horn_faces
    
    # 4. For each missing symbol, create a basis tensor with a 1 in its place
    basis_tensors = []
    symbol_to_index_map = {v: k for k, v in np.ndenumerate(T_orig.tensor)}
    
    for symbol in sorted(list(missing_symbols), key=str):
        idx = symbol_to_index_map[symbol]
        basis_tensor_data = np.zeros(shape, dtype=int)
        basis_tensor_data[idx] = 1
        basis_tensors.append(SymbolicTensor(shape, tensor=basis_tensor_data))
        
    return basis_tensors

if __name__ == "__main__":
    shape = (3, 3, 3)
    j = 1
    n = min(shape) - 1

    print(f"Constructing homology basis for shape={shape} (k=3, n=2), horn Λ²_{j}...")

    homology_basis = construct_relative_homology_basis(shape, j)
    rank = len(homology_basis)

    print(f"\nCalculation complete.")
    print(f"Computed Homological Rank: {rank}")

    # We know from your paper and filler algorithm that the theoretical rank is 12
    if rank == 12:
        print("✅ This result correctly matches the theoretical rank of 12.")
    else:
        print(f"❌ This result (rank={rank}) does not match the theoretical rank of 12.")

    # You can optionally print one of the basis vectors
    if homology_basis:
        print("\nExample basis tensor (one of 12):")
        print(homology_basis[0])