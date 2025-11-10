#    Opérations Tensorielles Simpliciales
#    Simplicial Operations on Matrices and Hypermatrices
#    sagemath_compatible_tensor_ops.py
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

# This is the ULTIMATE, complete, and corrected version, intended to be loaded
# into a SageMath notebook. It merges ALL functionality from the user-provided
# files and uses Sage's symbolic engine. Nothing has been omitted.

import numpy as np
from typing import Tuple, List, Union

# Sage imports
# This script is intended to be run within a SageMath environment.
try:
    from sage.all import var, latex, simplify
    from sage.symbolic.expression import Expression
except ImportError:
    print("Warning: SageMath library not found. This script should be run in a Sage environment.")
    # Define dummy placeholders if not in Sage, so the file can be inspected
    def var(s): return str(s)
    def latex(s): return str(s)
    def simplify(s): return s
    class Expression: pass

class SimplicialException(Exception):
    """Custom exception for simplicial operations."""
    pass

class SymbolicTensor:
    """
    A tensor with symbolic entries, compatible with the SageMath ecosystem.
    """
    def __init__(self, shape: Tuple[int,...], tensor_data=None, init_type: str = 'range'):
        self.shape = tuple(map(int, shape)) # Ensure shape contains standard Python integers
        if tensor_data is not None:
            self.tensor = np.array(tensor_data, dtype=object)
            if self.tensor.shape != self.shape:
                raise ValueError(f"Provided data shape {self.tensor.shape} does not match specified shape {self.shape}")
        else:
            self.tensor = np.empty(self.shape, dtype=object)
            for idx in np.ndindex(self.shape):
                idx_str = '_'.join(map(str, idx))
                if init_type == 'range':
                    self.tensor[idx] = var(f'x_{idx_str}')
                elif init_type == 'zeros':
                    self.tensor[idx] = 0
                elif init_type == 'ones':
                    self.tensor[idx] = 1
                else:
                    raise ValueError(f"Unsupported init_type: {init_type}")

    @staticmethod
    def from_tensor(tensor_array):
        return SymbolicTensor(tensor_array.shape, tensor_data=tensor_array)

    def __str__(self): return str(self.tensor)
    def __repr__(self): return f"SymbolicTensor(shape={self.shape})"
    def __sub__(self, other):
        if not isinstance(other, SymbolicTensor) or self.shape != other.shape: return NotImplemented
        return SymbolicTensor(self.shape, tensor_data=(self.tensor - other.tensor))
    def __add__(self, other):
        if not isinstance(other, SymbolicTensor) or self.shape != other.shape: return NotImplemented
        return SymbolicTensor(self.shape, tensor_data=(self.tensor + other.tensor))

    def dimen(self) -> int:
        if not self.shape or any(s == 0 for s in self.shape): return -1
        return min(self.shape) - 1

    def order(self) -> int:
        return len(self.shape)

    def _dims(self):
        return tuple([np.arange(dim_size) for dim_size in self.shape])

    def face(self, i: int) -> "SymbolicTensor":
        num_faces = min(self.shape) if self.shape else 0
        if not (0 <= i < num_faces):
            raise IndexError(f"Face index {i} out of bounds for shape {self.shape}.")
        axes = self._dims()
        indices_for_grid = [np.delete(ax, i) for ax in axes]
        grid = np.ix_(*indices_for_grid)
        return SymbolicTensor.from_tensor(self.tensor[grid])

    def degen(self, k: int) -> "SymbolicTensor":
        n = self.dimen()
        if not (0 <= k <= n):
            raise IndexError(f"Degeneracy index {k} is out of bounds for dimension {n}.")
        result_data = self.tensor
        for axis in range(self.order()):
            slice_obj = [slice(None)] * self.order()
            slice_obj[axis] = k
            slice_to_duplicate = result_data[tuple(slice_obj)]
            result_data = np.insert(result_data, k, slice_to_duplicate, axis=axis)
        return SymbolicTensor.from_tensor(result_data)

    def bdry(self) -> "SymbolicTensor":
        n = self.dimen()
        if n < 0: raise SimplicialException("Boundary of tensor with dim < 0 is undefined.")
        result_shape = tuple(s - 1 for s in self.shape)
        result_tensor_data = np.zeros(result_shape, dtype=object)
        for i in range(n + 1):
            face_i = self.face(i)
            if (i % 2) == 0: result_tensor_data += face_i.tensor
            else: result_tensor_data -= face_i.tensor
        return SymbolicTensor.from_tensor(result_tensor_data)

    def horn(self, k: int) -> list:
        n = self.dimen()
        if not (0 <= k <= n): raise ValueError(f"Horn index {k} must be in [0, {n}]")
        faces = []
        for i in range(n + 1):
            if i == k:
                faces.append(SymbolicTensor(tuple(s - 1 for s in self.shape), init_type='zeros'))
            else:
                faces.append(self.face(i))
        return faces

    def filler(self, horn_list: list, k: int) -> "SymbolicTensor":
        g = horn_list[k].degen(0)
        for r in range(k):
            face_gr = g.face(r)
            diff_tensor = np.zeros(face_gr.shape, dtype=object)
            for idx in np.ndindex(face_gr.shape):
                diff_tensor[idx] = face_gr.tensor[idx] - horn_list[r].tensor[idx]
            degen_diff = SymbolicTensor(face_gr.shape, tensor_data=diff_tensor).degen(r)
            g.tensor = g.tensor - degen_diff.tensor
        t = len(horn_list) - 1
        while t > k:
            face_gt = g.face(t)
            diff_tensor = np.zeros(face_gt.shape, dtype=object)
            for idx in np.ndindex(face_gt.shape):
                diff_tensor[idx] = horn_list[t].tensor[idx] - face_gt.tensor[idx]
            degen_diff = SymbolicTensor(face_gt.shape, tensor_data=diff_tensor).degen(t - 1)
            g.tensor = g.tensor + degen_diff.tensor
            t -= 1
        return g
    
    def is_degen(self) -> bool:
        n = self.dimen()
        for i in range(n + 1):
            try:
                if np.all([(simplify(a - b) == 0) for a, b in zip(self.tensor.flatten(), self.face(i).degen(i).tensor.flatten())]):
                    return True
            except IndexError: continue
        return False

    def n_hypergroupoid_comparison(self, outer_horns=False, verbose=False, allow_degen=False):
        boundary = self.bdry()
        if not allow_degen and boundary.is_degen():
            if verbose: print("Boundary is degenerate.")
            raise SimplicialException("Degenerate boundary.")
        dim = self.dimen()
        horn_range = range(0 if outer_horns else 1, dim + 1 if outer_horns else dim)
        filler_i = None # To ensure it's in scope for the final check
        for i in horn_range:
            if verbose: print(f"Testing horn {i}...")
            horn_i = self.horn(i)
            filler_i = self.filler(horn_i, i)
            horn_i_prime = filler_i.horn(i)
            for j in range(len(horn_i)):
                if j == i: continue
                original = horn_i[j]
                reproduced = horn_i_prime[j]
                if np.any([(simplify(o-r) != 0) for o,r in zip(original.tensor.flatten(), reproduced.tensor.flatten())]):
                    # This verbose block was missing
                    if verbose:
                        diff = original - reproduced
                        print(f"Disagreement at face {j}: {diff}")
                    raise SimplicialException(f"Original horn and filler horn disagree at face {j}.")
        
        if filler_i is None: return True
        
        diff = self - filler_i
        # This verbose block was also missing
        if np.any(diff.simplify().tensor != 0):
            if verbose:
                print("Multiple fillers exist. The original tensor and the filler differ at the following indices:")
                monomial_count = lambda expr: len(expr.operands()) if hasattr(expr, 'operands') and expr.operands() else (1 if expr != 0 else 0)
                indices_with_correction_terms = 0
                for idx in np.ndindex(self.shape):
                    orig = self.tensor[idx]
                    fill = filler_i.tensor[idx]
                    d = simplify(orig - fill)
                    if d != 0:
                        count = monomial_count(fill)
                        print(f"  At index {idx}:")
                        print(f"    Original: {orig}")
                        print(f"    Filler:   {fill}")
                        print(f"    Monomial count: {count}")
                        indices_with_correction_terms += 1
                print(f"    Indices with correction terms: {indices_with_correction_terms}")
            return False
        if verbose: print("Unique filler.")
        return True

    def simplify(self):
        for idx in np.ndindex(self.shape): self.tensor[idx] = simplify(self.tensor[idx])
        return self
    
    def subs(self, substitutions: dict):
        for idx in np.ndindex(self.shape): self.tensor[idx] = self.tensor[idx].subs(substitutions)
        return self

    def to_latex(self):
        if len(self.shape) != 2: return "LaTeX representation only available for 2D tensors."
        rows, cols = self.shape
        latex_str = "\\begin{bmatrix}\n"
        for i in range(rows):
            latex_str += " & ".join([latex(self.tensor[i, j]) for j in range(cols)])
            if i < rows - 1: latex_str += " \\\\\n"
        latex_str += "\n\\end{bmatrix}"
        return latex_str

    def decompose_degen(self) -> Tuple["SymbolicTensor", List[Tuple["SymbolicTensor", int]]]:
        operations = []
        def helper(tensor: "SymbolicTensor", ops: List) -> "SymbolicTensor":
            d = tensor.dimen()
            for i in range(d + 1):
                try:
                    face_i = tensor.face(i)
                    degen_i = face_i.degen(i)
                    if degen_i.shape == tensor.shape and np.all([(simplify(a - b) == 0) for a, b in zip(tensor.tensor.flatten(), degen_i.tensor.flatten())]):
                        ops.append((face_i, i))
                        return helper(face_i, ops)
                except IndexError: continue
            return tensor
        base = helper(self, operations)
        return base, operations

# --- Standalone helper functions ---

def correction_rank(original: SymbolicTensor, filler: SymbolicTensor) -> int:
    if original.shape != filler.shape:
        raise ValueError("Tensors must have the same shape to compare.")
    differences = set()
    for idx in np.ndindex(original.shape):
        diff = simplify(original.tensor[idx] - filler.tensor[idx])
        if diff != 0:
            differences.add(str(diff))
    return len(differences)

def n_hypergroupoid_conjecture(shape: Tuple[int, ...], verbose=False) -> bool:
    if not shape or any(s == 0 for s in shape): return True 
    k = len(shape)
    N = min(shape) - 1
    if verbose:
        print(f"Conjecture check for shape {shape}: k={k}, N={N}. Prediction (unique?): {k < N}")
    return k < N

def test_symbolic_n_hypergroupoid(shape: Tuple[int,...], verbose=True):
    sym_tensor = SymbolicTensor(shape)
    conjecture = n_hypergroupoid_conjecture(shape, verbose=verbose)
    try:
        comparison = sym_tensor.n_hypergroupoid_comparison(outer_horns=True, verbose=verbose)
        if verbose:
            # This print block is now redundant because n_hypergroupoid_comparison is fully verbose
            if conjecture == comparison: print("✔️  The n-hypergroupoid conjecture is confirmed for this shape.")
            else: print("❌  Observation does not match conjecture prediction.")
        return conjecture, comparison, sym_tensor
    except SimplicialException as e:
        if "Degenerate boundary" in str(e):
            if verbose: print("Skipping comparison due to degenerate boundary.")
            return conjecture, None, sym_tensor
        raise

def check_symbolic_corrections(t: SymbolicTensor, t_prime: SymbolicTensor, horn_faces: list, k: int) -> bool:
    n = t.dimen()
    print(f"Checking horn({n},{k}) indices missing from symbolic tensor with shape {t.shape}.")
    all_symbols = set(str(s) for s in t.tensor.flatten() if s != 0)
    face_symbol_union = set()
    for face_idx, face in enumerate(horn_faces):
        if face_idx == k: continue
        for expr in face.tensor.flatten():
            if simplify(expr) != 0: face_symbol_union.add(str(expr))
    missing_symbols = all_symbols - face_symbol_union
    changed_symbols = set()
    diff = t - t_prime
    for idx in np.ndindex(t.shape):
        if simplify(diff.tensor[idx]) != 0:
            original_symbol = t.tensor[idx]
            if original_symbol != 0: changed_symbols.add(str(original_symbol))
            else: changed_symbols.add(str(t_prime.tensor[idx]))
    if changed_symbols == missing_symbols:
        print(f"Success: The filler differed from the original at {len(missing_symbols)} indices, matching the set of missing symbols.")
        return True
    else:
        print("Mismatch in correction terms vs. missing symbols.")
        extra = changed_symbols - missing_symbols
        missed = missing_symbols - changed_symbols
        if extra: print("Symbols changed that were not missing:", extra)
        if missed: print("Symbols missing but unchanged:", missed)
        return False

def main():
    # This block is intended to be run in a SageMath environment.
    print("--- Running Full Test and Validation Suite from Original Files ---")
    try:
        _ = var # Check if we are in a Sage environment
    except NameError:
        print("\nERROR: This script must be run in a SageMath environment.")
    else:
        # Test from original sagemath_compatible_tensor_ops.py
        print("\n--- Test from original sagemath_compatible_tensor_ops.py ---")
        shape = (3, 3)
        conjecture, comparison, sym_tensor = test_symbolic_n_hypergroupoid(shape)
        horn_1 = sym_tensor.horn(1)
        filler_1 = sym_tensor.filler(horn_1, 1)
        print("\nComparison of original and filler tensors for shape (3,3):")
        result = check_symbolic_corrections(sym_tensor, filler_1, horn_1, 1)
        print(f"Check result: {result}")
    
        # Test loop from original symbolic_tensor_ops.py
        print("\n--- Test loop from original symbolic_tensor_ops.py ---")
        def build_shape(n: int) -> Tuple[int,...]:
            return tuple(n+1 for _ in range(n))
        for k_order in range(3, 6):
            for j_horn in range(k_order + 1):
                shape = build_shape(k_order)
                print(f"\nBuilding Horn({k_order},{j_horn}) of generic tensor of shape: {shape}")
                try:
                    sym_tensor = SymbolicTensor(shape=shape)
                    horn = sym_tensor.horn(j_horn)
                    filler = sym_tensor.filler(horn, j_horn)
                    result = check_symbolic_corrections(sym_tensor, filler, horn, j_horn)
                    print(f"Result for shape {shape}, horn {j_horn}: {result}")
                except Exception as e:
                    print(f"An error occurred for shape {shape}, horn {j_horn}: {e}")
    
        # Additional test cases from original files
        print("\n--- Additional test cases ---")
        shape = (4, 5, 6)
        test_symbolic_n_hypergroupoid(shape, verbose=True)
    
        for d in range(2, 7):
            shape = build_shape(d)
            test_symbolic_n_hypergroupoid(shape, verbose=True)
    
