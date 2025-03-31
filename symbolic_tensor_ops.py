#    Opérations Tensorielles Simpliciales
#    Simplicial Operations on Matrices and Hypermatrices
#    tensor_ops.py
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


import sympy as sp
import numpy as np
from typing import Tuple, List, Union, Any
from tensor_ops import (order, dimen, n_hypergroupoid_conjecture, 
                       SimplicialException)  # Import the exception

class SymbolicTensor:

    def __init__(self, shape: Tuple[int], tensor=None, init_type: str = 'range'):
        """Initialize a symbolic tensor."""
        self.shape = shape

        if tensor is not None:
            self.tensor = tensor
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

    
    @staticmethod
    def from_tensor(tensor):
        """Create a SymbolicTensor from an existing tensor"""
        shape = tensor.shape
        return SymbolicTensor(shape, tensor=tensor)

    def _dims(self):
        """
        Get dimensions of the tensor as a tuple of arrays of indices.
        This matches tensor_ops._dims()
        """
        return tuple([np.arange(dim_size) for dim_size in self.shape])
    
    def face(self, i: int):
        """
        Apply the i-th simplicial face operation to the symbolic tensor.
        Matches tensor_ops.face() by removing index i from each axis.
        """
        d = min(self.shape)
        if not (0 <= i < d):
            raise IndexError(f"Face index {i} out of bounds for simplicial dimension {d}")

        axes = self._dims()
        indices = [np.delete(axes[dim], i) for dim in range(len(self.shape))]
        grid = np.ix_(*indices)
        result = self.tensor[grid]
        return SymbolicTensor(result.shape, tensor=result)

    def degen(self, k: int):
        """
        Apply k-th simplicial degeneracy by duplicating index k in every axis.
        Matches tensor_ops.degen().
        """
        result = self.tensor
        for axis in range(result.ndim):
            slices = [slice(None)] * result.ndim
            slices[axis] = k
            insert_slice = result[tuple(slices)]
            result = np.insert(result, k, insert_slice, axis=axis)
        return SymbolicTensor(result.shape, tensor=result)

    def bdry(self):
        """
        Compute the simplicial boundary of the symbolic tensor.
        Matches tensor_ops.bdry(), using alternating sum of faces.
        """
        d = min(self.shape)
        result_shape = tuple(dim - 1 for dim in self.shape)
        result = np.zeros(result_shape, dtype=object)

        for idx in np.ndindex(result_shape):
            result[idx] = sp.S.Zero

        for i in range(d):
            face_i = self.face(i)
            for idx in np.ndindex(result_shape):
                if i % 2 == 0:
                    result[idx] += face_i.tensor[idx]
                else:
                    result[idx] -= face_i.tensor[idx]

        return SymbolicTensor(result_shape, tensor=result)
    
    def horn(self, k: int):
        """
        Construct the k-th horn of the symbolic tensor.
        Matches tensor_ops.horn().
        """
        d = dimen(self.tensor) + 1  # simplicial dimension + 1
        if not (0 <= k < d):
            raise ValueError(f"Horn index {k} must be in [0, {d-1}]")

        faces = []
        zero_shape = tuple(dim - 1 for dim in self.shape)

        for i in range(d):
            if i == k:
                # k-th face is zero tensor
                zero_tensor = np.empty(zero_shape, dtype=object)
                for idx in np.ndindex(zero_shape):
                    zero_tensor[idx] = sp.S.Zero
                faces.append(SymbolicTensor(zero_shape, tensor=zero_tensor))
            else:
                faces.append(self.face(i))

        return faces
    
    def filler(self, horn_list, k: int):
        """
        Compute a filler for the horn using Moore's algorithm.
        Matches tensor_ops.filler().
        """
        g = horn_list[k].degen(0)  # Start from degenerate zero face

        for r in range(k):
            face_gr = g.face(r)
            diff_tensor = np.zeros(face_gr.shape, dtype=object)
            for idx in np.ndindex(face_gr.shape):
                diff_tensor[idx] = face_gr.tensor[idx] - horn_list[r].tensor[idx]
            degen_diff = SymbolicTensor(face_gr.shape, tensor=diff_tensor).degen(r)

            for idx in np.ndindex(g.shape):
                g.tensor[idx] = g.tensor[idx] - degen_diff.tensor[idx]

        t = len(horn_list) - 1
        while t > k:
            face_gt = g.face(t)
            diff_tensor = np.zeros(face_gt.shape, dtype=object)
            for idx in np.ndindex(face_gt.shape):
                diff_tensor[idx] = horn_list[t].tensor[idx] - face_gt.tensor[idx]
            degen_diff = SymbolicTensor(face_gt.shape, tensor=diff_tensor).degen(t - 1)

            for idx in np.ndindex(g.shape):
                g.tensor[idx] = g.tensor[idx] + degen_diff.tensor[idx]

            t -= 1

        return g
    
    def is_degen(self):
        """
        Determine if the symbolic tensor is degenerate.
        A tensor is degenerate if it equals a degeneracy of one of its faces.
        Matches tensor_ops.is_degen().
        """
        d = dimen(self.tensor)  # simplicial dimension
        for i in range(d):
            face_i = self.face(i)
            degen_i = face_i.degen(i)
            for idx in np.ndindex(self.shape):
                if sp.simplify(self.tensor[idx] - degen_i.tensor[idx]) != 0:
                    break
            else:
                return True  # All entries matched ⇒ degenerate
        return False
    
    def n_hypergroupoid_comparison(self, outer_horns=False, verbose=False, allow_degen=False):
        """
        Test the n-hypergroupoid conjecture: uniqueness of fillers.
        Matches tensor_ops.n_hypergroupoid_comparison().
        """
        boundary = self.bdry()
        if not allow_degen and boundary.is_degen():
            if verbose:
                print("Boundary is degenerate.")
            raise SimplicialException("Degenerate boundary.")

        dim = dimen(self.tensor)
        horn_range = range(0 if outer_horns else 1, dim + 1 if outer_horns else dim)

        for i in horn_range:
            if verbose:
                print(f"Testing horn {i}...")

            horn_i = self.horn(i)
            filler_i = self.filler(horn_i, i)
            horn_i_prime = filler_i.horn(i)

            for j in range(len(horn_i)):
                if j == i:
                    continue
                original = horn_i[j]
                reproduced = horn_i_prime[j]
                for idx in np.ndindex(original.shape):
                    diff = sp.simplify(original.tensor[idx] - reproduced.tensor[idx])
                    if diff != 0:
                        if verbose:
                            print(f"Disagreement at face {j}, index {idx}: {diff}")
                        raise SimplicialException(f"Original horn and filler horn disagree at face {j}, position {idx}.")

        monomial_count = lambda expr: len(expr.as_ordered_terms()) if expr != 0 else 0
        differences = []
        indices_with_correction_terms = 0
        for idx in np.ndindex(self.shape):
            orig = self.tensor[idx]
            fill = filler_i.tensor[idx]
            diff = sp.simplify(orig - fill)
            if diff != 0:
                count = monomial_count(fill)
                differences.append((idx, orig, fill, count))
                indices_with_correction_terms += 1

        if differences:
            if verbose:
                print("Multiple fillers exist. The original tensor and the filler differ at the following indices:")
                for idx, orig, fill, count in differences:
                    print(f"  At index {idx}:")
                    print(f"    Original: {sp.pretty(orig)}")
                    print(f"    Filler:   {sp.pretty(fill)}")
                    print(f"    Monomial count: {count}")
                print(f"    Indices with correction terms: {indices_with_correction_terms}")
            return False

        if verbose:
            print("Unique filler.")
        return True

    ############### Simplification and Substitution Methods ###############    
    def simplify(self):
        """Simplify all expressions in the tensor"""
        for idx in np.ndindex(self.shape):
            self.tensor[idx] = sp.simplify(self.tensor[idx])
        return self
    
    def subs(self, substitutions: dict):
        """Apply substitutions to all expressions"""
        for idx in np.ndindex(self.shape):
            self.tensor[idx] = self.tensor[idx].subs(substitutions)
        return self
    
    def to_latex(self):
        """Convert to LaTeX representation"""
        if len(self.shape) == 2:
            rows, cols = self.shape
            latex = "\\begin{bmatrix}\n"
            for i in range(rows):
                for j in range(cols):
                    latex += sp.latex(self.tensor[i, j])
                    if j < cols - 1:
                        latex += " & "
                if i < rows - 1:
                    latex += " \\\\\n"
            latex += "\n\\end{bmatrix}"
            return latex
        else:
            # For higher dimensions, return a list of 2D slices
            slices = []
            if len(self.shape) == 3:
                for i in range(self.shape[0]):
                    slice_tensor = SymbolicTensor((self.shape[1], self.shape[2]), 
                                                 tensor=self.tensor[i, :, :])
                    slices.append(f"Slice {i}:\n{slice_tensor.to_latex()}")
            return "\n\n".join(slices)
    
    def __str__(self):
        """String representation of the tensor"""
        return str(self.tensor)
    
    def __repr__(self):
        """Representative string of the tensor"""
        return f"SymbolicTensor(shape={self.shape})"

    def decompose_degen(self) -> Tuple["SymbolicTensor", List[Tuple["SymbolicTensor", int]]]:
        """
        Decompose a degenerate symbolic tensor into a non-degenerate base
        and a sequence of degeneracy operations.
        Matches tensor_ops.decompose_degen().
        
        Returns:
            Tuple of (non-degenerate base tensor, list of (face, index) degeneracies)
        """
        operations = []

        def helper(tensor: "SymbolicTensor", ops: List[Tuple["SymbolicTensor", int]]) -> "SymbolicTensor":
            d = dimen(tensor.tensor)
            for i in range(d):
                face_i = tensor.face(i)
                degen_i = face_i.degen(i)

                # Check symbolic equality
                for idx in np.ndindex(tensor.shape):
                    if sp.simplify(tensor.tensor[idx] - degen_i.tensor[idx]) != 0:
                        break
                else:
                    # Found degeneracy at face(i)
                    ops.append((face_i, i))
                    return helper(face_i, ops)

            return tensor  # Base case: non-degenerate

        base = helper(self, operations)
        return base, operations

def correction_rank(original: "SymbolicTensor", filler: "SymbolicTensor") -> int:
    """
    Returns the number of distinct nonzero symbolic differences between original and filler tensors.
    This quantifies the number of independently modified entries.
    """
    if original.shape != filler.shape:
        raise ValueError("Tensors must have the same shape to compare.")

    differences = set()

    for idx in np.ndindex(original.shape):
        diff = sp.simplify(original.tensor[idx] - filler.tensor[idx])
        if diff != 0:
            differences.add(str(diff))  # Use string representation to test symbolic structure

    return len(differences)


def test_symbolic_n_hypergroupoid(shape: Tuple[int], verbose=True):
    """
    Test the n-hypergroupoid conjecture using symbolic tensors.

    Args:
        shape: Shape of the tensor to test
        verbose: Whether to print verbose output

    Returns:
        Tuple of (conjecture_result, comparison_result, symbolic_tensor)
    """
    # Create a symbolic range tensor
    sym_tensor = SymbolicTensor(shape)

    # Evaluate the conjecture condition: should uniqueness hold?
    conjecture = n_hypergroupoid_conjecture(shape, verbose=verbose)

    try:
        # Compare symbolic horn fillers
        comparison = sym_tensor.n_hypergroupoid_comparison(verbose=verbose)

        if verbose:
            print(f"Conjecture predicts unique fillers: {conjecture}")
            print(f"Filler uniqueness observed: {comparison}")
            if conjecture == comparison:
                print("✔️  The n-hypergroupoid conjecture is confirmed for this shape.")
            else:
                print("❌  Observation does not match conjecture prediction.")

        return conjecture, comparison, sym_tensor

    except SimplicialException as e:
        if "Degenerate boundary" in str(e):
            if verbose:
                print("Skipping comparison due to degenerate boundary.")
            return conjecture, None, sym_tensor
        raise


if __name__ == "__main__":
    # Example usage
    shape = (3, 3)  # A shape where fillers are not unique
    # Test the conjecture
    conjecture, comparison, sym_tensor = test_symbolic_n_hypergroupoid(shape)

    # Examine the specific case of a 1-horn
    horn_1 = sym_tensor.horn(1)
    filler_1 = sym_tensor.filler(horn_1, 1)

    # Compare the original tensor and its filler
    print("Original tensor:")
    print(sym_tensor.to_latex())
    print("\nFiller tensor:")
    print(filler_1.to_latex())

    shape = (4, 4, 4)
    conjecture, comparison, sym_tensor = test_symbolic_n_hypergroupoid(shape)    
    shape = (4, 5, 6)
    conjecture, comparison, sym_tensor = test_symbolic_n_hypergroupoid(shape)    
    shape = (5, 5, 5, 5)
    conjecture, comparison, sym_tensor = test_symbolic_n_hypergroupoid(shape)

    # Test with larger shapes
    shape = (6, 6, 6, 6, 6)
    conjecture, comparison, sym_tensor = test_symbolic_n_hypergroupoid(shape)

    shape = (7, 7, 7, 7, 7, 7)
    conjecture, comparison, sym_tensor = test_symbolic_n_hypergroupoid(shape)
