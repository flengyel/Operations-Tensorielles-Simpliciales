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
from tensor_ops import (n_hypergroupoid_conjecture, 
                       SimplicialException)  # Import the exception



class SymbolicTensor:

    def __init__(self, shape: Tuple[int, ...], tensor=None, init_type: str = 'range'):
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

    # Tensor addition 
    def __add__(self, other: "SymbolicTensor") -> "SymbolicTensor":
        if not isinstance(other, SymbolicTensor):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError(f"Cannot add tensors of different shapes {self.shape} vs {other.shape}")
        
        sum_tensor = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            sum_tensor[idx] = self.tensor[idx] + other.tensor[idx]
        return SymbolicTensor(self.shape, tensor=sum_tensor)
    
    def __sub__(self, other: "SymbolicTensor") -> "SymbolicTensor":
        if not isinstance(other, SymbolicTensor):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError(f"Cannot subtract tensors of different shapes {self.shape} vs {other.shape}")
        # elementwise
        diff = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            diff[idx] = self.tensor[idx] - other.tensor[idx]
        return SymbolicTensor(self.shape, tensor=diff)
    
    @staticmethod
    def from_tensor(tensor):
        """Create a SymbolicTensor from an existing tensor"""
        shape = tensor.shape
        return SymbolicTensor(shape, tensor=tensor)

    def dimen(self) -> int:
        return min(self.shape) - 1

    def order(self) -> int:
        """
        Calculates the order of this tensor, defined as the number of its dimensions.
        """
        return len(self.shape)

    def _dims(self) -> Tuple[np.ndarray, ...]:
        """
        Get dimensions of the tensor as a tuple of arrays of indices.
        This matches tensor_ops._dims()
        """
        return tuple([np.arange(start=0, stop=dim_size) for dim_size in self.shape])
    
    def face(self, i: int) -> "SymbolicTensor":
        """
        Apply the i-th simplicial face operation to the symbolic tensor.
        Matches tensor_ops.face() by removing index i from each axis.
        """
        d = self.dimen() + 1  # simplicial dimension + 1
        if not (0 <= i < d):
            raise IndexError(f"Face index {i} out of bounds for simplicial dimension {d}")

        axes = self._dims()
        indices = [np.delete(axes[dim], i) for dim in range(len(self.shape))]
        grid = np.ix_(*indices)
        result = self.tensor[grid]
        return SymbolicTensor(result.shape, tensor=result)

    def degen(self, k: int) -> "SymbolicTensor":
        """
        Apply k-th simplicial degeneracy by duplicating index k in every axis.
        Matches tensor_ops.degen().
        """
        result = self.tensor
        for axis in range(result.ndim):
            # Each element of slices can be either a slice or an int, 
            # which NumPy indexing with tuple(slices) accepts
            slices: list[Union[int, slice]] = [slice(None)] * result.ndim
            slices[axis] = k  # duplicate the k-th index 
            insert_slice = result[tuple(slices)]
            result = np.insert(result, k, insert_slice, axis=axis)
        return SymbolicTensor(result.shape, tensor=result)

    def bdry(self) -> "SymbolicTensor":
        """
        Compute the simplicial boundary of the symbolic tensor.
        Matches tensor_ops.bdry(), using alternating sum of faces.
        """
        d = self.dimen() + 1  # simplicial dimension + 1
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
    
    def horn(self, k: int) -> List["SymbolicTensor"]:
        """
        Construct the k-th horn of the symbolic tensor.
        Matches tensor_ops.horn().
        """
        d = self.dimen() + 1  # simplicial dimension + 1
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
    
    def filler(self, horn_list:List["SymbolicTensor"], k: int) -> "SymbolicTensor":
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
    
    def n_hypergroupoid_comparison(self, outer_horns=False, verbose=False, allow_degen=False) -> bool:
        """
        Test the n-hypergroupoid conjecture: uniqueness of fillers.
        Matches tensor_ops.n_hypergroupoid_comparison().
        """
        boundary = self.bdry()
        if not allow_degen and boundary.is_degen():
            if verbose:
                print("Boundary is degenerate.")
            raise SimplicialException("Degenerate boundary.")

        dim = self.dimen()  # Updated to use self.dimen() instead of dimen(self.tensor)
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
    def simplify(self) -> "SymbolicTensor":
        """Simplify all expressions in the tensor"""
        for idx in np.ndindex(self.shape):
            self.tensor[idx] = sp.simplify(self.tensor[idx])
        return self
    
    def subs(self, substitutions: dict) -> "SymbolicTensor":
        """Apply substitutions to all expressions"""
        for idx in np.ndindex(self.shape):
            val = self.tensor[idx]
            subs_method = getattr(val, "subs", None)
            if callable(subs_method):
                # Only call subs if the element provides such a method (e.g., a SymPy expression)
                self.tensor[idx] = subs_method(substitutions)
            else:
                # Leave non-symbolic entries unchanged
                self.tensor[idx] = val
        return self
    
    def to_latex(self) -> str:
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
    
    def __str__(self) -> str:
        """String representation of the tensor"""
        return str(self.tensor)
    
    def __repr__(self) -> str:
        """Representative string of the tensor"""
        return f"SymbolicTensor(shape={self.shape})"

# Replace the decompose_degen method in the SymbolicTensor class with this version

    def decompose_degen(self) -> Union[List["SymbolicTensor"], None]:
        """
        Decomposes a symbolic tensor into a sum of degenerate components by
        solving a system of symbolic linear equations.

        A tensor T of dimension n is degenerate if it can be written as
        T = sum_{i=0}^{n-1} s_i(B_i), where B_i are symbolic tensors of
        dimension n-1.

        Returns:
            A list of the component tensors [B_0, B_1, ...], or None if no
            exact decomposition exists (i.e., the tensor is non-degenerate).
        """
        n = self.dimen()
        if n <= 0: # 0-simplices and lower cannot be degenerate
            return None

        shape_n_minus_1 = tuple(s - 1 for s in self.shape)
        if any(s < 1 for s in shape_n_minus_1): # Shape must be at least (1,1,...)
            return None

        # 1. Create the unknown B_i tensors with unique symbols
        unknown_b_tensors = []
        all_unknown_symbols = []
        # The loop must go from 0 to n-1. This is the fix.
        for i in range(n):
            prefix = f"b_{i}"
            b_i = SymbolicTensor(shape_n_minus_1)
            b_i_symbols = []
            for idx in np.ndindex(b_i.shape):
                idx_str = ','.join(map(str, idx))
                symbol = sp.Symbol(f'{prefix}_{{{idx_str}}}')
                b_i.tensor[idx] = symbol
                b_i_symbols.append(symbol)
            
            unknown_b_tensors.append(b_i)
            all_unknown_symbols.extend(b_i_symbols)

        # 2. Form the symbolic sum of the degeneracies: Sum_{i=0}^{n-1}(s_i(B_i))
        sum_of_degen = SymbolicTensor(self.shape, init_type='zeros')
        # The loop must go from 0 to n-1. This is the fix.
        for i in range(n):
            degen_bi = unknown_b_tensors[i].degen(i)
            sum_of_degen = sum_of_degen + degen_bi
        
        # 3. Create and solve the system of equations
        equations = [sp.Eq(self.tensor[idx], sum_of_degen.tensor[idx]) for idx in np.ndindex(self.shape)]
        solution = sp.solve(equations, all_unknown_symbols, dict=True)

        if not solution:
            return None

        # 5. Substitute the solution back to get the concrete B_i tensors
        subs_dict = solution[0]
        result_b_tensors = [b.subs(subs_dict) for b in unknown_b_tensors]
            
        return result_b_tensors
    
    def is_degen(self) -> bool:
        """
        Determine if the symbolic tensor is degenerate by attempting to find
        an exact additive decomposition.
        """
        return self.decompose_degen() is not None

def tensor_filler_difference_rank(original: "SymbolicTensor", filler: "SymbolicTensor") -> int:
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


def test_symbolic_n_hypergroupoid(shape: Tuple[int, ...], verbose=True):
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

def is_generator_symbolic(T: SymbolicTensor) -> bool:
    """
    Symbolic degeneracy test:
    - Let n = min(T.shape).
    - Zero tensor is degenerate.
    - If n <= 1 (0-simplices), non-degenerate.
    - If n == 2, degenerate if constant.
    - If n > 2, degenerate if T.tensor == (T.face(i).degen(i)).tensor for some i.
    """
    #print(f"[DEBUG] is_generator_symbolic: tensor shape={T.shape}")
    entries = list(T.tensor.flatten())
    if all(e == 0 for e in entries):
        #print("[DEBUG] is_generator_symbolic: zero tensors cannot be generators")
        return False
    n = min(T.shape)
    if n <= 1:
        #print("[DEBUG] is_generator_symbolic: n<=1, non-degenerate")
        return True
    if n == 2:
        constant = all(entry == entries[0] for entry in entries)
        #print(f"[DEBUG] is_generator_symbolic: n==2, constant? {constant}")
        return not constant
    for i in range(n):
        try:
            D = T.face(i).degen(i)
        except IndexError:
            continue
        if D.tensor.tolist() == T.tensor.tolist():
            #print(f"[DEBUG] is_generator_symbolic: degeneracy via face+degen at i={i}")
            return False
    #print("[DEBUG] is_generator_symbolic: nonzero and non-degenerate. Generator")
    return True


def check_missing_symbols(t, t_prime, horn_faces, k):
    """
    For symbolic tensors t and t_prime (both SymbolicTensor or at least
    numpy arrays of sympy expressions), ensures t_prime differs from t exactly
    at the 'missing' symbols. Missing symbols are those that do not appear
    in any non-missing face of horn_faces.

    t: original SymbolicTensor
    t_prime: filler result (also SymbolicTensor)
    horn_faces: result of t.horn(k)
    k: the face index that is replaced by zeros in horn_faces

    Returns True if everything matches, else False.
    """
    shape = t.shape
    n = t.dimen()  # simplicial dimension of t

    print(f"Checking horn({n},{k}) indices missing from symbolic tensor with shape {shape}.")
    # Gather set of all symbols in T (by name).
    # For example, each T.tensor[idx] is x_{0,1} etc. If T might have zeros, skip them.
    all_symbols = set()
    for idx in np.ndindex(shape):
        expr = t.tensor[idx]
        if expr != sp.S.Zero:
            all_symbols.add(str(expr))

    # Gather union of symbol names in the non-missing faces
    face_symbol_union = set()
    for face_idx, face in enumerate(horn_faces):
        if face_idx == k:  # skip missing (zero) face
            continue
        fshape = face.shape
        for subidx in np.ndindex(fshape):
            expr = face.tensor[subidx]
            # If it isn't literally zero, gather its name
            if sp.simplify(expr) != sp.S.Zero:
                face_symbol_union.add(str(expr))

    # Missing = all symbols not in face_symbol_union
    missing_symbols = all_symbols - face_symbol_union

    # Now we see which entries Tprime changed vs. T.
    # We'll gather the symbolic name of T[idx] or Tprime[idx].
    changed_symbols = set()
    for idx in np.ndindex(shape):
        expr_orig = t.tensor[idx]
        expr_new = t_prime.tensor[idx]
        diff_expr = sp.simplify(expr_new - expr_orig)
        if diff_expr != sp.S.Zero:
            # We say that t'[idx] differs from t[idx]. So we record the name of the original symbol.
            # If the original was zero, there's no symbol to record, so we might record str(expr_new).
            if expr_orig == sp.S.Zero:
                changed_symbols.add(str(expr_new))
            else:
                changed_symbols.add(str(expr_orig))

    if changed_symbols == missing_symbols:
        print(f"Success: the filler differed from the original at {len(missing_symbols)} indices.")
        return True
    else:
        print("Mismatch in correction terms vs. missing symbols.")
        extra = changed_symbols - missing_symbols
        missed = missing_symbols - changed_symbols
        if extra:
            print("Symbols changed that were not missing:", extra)
        if missed:
            print("Symbols missing but unchanged:", missed)
        return False


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
    print(sym_tensor)
    print("\nFiller tensor:")
    print(filler_1)

    print("\nComparison of original and filler tensors:")
    result = check_missing_symbols(sym_tensor, filler_1, horn_1, 1)
    print("Check result:", result)


    def build_shape(n: int) -> Tuple[int, ...]:
        return tuple(n+1 for _ in range(n))

    for k in range(3, 6):
        for j in range(1,k):
            shape = build_shape(k)
            print(f"Building inner Horn({k},{j}) of generic tensor of shape: {shape}")
            sym_tensor = SymbolicTensor(shape=shape)
            inner_horn = sym_tensor.horn(j)
            filler = sym_tensor.filler(inner_horn, j)
            result = check_missing_symbols(sym_tensor, filler, inner_horn, j)
            print(f"Result for shape {shape}: {result}")
            

#    for d in range(2, 7):
#        print(f"build_shape({d}): {build_shape(d)}")


#    for d in range(2, 7):
#        shape = build_shape(d)
#        conjecture, comparison, sym_tensor = test_symbolic_n_hypergroupoid(shape)

    shape = (3,3,3)
    conjecture, comparison, sym_tensor = test_symbolic_n_hypergroupoid(shape)
    print(f"Shape: {shape}, Conjecture: {conjecture}, Comparison: {comparison}")