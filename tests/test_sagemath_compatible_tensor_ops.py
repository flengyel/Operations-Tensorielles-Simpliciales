#    Opérations Tensorielles Simpliciales
#    Simplicial Operations on Matrices and Hypermatrices
#    test_sagemath_compatible_tensor_ops.py
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

# This is a pytest suite adapted to test the SageMath-compatible version of SymbolicTensor.
# It should be run within a SageMath environment.

import pytest
import random

np = pytest.importorskip("numpy")
sage_all = pytest.importorskip("sage.all")
sage_symbolic = pytest.importorskip("sage.symbolic.expression")
var = sage_all.var
simplify = sage_all.simplify
Expression = sage_symbolic.Expression

# Import the class and helpers from the Sage-compatible file
# Ensure sagemath_compatible_tensor_ops.py is in the same directory or Python path.
from simplicial_tensors.sagemath_compatible_tensor_ops import SymbolicTensor, SimplicialException, n_hypergroupoid_conjecture

# Seed for reproducibility
___SEED___ = 42
random.seed(___SEED___)

def test_multiindex_variable_names():
    """Tests if variables are created with the correct Sage-compatible names."""
    shape = (2, 3)
    tensor = SymbolicTensor(shape)
    expected = np.array([
        [var("x_0_0"), var("x_0_1"), var("x_0_2")],
        [var("x_1_0"), var("x_1_1"), var("x_1_2")]
    ], dtype=object)
    assert tensor.shape == expected.shape
    for idx in np.ndindex(shape):
        assert tensor.tensor[idx] == expected[idx]

def test_face_degen_preserve_structure():
    """Tests if face and degen operations produce tensors of the correct shape."""
    shape = (3, 3)
    T = SymbolicTensor(shape)
    face = T.face(1)
    assert face.shape == (2, 2)
    degen = face.degen(1)
    assert degen.shape == shape

def test_boundary_simple():
    """Tests the boundary of a simple tensor."""
    T = SymbolicTensor((3, 3))
    bdry = T.bdry()
    assert bdry.shape == (2, 2)
    # Check if the result is a Sage symbolic expression
    assert isinstance(bdry.tensor[0, 0], Expression)

def test_filler_agrees_with_horn():
    """Tests if the filler correctly reconstructs the faces of the horn."""
    shape = (3, 3)
    T = SymbolicTensor(shape)
    horn_list = T.horn(1)
    filler = T.filler(horn_list, 1)
    
    # After filling, the faces of the filler should match the original horn faces
    for j, original_face in enumerate(horn_list):
        if j == 1: continue # Skip the missing face
        
        filler_face = filler.face(j)
        diff = original_face - filler_face
        
        # All entries in the difference tensor should simplify to zero
        assert np.all([(simplify(d) == 0) for d in diff.tensor.flatten()])

def test_bdry_squared_zero():
    """Tests the fundamental simplicial identity d^2 = 0."""
    for _ in range(5):
        dims = random.randint(2, 3)
        shape = tuple(random.randint(3, 5) for _ in range(dims))
        T = SymbolicTensor(shape)
        bdry1 = T.bdry()
        bdry2 = bdry1.bdry()
        # All entries in the second boundary should be zero
        assert all(simplify(entry) == 0 for entry in bdry2.tensor.flatten())

def test_simplicial_identities():
    """
    Groups all five simplicial identities into one parameterized test.
    This structure makes it easy to see which identity might fail.
    """
    
    # d_i d_j = d_{j-1} d_i for i < j
    def first_identity(T):
        n = T.dimen()
        for j in range(n + 1):
            for i in range(j):
                lhs = T.face(j).face(i)
                rhs = T.face(i).face(j - 1)
                assert np.all([(simplify(l-r)==0) for l,r in zip(lhs.tensor.flatten(), rhs.tensor.flatten())])
    
    # s_i s_j = s_{j+1} s_i for i <= j
    def fifth_identity(T): # Renamed from original test file for logical order
        n = T.dimen()
        for j in range(n + 1):
            for i in range(j + 1):
                lhs = T.degen(j).degen(i)
                rhs = T.degen(i).degen(j + 1)
                assert np.all([(simplify(l-r)==0) for l,r in zip(lhs.tensor.flatten(), rhs.tensor.flatten())])

    # d_i s_j relations
    def mixed_identities(T):
        n = T.dimen()
        for j in range(n + 1): # s_j
            sjT = T.degen(j)
            for i in range(n + 2): # d_i
                lhs = sjT.face(i)
                if i < j:
                    rhs = T.face(j - 1).degen(i)
                elif i == j or i == j + 1:
                    rhs = T # Identity
                else: # i > j + 1
                    rhs = T.face(j).degen(i - 1)
                
                assert np.all([(simplify(l-r)==0) for l,r in zip(lhs.tensor.flatten(), rhs.tensor.flatten())])

    for _ in range(3): # Run a few random trials
        dims = random.randint(2, 3)
        shape = tuple(random.randint(4, 5) for _ in range(dims))
        T = SymbolicTensor(shape)
        
        first_identity(T)
        fifth_identity(T)
        mixed_identities(T)

def test_n_hypergroupoid_conjecture_uniqueness():
    """Tests the conjecture's prediction against filler observation."""
    
    # Case 1: k >= N, expect non-unique
    shape_non_unique = (3, 3) # k=2, N=2
    T_non_unique = SymbolicTensor(shape_non_unique)
    conjecture_pred_1 = n_hypergroupoid_conjecture(shape_non_unique)
    diff = T_non_unique - T_non_unique.filler(T_non_unique.horn(1), 1)
    observed_unique_1 = not np.any(diff.simplify().tensor != 0)
    assert conjecture_pred_1 == False
    assert observed_unique_1 == False

    # Case 2: k < N, expect unique
    shape_unique = (5, 5) # k=2, N=4
    T_unique = SymbolicTensor(shape_unique)
    conjecture_pred_2 = n_hypergroupoid_conjecture(shape_unique)
    diff = T_unique - T_unique.filler(T_unique.horn(1), 1)
    observed_unique_2 = not np.any(diff.simplify().tensor != 0)
    assert conjecture_pred_2 == True
    assert observed_unique_2 == True


if __name__ == "__main__":
    # To run these tests, execute from a SageMath command line:
    # sage -python -m pytest test_sagemath_compatible_tensor_ops.py
    pytest.main([__file__])
