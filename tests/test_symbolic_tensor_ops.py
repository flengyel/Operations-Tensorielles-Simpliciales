#    Opérations Tensorielles Simpliciales
#    Simplicial Operations on Matrices and Hypermatrices
#    test_symbolic_tensor_ops.py
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


import pytest
import sympy as sp
import numpy as np
from src.symbolic_tensor_ops import SymbolicTensor, correction_rank
from src.tensor_ops import n_hypergroupoid_conjecture, SimplicialException, ___SEED___
import random

random.seed(___SEED___)

def test_multiindex_variable_names():
    shape = (2, 3)
    tensor = SymbolicTensor(shape)
    expected = np.array([
        [sp.Symbol("x_{0,0}"), sp.Symbol("x_{0,1}"), sp.Symbol("x_{0,2}")],
        [sp.Symbol("x_{1,0}"), sp.Symbol("x_{1,1}"), sp.Symbol("x_{1,2}")]
    ])
    assert tensor.shape == expected.shape
    for idx in np.ndindex(shape):
        assert tensor.tensor[idx] == expected[idx]

def test_face_degen_preserve_structure():
    shape = (3, 3)
    T = SymbolicTensor(shape)
    face = T.face(1)
    assert face.shape == (2, 2)
    degen = face.degen(1)
    assert degen.shape == shape

def test_boundary_simple():
    T = SymbolicTensor((3, 3))
    bdry = T.bdry()
    assert bdry.shape == (2, 2)
    assert isinstance(bdry.tensor[0, 0], sp.Basic)

def test_filler_agrees_with_horn():
    shape = (3, 3)
    T = SymbolicTensor(shape)
    horn_list = T.horn(1)
    filler = T.filler(horn_list, 1)
    horn_prime = filler.horn(1)
    for j, face_j in enumerate(horn_list):
        if j == 1:
            continue
        for idx in np.ndindex(face_j.shape):
            diff = sp.simplify(face_j.tensor[idx] - horn_prime[j].tensor[idx])
            assert diff == 0

def test_correction_rank():
    T = SymbolicTensor((3, 3))
    filler = T.filler(T.horn(1), 1)
    rank = correction_rank(T, filler)
    assert isinstance(rank, int)
    assert rank >= 0

def test_n_hypergroupoid_comparison():
    shape = (3, 3)
    T = SymbolicTensor(shape)
    conjecture = n_hypergroupoid_conjecture(shape)
    try:
        result = T.n_hypergroupoid_comparison()
    except SimplicialException:
        result = None
    assert conjecture == result or result is None

def test_bdry_squared_zero():
    for _ in range(5):
        dims = random.randint(2, 3)
        shape = tuple(random.randint(3, 5) for _ in range(dims))
        T = SymbolicTensor(shape)
        bdry1 = T.bdry()
        bdry2 = bdry1.bdry()
        assert all(sp.simplify(bdry2.tensor[idx]) == 0 for idx in np.ndindex(bdry2.shape))

def test_first_identity():
    def first_identity(tensor: SymbolicTensor) -> bool:
        d = min(tensor.shape)
        for j in range(d):
            for i in range(j):
                X = tensor.face(j).face(i)
                Y = tensor.face(i).face(j - 1)
                for idx in np.ndindex(X.shape):
                    if sp.simplify(X.tensor[idx] - Y.tensor[idx]) != 0:
                        return False
        return True
    for _ in range(5):
        dims = random.randint(2, 3)
        shape = tuple(random.randint(3, 5) for _ in range(dims))
        assert first_identity(SymbolicTensor(shape))

def test_second_identity():
    def second_identity(tensor: SymbolicTensor) -> bool:
            d = min(tensor.shape)
            for j in range(d):
                for i in range(j):
                    X = tensor.degen(j).face(i)
                    Y = tensor.face(i).degen(j - 1)
                    for idx in np.ndindex(X.shape):
                        if sp.simplify(X.tensor[idx] - Y.tensor[idx]) != 0:
                            return False
            return True
    for _ in range(5):
        dims = random.randint(2, 3)
        shape = tuple(random.randint(3, 5) for _ in range(dims))
        assert second_identity(SymbolicTensor(shape))

def test_third_identity():
    def third_identity(tensor: SymbolicTensor) -> bool:
        d = min(tensor.shape)
        for j in range(d):
            X = tensor.degen(j).face(j)
            Y = tensor.degen(j).face(j + 1)
            for idx in np.ndindex(X.shape):
                if not (sp.simplify(X.tensor[idx] - tensor.tensor[idx]) == 0 and sp.simplify(Y.tensor[idx] - tensor.tensor[idx]) == 0):
                    return False
        return True
    for _ in range(5):
        dims = random.randint(2, 3)
        shape = tuple(random.randint(3, 5) for _ in range(dims))
        assert third_identity(SymbolicTensor(shape))

def test_fourth_identity():
    def fourth_identity(tensor: SymbolicTensor) -> bool:
        d = min(tensor.shape)
        return all(check_faces(tensor, i) for i in range(d + 1))

    def check_faces(tensor: SymbolicTensor, i: int) -> bool:
        return all(j + 1 >= i or compare_faces(tensor, i, j) for j in range(i + 1))

    def compare_faces(tensor: SymbolicTensor, i: int, j: int) -> bool:
        X = tensor.degen(j).face(i)
        Y = tensor.face(i - 1).degen(j)
        return all(sp.simplify(X.tensor[idx] - Y.tensor[idx]) == 0 for idx in np.ndindex(X.shape))

    for _ in range(5):
        dims = random.randint(2, 3)
        shape = tuple(random.randint(3, 5) for _ in range(dims))
        assert fourth_identity(SymbolicTensor(shape))

def test_fifth_identity():
    def fifth_identity(tensor: SymbolicTensor) -> bool:
        d = min(tensor.shape)
        for j in range(d):
            if not check_degens(tensor, j):
                return False
        return True

    def check_degens(tensor: SymbolicTensor, j: int) -> bool:
        for i in range(j + 1):
            if not compare_degens(tensor, j, i):
                return False
        return True

    def compare_degens(tensor: SymbolicTensor, j: int, i: int) -> bool:
        try:
            X = tensor.degen(j).degen(i)
            Y = tensor.degen(i).degen(j + 1)
        except IndexError:
            return False
        return all(sp.simplify(X.tensor[idx] - Y.tensor[idx]) == 0 for idx in np.ndindex(X.shape))

    for _ in range(5):
        dims = random.randint(2, 3)
        shape = tuple(random.randint(4, 6) for _ in range(dims))
        assert fifth_identity(SymbolicTensor(shape))

if __name__ == "__main__":
    pytest.main([__file__])
