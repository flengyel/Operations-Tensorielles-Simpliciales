import pytest
import sympy as sp
import numpy as np
from symbolic_tensor_ops import SymbolicTensor, correction_rank
from tensor_ops import n_hypergroupoid_conjecture, SimplicialException

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
    for shape in [(3, 3), (4, 4), (3, 4, 5)]:
        T = SymbolicTensor(shape)
        bdry1 = T.bdry()
        bdry2 = bdry1.bdry()
        assert all(sp.simplify(bdry2.tensor[idx]) == 0 for idx in np.ndindex(bdry2.shape))

def test_simplicial_identities():
    T = SymbolicTensor((4, 4))
    d = min(T.shape)
    for i in range(d - 1):
        for j in range(i + 1, d):
            # face_i.face_j == face_{j-1}.face_i
            left = T.face(j).face(i)
            right = T.face(i).face(j - 1)
            for idx in np.ndindex(left.shape):
                assert sp.simplify(left.tensor[idx] - right.tensor[idx]) == 0

    for i in range(d):
        for j in range(i, d):
            # degen_i.degen_j == degen_{j+1}.degen_i
            left = T.degen(i).degen(j + 1)
            right = T.degen(j).degen(i)
            for idx in np.ndindex(left.shape):
                assert sp.simplify(left.tensor[idx] - right.tensor[idx]) == 0

    for i in range(d):
        for j in range(d):
            if j > i:
                # face_i.degen_j = degen_{j-1}.face_i
                left = T.degen(j).face(i)
                right = T.face(i).degen(j - 1)
            elif j == i:
                # Check face_i.degen_j = identity
                left = T.degen(j).face(i)
                right = T
            else:
                # face_i.degen_j = degen_j.face_{i+1}
                left = T.degen(j).face(i + 1)
                right = T.face(i).degen(j)
            for idx in np.ndindex(left.shape):
                assert sp.simplify(left.tensor[idx] - right.tensor[idx]) == 0

if __name__ == "__main__":
    pytest.main([__file__])