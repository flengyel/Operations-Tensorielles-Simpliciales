#    Opérations Tensorielles Simpliciales
#    Simplicial Operations on Matrices and Hypermatrices
#
#    Copyright (C) 2021-2024 Florian Lengyel
#    Email: florian.lengyel at cuny edu, florian.lengyel at gmail
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

import numpy as np
from numpy import ndarray
import pytest
from typing import Tuple

from tensor_ops import SimplicialException, face, hface, vface, bdry, hbdry, vbdry 
from tensor_ops import degen, hdegen, vdegen, horn, kan_condition, filler
from tensor_ops import standard_basis_matrix, cobdry
from tensor_ops import n_hypergroupoid_comparison, n_hypergroupoid_conjecture
from tensor_ops import is_degen, decompose_degen, max_norm, bdry_n

Z = np.arange(7*9)
Z = Z.reshape(7,9)

def test_face() -> None:
    expected_face = np.array([[ 0,  1,  2,  3,  4,  6,  7,  8],
                              [ 9, 10, 11, 12, 13, 15, 16, 17],
                              [18, 19, 20, 21, 22, 24, 25, 26],
                              [27, 28, 29, 30, 31, 33, 34, 35],
                              [36, 37, 38, 39, 40, 42, 43, 44],
                              [54, 55, 56, 57, 58, 60, 61, 62]])
    assert np.allclose(face(Z,5), expected_face)

def test_hface() -> None:
    expected_hface = np.array( [[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
                                [ 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                [18, 19, 20, 21, 22, 23, 24, 25, 26],
                                [36, 37, 38, 39, 40, 41, 42, 43, 44],
                                [45, 46, 47, 48, 49, 50, 51, 52, 53],
                                [54, 55, 56, 57, 58, 59, 60, 61, 62]])
    assert np.allclose(hface(Z,3), expected_hface)  
       
def test_hbdry_hbdry() -> None:
    expected_hbdry_hbdry = np.zeros((5,9))
    assert np.allclose(hbdry(hbdry(Z)), expected_hbdry_hbdry)   

def test_vbdry_vbdry() -> None:
    expected_vbdry_vbdry = np.zeros((7,7))  
    assert np.allclose(vbdry(vbdry(Z)), expected_vbdry_vbdry)

def test_bdry_bdry() -> None:
    expected_bdry_bdry = np.zeros((5,7))  
    assert np.allclose(bdry(bdry(Z)), expected_bdry_bdry)   

def test_bdry_hbdry() -> None:
    expected_bdry_hbdry = np.array([[1., 0., 1., 0., 1., 0., 0., 0.],
                                    [1., 0., 1., 0., 1., 0., 0., 0.],
                                    [1., 0., 1., 0., 1., 0., 0., 0.],
                                    [1., 0., 1., 0., 1., 0., 0., 0.],
                                    [1., 0., 1., 0., 1., 0., 0., 0.]])
    assert np.allclose(bdry(hbdry(Z)), expected_bdry_hbdry) 

## Tests of basic properties of the simplicial operations

# verify that face(A,j) = hface(vface(A,j),j) = vface(hface(A,j),j)
# This is how Daniel Quillen defines the face operation of the
# diagonal simplicial group in Quillen, D.G. 1966. “Spectral Sequences of 
# a Double Semi-Simplical Group.” Topology 5 (2): 155–57. 
# https://doi.org/10.1016/0040-9383(66)90016-4. The application to matrices
# appears new, though it was implicit in Quillen's first paper.

W = np.random.randint(low=-10, high=73, size=(73,109))

def facecommute(A)  -> bool:
    d = min(A.shape)
    for j in range(d):
        B = face(A,j); # face operation of diagonal module
        if not np.array_equal(B, hface(vface(A,j),j)) or not np.array_equal(B, vface(hface(A,j),j)):
            return False
    return True

# the diagonal face operation is the composite of vertical and horizontal face operations
# This is given axiomatically in 
def test_facecommute() -> None:
    assert np.allclose(facecommute(W), True)

def transcommute(A) -> bool:
    d = min(A.shape)
    for j in range(d):
        if not np.array_equal(face(np.transpose(A), j), np.transpose(face(A, j))):
            return False
    return True

# The face operation preserves transposition
def test_transcommute() -> None:
    assert np.allclose(transcommute(W), True)

# Verify that degen(A,j) = hdegen(vdegen(A,j),j) = vdegen(hdegen(A,j),j)
# This is how Daniel Quillen defines the degeneracy operation of the
# simplicial diagonal group in Quillen, D.G. 1966. “Spectral Sequences of 
# a Double Semi-Simplical Group.” Topology 5 (2): 155–57. 
# https://doi.org/10.1016/0040-9383(66)90016-4.

def degencommute(A) -> bool:
    (m,n) = A.shape
    d = min(m,n)
    for j in range(d):
        B = degen(A,j);
        if not np.array_equal(B, hdegen(vdegen(A,j),j)) or not np.array_equal(B, vdegen(hdegen(A,j),j)):
            return False
    return True

def test_degencommute() -> None:
    assert np.allclose(degencommute(W), True) 

# degeneracies commute with transpose

def degentranscommute(A) -> bool:
    (m, n) = A.shape
    d = min(m,n)
    for j in range(d):
        if not np.array_equal(degen(np.transpose(A), j), np.transpose(degen(A, j))):
            return False
    return True

def test_degentranscommute() -> None:
    assert np.allclose(degentranscommute(W), True)

# The five simplicial identities for the diagonal simplicial operations 

def test_first_identity() -> None:
    def first_identity(M) -> bool:
        d = min(M.shape)
        for j in range(d):
            for i in range(j):  # i < j here
                X = face(face(M, j), i)
                Y = face(face(M, i), j-1)
                if not np.array_equal(X, Y):
                    return False
        return True
    assert np.allclose(first_identity(W), True)       

def test_second_identity() -> None:
    def second_identity(M) -> bool:
        d = min(M.shape) # d is the dimension
        for j in range(d):
            for i in range(j):
                X = face(degen(M, j), i)
                Y = degen(face(M, i), j-1)
                if not np.array_equal(X, Y):
                    return False
        return True
    assert np.allclose(second_identity(W), True)

def test_third_identity() -> None:
    def third_identity(M) -> bool:
        d = min(M.shape) # d is the dimension
        for j in range(d):
            X = face(degen(M,j), j)
            Y = face(degen(M,j), j+1)
            if ((not np.array_equal(X, Y)) or (not np.array_equal(X , M)) or (not np.array_equal(Y, M))):
                return False
        return True    
    assert np.allclose(third_identity(W), True)

def test_fourth_identity() -> None:
    def fourth_identity(M) -> bool:
        d = min(M.shape) # d is the dimension
        for i in range(d):
            for j in range(i+1):
                if j+1 < i:
                    X = face(degen(M, j), i)
                    Y = degen(face(M, i-1), j)
                    if not np.array_equal(X , Y):
                        return False
        return True
    assert np.allclose(fourth_identity(W), True)

def test_fifth_identity() -> None:
    def fifth_identity(M) -> bool:
        d = min(M.shape) # d is the dimension
        for j in range(d):
            for i in range(j+1):
                X = degen(degen(M, j), i)
                Y = degen(degen(M, i), j+1)
                if not np.array_equal(X, Y):
                    return False
        return True
    assert np.allclose(fifth_identity(W), True)

# module structure (horizontal)

def h_module(A: ndarray, L: ndarray, B: ndarray) -> bool:
    n = A.shape[1]
    p = L.shape[0]
    d = min(n,p)
    X = np.dot(A, L)
    print(X.shape)
    print(X)
    for i in range(d):
        if not np.array_equal(hface(X,i), np.dot(hface(A,i), L)):
            return False
    q = L.shape[1]
    r = B.shape[0]
    e = min(q,r)
    Y = np.dot(L, B)
    for i in range(e):
        if not np.array_equal(hface(Y,i), np.dot(hface(L,i), B)):
            return False
    return True

def test_h_module() -> None:
    A = np.random.randint(low=-11, high=73, size=(7,5))
    L = np.random.randint(low=-133, high=103, size=(5,5))
    B = np.random.randint(low=44, high=83, size=(5,13))
    assert np.allclose(h_module(A,L,B), True)

def h_module2(A: ndarray, L: ndarray, B: ndarray) -> bool:
    n = A.shape[1]
    p = L.shape[0]
    d = min(n,p)
    X = np.dot(A, L)
    print(X.shape)
    print(X)
    for i in range(d):
        if not np.array_equal(hdegen(X,i), np.dot(hdegen(A,i), L)):
            return False
    q = L.shape[1]
    r = B.shape[0]
    e = min(q,r)
    Y = np.dot(L, B)
    for i in range(e):
        if not np.array_equal(hdegen(Y,i), np.dot(hdegen(L,i), B)):
            return False
    return True

def test_h_module2() -> None:
    A = np.random.randint(low=-11, high=73, size=(7,5))
    L = np.random.randint(low=-133, high=103, size=(5,5))
    B = np.random.randint(low=44, high=83, size=(5,13))
    assert np.allclose(h_module2(A,L,B), True)

# face and degeneracy operations commute with the hadamard product
def hadamard_face(A: ndarray, B: ndarray) -> bool:
    if A.shape != B.shape:
        return False
    d = min(A.shape)
    for i in range(d):
        if not np.array_equal(face(np.multiply(A, B), i), np.multiply(face(A,i), face(B, i)) ):
            return False
    return True

def test_hadamard_face() -> None:
    A = np.random.randint(low=-11, high=73, size=(7,11))
    B = np.random.randint(low=1, high=43, size=(7,11))
    assert np.allclose(hadamard_face(A, B), True)

def hadamard_degen(A: ndarray, B: ndarray) -> bool:
    if A.shape != B.shape:
        return False
    d = min(A.shape)
    for i in range(d):
        if not np.array_equal(degen(np.multiply(A, B), i), np.multiply(degen(A,i), degen(B, i)) ):
            return False
    return True

def test_hadamard_degen() -> None:
    A = np.random.randint(low=-11, high=73, size=(7,11))
    B = np.random.randint(low=1, high=43, size=(7,11))
    assert np.allclose(hadamard_degen(A, B), True)

# The transpose of the boundary is the boundary of the transpose
def test_transpose_bdry() -> None:
    def transpose_bdry(A) -> bool:
        return(np.array_equal(np.transpose(bdry(A)), bdry(np.transpose(A))))
    A = np.random.randint(low=-11, high=73, size=(7,11,4))
    assert np.allclose(transpose_bdry(A), True)

# The face operations are linear and commute with hermitian transpose as well
def test_linear_map() -> None:
    def linear_map() -> bool:
        A = np.random.randint(low=-11, high=73, size=(7,11))
        B = np.random.randint(low=1, high=43, size=(7,11))
        d = min(A.shape)
        for i in range(d):
            if not np.array_equal(face(np.add(A, B), i), np.add(face(A,i), face(B,i))):
                return False
            if not np.array_equal(degen(np.add(A, B), i), np.add(degen(A,i), degen(B,i))):
                return False
            if not np.array_equal(6 * face(A,i), face(6*A,i)):
                return False
            if not np.array_equal(-6 * degen(A,i), degen(-6*A,i)):
                return False
        return True
    assert np.allclose(linear_map(), True)

# Simplicial matrix modules are fibrant
# Check that the k-horn function produces a list of matrix-simplices that satisfies the Kan condition    

def test_kan_condition() -> None:
    def _kan_condition() -> bool:
        X = np.random.randint(low=-11, high=11, size=(11,11))
        d = min(X.shape)
        for k in range(d):
            H = horn(X, k)
            if not kan_condition(H, k):
                return False
        return True
    assert np.allclose(_kan_condition(), True)

# The filler is not unique in dimension 2 (a 3x3 matrix has dimension max(shape)-1)
def test_filler_dimension2() -> None:
    X = np.array([[-6,  5,  7],
                  [-5,  4,  8],
                  [ 1,  9,  0]])
    H = horn(X, 1)
    Y = filler(H, 1)
    HH = horn(Y, 1)
    # This tests the existence of a non-unique filler im dimension two
    assert np.allclose(np.array_equal(H, HH) and not np.array_equal(X, Y), True)

# in dimension 3 and higher, the filler is unique
def test_filler_dimension3() -> None:
    X = np.random.randint(low=-11, high=73, size=(8,4))
    H = horn(X, 1)
    Y = filler(H, 1)
    assert np.allclose(np.array_equal(X, Y), True)

# in dimension 10 the filler is unique
def test_filler_dimension10() -> None:
    X = np.random.randint(low=-11, high=11, size=(11,11))
    H = horn(X, 2)
    Y = filler(H, 2)
    assert np.allclose(np.array_equal(X, Y), True)

def test_standard_basis_matrix() -> None:
    # Example usage
    m = 3
    n = 3
    i = 1
    j = 2
    M_ij = standard_basis_matrix(m, n, i, j)
    B = np.array([[0., 0., 0.],
                  [0., 0., 1.],
                  [0., 0., 0.]])
    assert np.allclose(M_ij, B)

def test_coboundary() -> None:
    X = np.random.randint(low=-11, high=11, size=(11,11))
    Y = bdry(X)
    Z = cobdry(bdry(Y))    
    print(Z)
    assert np.allclose(np.linalg.matrix_rank(Z), 0)

def test_tensor_kan_condition() -> None:
    def _kan_condition() -> bool:
        X = np.random.randint(low=-11, high=11, size=(4,4,4,4))
        d = min(X.shape)
        for k in range(d):
            H = horn(X, k)
            if not kan_condition(H, k):
                return False
        return True
    assert np.allclose(_kan_condition(), True)

def test_n_hypergroupoid_conjecture() -> None:
    import random
    def random_shape() -> Tuple[int]:
        length = random.randint(2, 10)  # Length of at least two and bounded by 10
        return tuple(random.randint(2, 10) for _ in range(length))  # Positive integers at least two and bounded by 10

    shape = random_shape()
    # create a random non-zero tensor of the given shape
    A = np.random.randint(low=1, high=10, size=shape, dtype=np.int16)
    # exclude known counterexamples
    while is_degen(A) or is_degen(bdry(A)):
        A = np.random.randint(low=1, high=10, size=shape, dtype=np.int16)
    assert np.allclose(n_hypergroupoid_comparison(A),
                       n_hypergroupoid_conjecture(shape))

def test_manvel_stockmeyer() -> None:
    # Counterexample from On Reconstruction of Matrices
    # Bennet Manvel and Paul K. Stockmeyer
    # Mathematics Magazine , Sep., 1971, Vol. 44, No. 4 (Sep., 1971), pp. 218-221 
    def checkReconstruction(A: np.ndarray) -> bool:
        dim  = min(A.shape)-1 
        for i in range(dim+1):
            H = horn(A, i)
            B = filler(H, i)
            Hprime = horn(B, i)
            if not np.array_equal(H,Hprime):
                raise SimplicialException("Original horn and filler horn disagree!")
            if not np.array_equal(A, B):
                print("Reconstructed matrix disagreement.", A, B, sep="\n" )
                return False
        print("All reconstructions agree", A, B, sep="\n")
        return True

    # Since the set S of principal submatrices of A equals the set of principal 
    # submatrices of B, A and B cannot be reconstructed from S. However, A and B
    # can be reconstructed from their inner horns.

    A = np.array([[2, 4, 3, 4], 
                  [5, 2, 3, 3],
                  [6, 6, 2, 4],
                  [5, 6, 5, 2]])
    B = np.array([[2, 3, 4, 3], 
                  [6, 2, 4, 4],
                  [5, 5, 2, 3],
                  [6, 5, 6, 2]])
    assert np.allclose(checkReconstruction(A) and checkReconstruction(B), True)

def test_is_degen() -> None:
    D = np.array([[[4, 4, 8, 2],[4, 4, 8, 2],[4, 4, 2, 8],[1, 1, 6, 7],[1, 1, 8, 8]],
    [[4, 4, 8, 2],[4, 4, 8, 2],[4, 4, 2, 8],[1, 1, 6, 7],[1, 1, 8, 8]], 
    [[5, 5, 9, 9],[5, 5, 9, 9],[1, 1, 2, 6],[5, 5, 4, 4],[1, 1, 3, 6]],
    [[2, 2, 7, 4],[2, 2, 7, 4],[2, 2, 3, 9],[7, 7, 7, 2],[9, 9, 3, 4]]])
    assert np.allclose(is_degen(D), True)

def test_decompose_degen() -> None:
    # more counterexamples from manvel and stockmeyer 1971
    def matrix_n(n: int) -> np.ndarray:
        A = np.zeros((n,n))
        j = int(np.ceil(n // 2))+1
        A[0,j] = 1
        A[j,0] = 1
        return A
    
    def comp_op_seq(ops1, ops2):
        if len(ops1) != len(ops2):
            return False
        for (array1, int1), (array2, int2) in zip(ops1, ops2):
            if not np.array_equal(array1, array2) or int1 != int2:
                return False
        return True

    N = degen(degen(degen(matrix_n(3),2),3),4)
    isDegen = is_degen(N)
    non_degenerate_base, ops = decompose_degen(N)
    print("Non-degenerate base matrix:", non_degenerate_base)
    print("Sequence of degeneracy operations:", ops)
    assert np.allclose(isDegen 
                       and np.array_equal( non_degenerate_base, matrix_n(3))
                       and comp_op_seq(ops, 
                                          [(np.array([[0., 0., 1., 1., 1.],
                                                   [0., 0., 0., 0., 0.],
                                                   [1., 0., 0., 0., 0.],
                                                   [1., 0., 0., 0., 0.],
                                                   [1., 0., 0., 0., 0.]]), 2), 
                                           (np.array([[0., 0., 1., 1.],
                                                   [0., 0., 0., 0.],
                                                   [1., 0., 0., 0.],
                                                   [1., 0., 0., 0.]]), 2), 
                                           (np.array([[0., 0., 1.],
                                                   [0., 0., 0.],
                                                   [1., 0., 0.]]), 2)]), True)

def test_counterexample_with_degenerate_boundary() -> None:
    # Counterexample with degenerate boundary: 
    # [[6 4 7 2 4 7 5 6]
    # [4 3 6 3 9 5 3 4]
    # [6 5 7 7 1 8 4 9]]
    # isDegeneracy(counterexample2): False
    # bdry(counterexample):  
    # [[3. 3. 3. 9. 5. 3. 4.]
    # [3. 3. 3. 9. 5. 3. 4.]]
    # isDegeneracy(bdry(counterexample)): True
    # Unique filler.
    # shape:(3, 8) rank:2 >= dim:2
    # Conjecture: False Comparison: True
    # Counterexample and filler agree: True
    counterexample2 = np.array([[6, 4, 7, 2, 4, 7, 5, 6],
                                [4, 3, 6, 3, 9, 5, 3, 4],
                                [6, 5, 7, 7, 1, 8, 4, 9]])
    is_degenerate = is_degen(counterexample2)
    bdry_is_degen = is_degen(bdry(counterexample2))
    comparison = n_hypergroupoid_comparison(counterexample2, verbose=True)
    conjecture = n_hypergroupoid_conjecture(counterexample2.shape, verbose=True)
    assert np.allclose(not is_degenerate and bdry_is_degen and comparison and not conjecture, True)
    
    
def test_normed_bdry() -> None:
    A = np.random.randint(low=-11, high=73, size=(5, 7, 9, 11))   
    expected_bdry_bdry = np.zeros((3,5,7,9))
    assert np.allclose(bdry_n(bdry_n(A)), expected_bdry_bdry)

if __name__ == "__main__":
    pytest.main([__file__])
    exit(0)
    
    def pretty_print_coefficients(coefficients: dict) -> None: 
        # coefficients[i, j, k, l] holds the coefficient for mapping M_{i, j} to M_{k, l} 
        for key, value in coefficients.items():
            print(f"({key[0]},{key[1]}): [")
            for coeff, matrix in value:
                formatted_matrix = np.array2string(matrix, formatter={'float_kind': lambda x: "%.2f" % x})
                print(f"  ({coeff}, {formatted_matrix})")
            print("]")

    m, n = 3, 3
    coefficients = {}

    # Loop through each standard basis matrix in the domain
    for i in range(m):
        for j in range(n):
            domain_matrix = standard_basis_matrix(m, n, i, j)
            codomain_matrix = bdry(domain_matrix)
        
            coefficients[(i, j)] = []

            # Express codomain_matrix as a linear combination of standard basis matrices in the codomain
            for k in range(m-1):
                for l in range(n-1):
                    codomain_basis_matrix = standard_basis_matrix(m-1, n-1, k, l)
                    coeff = np.sum(codomain_matrix * codomain_basis_matrix)
                    coefficients[(i, j)].append((coeff, standard_basis_matrix(m-1, n-1, k, l)))            # coefficients[i, j, k, l] now holds the coefficient for mapping M_{i, j} to M_{k, l}
    
    #print(coefficients)
    # Custom pretty print
    pretty_print_coefficients(coefficients)

    X = np.random.randint(low=-11, high=11, size=(11,11))
    print(X)
    print(np.linalg.matrix_rank(X))
    
    W = cobdry(X)
    print(W)
    print(np.linalg.matrix_rank(W))
    
    Y = bdry(X)
    print(Y)
    print(np.linalg.matrix_rank(Y))

    Z = cobdry(Y)  
    print(Z)
    print(np.linalg.matrix_rank(Z))  
    
    
   # for i in range(3):
   #     for j in range(3):
   #         print( bdry(standard_basis_matrix(3,3,i,j)) )
