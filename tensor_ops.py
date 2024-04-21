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
from typing import Tuple, List, Union, Any
import logging

# Set up logging
logging.basicConfig(level=logging.WARNING)

# dimensions d'une matrice sous forme d'un tuple de listes d'indices de lignes et de colonnes
# par exemple, _dims(M) = ([0,1,2,3], [0,1,2,3,4,5,6,7,8])
# Cette précomputation est utilisée pour éviter de recalculer la même liste d'indices
# dans les calculs de frontière

def _dims(m: np.ndarray) -> Tuple[List[np.ndarray]]:
    return tuple([np.arange(dim_size) for dim_size in m.shape])

# Généralisation de l'opération face aux tenseurs
def _face(m: np.ndarray, axes: Tuple[List[np.ndarray]], i: int) -> np.ndarray:
    indices = [np.delete(axis, i) if len(axis) > i else axis for axis in axes]
    grid = np.ix_(*indices)
    return m[grid]

# i-ème face d'une matrice
def face(m: np.ndarray, i: int) -> np.ndarray:
    axes = _dims(m)
    return _face(m, axes, i)

# i-ème face horizontale d'une matrice, avec des indices dimensionnels donnés par les lignes et les colonnes
def _hface(m: np.ndarray, rows: np.ndarray, cols: np.ndarray, i: int) -> np.ndarray:
    grid = np.ix_(np.delete(rows,i),cols)
    return m[grid]

# i-ème face horizontale d'une matrice
def hface(m: np.ndarray, i: int) -> np.ndarray:
    (r, c) = _dims(m)
    return _hface(m, r, c, i)

# i-ème face verticale d'une matrice, avec des indices dimensionnels donnés par les lignes et les colonnes
def _vface(m: np.ndarray, rows: np.ndarray, cols: np.ndarray, i: int) -> np.ndarray:
    grid = np.ix_(rows,np.delete(cols,i))
    return m[grid]

#  i-ème face verticale d'une matrice
def vface(m: np.ndarray, i: int) -> np.ndarray:
    (r, c) = _dims(m)
    return _vface(m, r, c, i)

#  j-ème dégénérescence verticale d'une matrice
def vdegen(z: np.ndarray, j: int) -> np.ndarray:
    return np.insert(z, j, z[:, j], axis=1)

#  j-ème dégénérescence horizontale d'une matrice
def hdegen(z: np.ndarray, j: int) -> np.ndarray:
    return np.insert(z, j, z[j, :], axis=0)

# j-ème dégénérescence d'une matrice
def degen_matrix(z: np.ndarray, j: int) -> np.ndarray:
    z_with_row = np.insert(z, j, z[j, :], axis=0)
    z_with_row_and_col = np.insert(z_with_row, j, z_with_row[:, j], axis=1)
    return z_with_row_and_col

# k-ème dégénérescence d'un tenseur
def degen(z: np.ndarray, k: int) -> np.ndarray:
    # Parcourez chaque dimension et dupliquez la k-ème hypercolonne
    for axis in range(z.ndim):
        slices = [slice(None)] * z.ndim
        slices[axis] = k
        z = np.insert(z, k, z[tuple(slices)], axis=axis)
    return z

def is_degen(a: np.ndarray) -> bool:
    d = min(a.shape) 
    for i in np.arange(d-1): # faces have dimension d-1
        if np.array_equal(a, degen(face(a, i), i)):
            return True
    return False

def find_degen(a: np.ndarray) -> Union[Tuple[np.ndarray, int], None]:   
    d = min(a.shape) 
    for i in np.arange(d-1): # faces have dimension d-1
        if np.array_equal(a, degen(face(a, i), i)):
            return face(a, i), i
    return None    

# Decompose a degenerate matrix into a non-degenerate base matrix and a sequence of degeneracy operations
# Example usage
# A = np.array([[...], [...]])  # Replace with an actual hypermatrix
# non_degenerate_base, ops = decomposeDegeneracy(A)
# print("Non-degenerate base matrix:", non_degenerate_base)
# print("Sequence of degeneracy operations:", ops)

def decompose_degen(a: np.ndarray) -> Tuple[np.ndarray, list]:
    operations = []

    def helper(b: np.ndarray, ops: list) -> np.ndarray:
        nonlocal operations
        degeneracy_info = find_degen(b)
        
        # Base case: If B is non-degenerate or no degeneracy found, set the operations
        if degeneracy_info is None:
            operations = ops
            return b
        
        # Recursive case
        f, i = degeneracy_info
        return helper(f, ops + [(f, i)])

    # Start the decomposition
    non_degenerate_base = helper(a, [])
    return non_degenerate_base, operations

# Frontière d'un tenseur
def bdry(m: np.ndarray) -> np.ndarray:
    d = np.min(m.shape)
    axes = _dims(m)
    #  soustraire 1 de chaque dimension
    a = np.zeros(np.subtract(m.shape,np.array([1])))
    for i in range(d):
       if i % 2 == 0:
           a = np.add(a, _face(m, axes, i))
       else:
           a = np.subtract(a, _face(m, axes, i))
    return a

#  Frontière horizontale d'une matrice
def hbdry(m: np.ndarray) -> np.ndarray:
    d = m.shape[0]
    rows, cols = _dims(m)
    # soustraire 1 de la dimension zéro
    a = np.zeros(np.subtract(m.shape,np.array([1,0])))
    for i in range(d):
        if i % 2 == 0:
            a = np.add(a, _hface(m, rows, cols, i))
        else:
            a = np.subtract(a, _hface(m, rows, cols, i))
    return a

#  Frontière verticale d'une matrice
def vbdry(m: np.ndarray) -> np.ndarray:
    d = m.shape[1]
    rows, cols = _dims(m)
    # soustraire 1 de la première dimension
    a = np.zeros(np.subtract(m.shape,np.array([0,1])))
    for i in range(d):
        if i % 2 == 0:
            a = np.add(a, _vface(m, rows, cols, i))
        else:
            a = np.subtract(a, _vface(m, rows, cols, i))  
    return a

# cobord d'une matrice. Cela donnera toujours une cohomologie nulle
def cobdry(m: np.ndarray) -> np.ndarray:
    d = np.min(m.shape)
    # Ajoutez 1 à chaque dimension
    a = np.zeros(np.add(m.shape,np.array([1])))
    for i in range(d):
       if i % 2 == 0:
           a = np.add(a, degen(m, i))
       else:
           a = np.subtract(a, degen(m, i))
    return a

# Fonction de cornet, donnée une matrice M et un indice k, 0 <= k <= min(M.shape)
# Ceci retourne la liste des faces diagonales de M en ordre à l'exception de la k-ème.
# La k-ème matrice est la matrice zéro de dimension (M.shape[0]-1)x(M.shape[1]-1)
def horn(m: np.ndarray, k: int) -> np.ndarray:
    d = min(m.shape)
    if k < 0 or k >= d:
        raise ValueError(k, "must be nonnegative and less than dim", d)
    return np.array([face(m,i) if i != k else np.zeros(np.subtract(m.shape, np.array([1]))) for i in range(d)])

# check the Kan condition
def kan_condition(w: np.ndarray, k: int) -> bool:
    d = len(w)
    for j in range(d):
        for i in range(j): # i < j
            if i < j and i != k and j != k:
                if not np.array_equal(face(w[j],i), face(w[i],j-1)):
                   return False
    return True

# Moore's algorithm for producing a filler of a horn
# Based on Lemma 8.2.6 in Weibel 2003, pp 262.
# The function filler() computes a hypermatrix, 
# given a horn H which omits the k-th face
# Here a k-horn has a zero matrix at the k-th position
def filler(horn: np.ndarray, k: int) -> np.ndarray:
    g = degen(horn[k],0) # zero matrix at k-th position in Horn
    for r in range(k):
        u = np.subtract(face(g, r), horn[r])
        g = np.subtract(g, degen(u, r))
    # begin the downward induction
    t = len(horn)-1
    while t > k:
        z = degen(np.subtract(horn[t], face(g,t)), t-1)
        g = np.add(g, z)
        t -= 1
    return g

def standard_basis_matrix(m: int, n: int, i: int, j: int) -> np.ndarray:
    # Create a zero matrix of dimensions (m, n)
    s = np.zeros((m, n))
    # Set the element at (i, j) to 1
    s[i, j] = 1
    return s

def s_dim(t: np.ndarray) -> int:
    """
    Calculates the simplex dimension (s-dim) of a hypermatrix t, defined as min(t.shape)-1.
    This represents the dimension of the hypermatrix t in the simplicial hypermatrix R-module
    generated by t, where R is a (commutative) ring and t has entries in the R-module M.

    Parameters:
        t (np.ndarray): The hypermatrix for which to calculate the simplex dimension.

    Returns:
        int: The simplex dimension of the hypermatrix.
    """
    return min(t.shape) - 1

def degree(t: np.ndarray) -> int:
    """
    Calculates the degree of a hypermatrix t, defined as the number of dimensions of t.

    Parameters:
        t (np.ndarray): The hypermatrix for which to calculate the degree.

    Returns:
        int: The degree of the hypermatrix.
    """
    return len(t.shape)


# Definition: the degree of a tensor t is the number of dimensions of t.

# The n-hypergroupoid conjecture. Let t be a (generic) non-degenerate hypermatrix
# with non-degenerate boundary. Then the inner horns of t are unique if and only if 
# the degree of t is less than its simplicial dimension: deg(t) < s_dim(t).
def n_hypergroupoid_conjecture(shape: Tuple[int], verbose: bool = False) -> bool:
    deg = len(shape)
    s_dimension = min(shape)-1 # simplicial dimension
    conjecture = deg < s_dimension
    if verbose:
        print(f"shape:{shape} rank:{deg} {'<' if conjecture else '>='} s-dim:{s_dimension}")
    return conjecture

# custom exception for simplicial tensors
class SimplicialException(Exception):
    pass

# Conjecture. Supppse that neither a, nor bdry(a) is degenerate. 
# Then every inner horn of a has a unique filler if and only if deg(a) < s_dim(a)
def n_hypergroupoid_comparison(a: np.ndarray, verbose: bool = False) -> bool:
    if is_degen(a):
        raise SimplicialException("Matrix is degenerate.")
    sdim = s_dim(a)
    for i in range(1,sdim+1):
        h = horn(a, i)
        b = filler(h, i)
        hprime = horn(b, i)
        if not np.array_equal(h,hprime):
            raise SimplicialException("Original horn and filler horn disagree!")
        if not np.array_equal(a, b):
            if verbose:
                print("There exist at least two fillers.")
            return False
    if verbose:
        print("Unique filler.")
    return True

# max norm of a hypermatrix
def max_norm(hyp : np.ndarray) -> Union[int, float]:
    return np.max(np.abs(hyp))

# Normed boundary
def bdry_n(h: np.ndarray) -> np.ndarray:
    if np.any(h):  # Check if h is non-zero
        norm = max_norm(h)
        if norm != 0:
            return bdry(h) / norm
        else:
            # Log a warning if h is non-zero but max_norm is effectively zero
            logging.warning("Non-zero array has an effective max norm of zero. Returning bdry(h) instead.")
            return bdry(h)  # Return bdry(h) instead of h when norm is zero
    else:  # Return bdry(h) directly if it is all zeros
        return bdry(h)

if __name__ == "__main__":
  
    # more counterexamples from manvel and stockmeyer 1971
    def matrix_m(n: int) -> np.ndarray:
        j = int(np.ceil(n // 2))
        A = np.zeros((n,n))
        A[0,j] = 1
        A[j,0] = 1
        return A

    def matrix_n(n: int) -> np.ndarray:
        A = np.zeros((n,n))
        j = int(np.ceil(n // 2))+1
        A[0,j] = 1
        A[j,0] = 1
        return A
    
    m = matrix_m(5)
    print("M:", m)
    print("is_degen(M):", is_degen(m))
    non_degen_base, ops = decompose_degen(m)
    print("Non-degenerate base matrix:", non_degen_base)
    print("Sequence of degeneracy operations:", ops)

    n = degen(degen(degen(matrix_n(3),2),3),4)
    print("n:", n)
    print("is_degen(n):", is_degen(n))
    non_degenerate_base, ops = decompose_degen(n)
    print("Non-degenerate base matrix:", non_degenerate_base)
    print("Sequence of degeneracy operations:", ops)

    X = np.array([[0, 0, 1, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0,],
                [0, 0, 0, 0]])
    print("X:", X)
    print("is_degen(X):", is_degen(X))
    non_degenerate_base, ops = decompose_degen(X)
    print("Non-degenerate base matrix:", non_degenerate_base)
    print("Sequence of degeneracy operations:", ops)

    Y = np.array([[0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0,],
                [0, 0, 0, 0]])
    print("Y:", Y)
    print("is_degen(Y):", is_degen(Y))
    non_degenerate_base, ops = decompose_degen(Y)
    print("Non-degenerate base matrix:", non_degenerate_base)
    print("Sequence of degeneracy operations:", ops)

    a = np.array([0, 1, 2])
    b = np.array([3, 4])
    c = np.array([5, 6])

    # Create an empty 3D array to store the tensor product
    tensor = np.zeros((3, 2, 2), dtype=int)

    # Fill the tensor by taking the pairwise product of elements from the 1D arrays
    for i in range(3):
        for j in range(2):
            for k in range(2):
                tensor[i, j, k] = a[i] * b[j] * c[k]
    print(tensor)
    n_hypergroupoid_comparison(tensor, verbose=True)
    # now compute the inner horn of the tensor one step at a time
    deg = degree(tensor)
    sdim = s_dim(tensor)
    print(f"shape {tensor.shape} degree {deg} s-dim {sdim}")
    for i in range(1, sdim+1):
        H = horn(tensor, i)
        B = filler(H, i)
        Hprime = horn(B, i)
        print("tensor:", tensor)
        print("B:", B)
        print("Hprime:", Hprime)
        print("np.array_equal(H,Hprime):", np.array_equal(H,Hprime))
        print("np.array_equal(tensor, B):", np.array_equal(tensor, B))

    hypermatrix = np.array([[[1, -2, 3], [4, -5, 6], [7, 8, -9]],
                        [[-1, 2, -3], [-4, 5, -6], [-7, -8, 9]],
                        [[10, -11, 12], [-13, 14, -15], [16, 17, -18]]])

    print(f"hypermatrix: {hypermatrix}")
    print(f"Max norm of hypermatrix: {max_norm(hypermatrix)}")

    print(f"bdry_n(hypermatrix) = {bdry_n(hypermatrix)}")
    print(f"bdry_n(hypermatrix) = {bdry_n(hypermatrix)}")
    print(f"bdry_n(bdry_n(hypermatrix)) = {bdry_n(bdry_n(hypermatrix))}")
