import numpy as np
from typing import Tuple, List, Union, Any

# dimensions d'une matrice sous forme d'un tuple de listes d'indices de lignes et de colonnes
# par exemple, _dims(M) = ([0,1,2,3], [0,1,2,3,4,5,6,7,8])
# Cette précomputation est utilisée pour éviter de recalculer la même liste d'indices
# dans les calculs de frontière




def _dims(M: np.ndarray) -> Tuple[List[np.ndarray]]:
    return tuple([np.arange(dim_size) for dim_size in M.shape])

# Généralisation de l'opération face aux tenseurs
def _face(M: np.ndarray, axes: Tuple[List[np.ndarray]], i: int) -> np.ndarray:
    indices = [np.delete(axis, i) if len(axis) > i else axis for axis in axes]
    grid = np.ix_(*indices)
    return M[grid]

# i-ème face d'une matrice
def face(M: np.ndarray, i: int) -> np.ndarray:
    axes = _dims(M)
    return _face(M, axes, i)

# i-ème face horizontale d'une matrice, avec des indices dimensionnels donnés par les lignes et les colonnes
def _hface(M: np.ndarray, rows: np.ndarray, cols: np.ndarray, i: int) -> np.ndarray:
    grid = np.ix_(np.delete(rows,i),cols)
    return M[grid]

# i-ème face horizontale d'une matrice
def hface(M: np.ndarray, i: int) -> np.ndarray:
    (r, c) = _dims(M)
    return _hface(M, r, c, i)

# i-ème face verticale d'une matrice, avec des indices dimensionnels donnés par les lignes et les colonnes
def _vface(M: np.ndarray, rows: np.ndarray, cols: np.ndarray, i: int) -> np.ndarray:
    grid = np.ix_(rows,np.delete(cols,i))
    return M[grid]

#  i-ème face verticale d'une matrice
def vface(M: np.ndarray, i: int) -> np.ndarray:
    (r, c) = _dims(M)
    return _vface(M, r, c, i)

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

# Frontière d'un tenseur
def bdry(M: np.ndarray) -> np.ndarray:
    d = np.min(M.shape)
    axes = _dims(M)
    #  soustraire 1 de chaque dimension
    A = np.zeros(np.subtract(M.shape,np.array([1])))
    for i in range(d):
       if i % 2 == 0:
           A = np.add(A, _face(M, axes, i))
       else:
           A = np.subtract(A, _face(M, axes, i))
    return A

#  Frontière horizontale d'une matrice
def hbdry(M: np.ndarray) -> np.ndarray:
    d = M.shape[0]
    rows, cols = _dims(M)
    # soustraire 1 de la dimension zéro
    A = np.zeros(np.subtract(M.shape,np.array([1,0])))
    for i in range(d):
        if i % 2 == 0:
            A = np.add(A, _hface(M, rows, cols, i))
        else:
            A = np.subtract(A, _hface(M, rows, cols, i))
    return A

#  Frontière verticale d'une matrice
def vbdry(M: np.ndarray) -> np.ndarray:
    d = M.shape[1]
    rows, cols = _dims(M)
    # soustraire 1 de la première dimension
    A = np.zeros(np.subtract(M.shape,np.array([0,1])))
    for i in range(d):
        if i % 2 == 0:
            A = np.add(A, _vface(M, rows, cols, i))
        else:
            A = np.subtract(A, _vface(M, rows, cols, i))  
    return A

# cobord d'une matrice. Cela donnera toujours une cohomologie nulle
def cobdry(M: np.ndarray) -> np.ndarray:
    d = np.min(M.shape)
    # Ajoutez 1 à chaque dimension
    A = np.zeros(np.add(M.shape,np.array([1])))
    for i in range(d):
       if i % 2 == 0:
           A = np.add(A, degen(M, i))
       else:
           A = np.subtract(A, degen(M, i))
    return A

# Fonction de cornet, donnée une matrice M et un indice k, 0 <= k <= min(M.shape)
# Ceci retourne la liste des faces diagonales de M en ordre à l'exception de la k-ème.
# La k-ème matrice est la matrice zéro de dimension (M.shape[0]-1)x(M.shape[1]-1)
def horn(M: np.ndarray, k: int) -> np.ndarray:
    d = min(M.shape)
    if k < 0 or k >= d:
        raise ValueError(k, "must be nonnegative and less than dim", d)
    return np.array([face(M,i) if i != k else np.zeros(np.subtract(M.shape, np.array([1]))) for i in range(d)])

# check the Kan condition
def Kan_condition(W: np.ndarray, k: int) -> bool:
    d = len(W)
    for j in range(d):
        for i in range(j): # i < j
            if i < j and i != k and j != k:
                if not np.array_equal(face(W[j],i), face(W[i],j-1)):
                   return False
    return True

# Moore's algorithm for producing a filler of a horn
# Based on Lemma 8.2.6 in Weibel 2003, pp 262.
# The function filler() computes a hypermatrix, 
# given a horn H which omits the k-th face
# Here a k-horn has a zero matrix at the k-th position
def filler(H: np.ndarray, k: int) -> np.ndarray:
    g = degen(H[k],0) # zero matrix at k-th position in Horn
    for r in range(k):
        u = np.subtract(face(g, r), H[r])
        g = np.subtract(g, degen(u, r))
    # begin the downward induction
    t = len(H)-1
    while t > k:
        z = degen(np.subtract(H[t], face(g,t)), t-1)
        g = np.add(g, z)
        t -= 1
    return g

def standard_basis_matrix(m: int, n: int, i: int, j: int) -> np.ndarray:
    # Create a zero matrix of dimensions (m, n)
    M = np.zeros((m, n))
    # Set the element at (i, j) to 1
    M[i, j] = 1
    return M

# the conjecture is that inner horns are non-unique if and only if
# the rank of a tensor is greater than or equal to its simplicial dimension
def tensor_inner_horn_rank_dimension_conjecture(shape: Tuple[int], verbose: bool = False) -> bool:
    rank = len(shape)
    dim = min(shape)-1 # simplicial dimension
    conjecture = rank >= dim
    if verbose:
        print( "shape:",shape,"dim:", dim, "<=", rank, ":rank", conjecture )
    return conjecture

# custom exception for simplicial tensors
class FillerException(Exception):
    pass

def tensor_inner_horn_rank_dimension_comparison(shape: Tuple[int], verbose: bool = False) -> bool:
    rank = len(shape)
    dim = min(shape)-1
    A = np.random.randint(low=-10, high=10, size=shape, dtype=np.int16)
    for i in range(1,dim+1):
        H = horn(A, i)
        B = filler(H, i)
        Hprime = horn(B, i)
        if not np.array_equal(H,Hprime):
            raise FillerException("Original horn and filler horn disagree!")
        if np.array_equal(A, B):
            if verbose:
                print("Unique filler.")
            return False
    if verbose:
        print("Non unique filler.")
    return True

if __name__ == "__main__":
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
                raise FillerException("Original horn and filler horn disagree!")
            if not np.array_equal(A, B):
                print("Reconstructed matrix disagreement.", A, B, sep="\n" )
                return False
        print("All reconstructions agree", A, B, sep="\n")
        return True
    # Because the horns are ordered, the matrices can be reconstructed from their inner horns
    A = np.array([[2, 4, 3, 4], 
                  [5, 2, 3, 3],
                  [6, 6, 2, 4],
                  [5, 6, 5, 2]])
    checkReconstruction(A)    
    B = np.array([[2, 3, 4, 3], 
                  [6, 2, 4, 4],
                  [5, 5, 2, 3],
                  [6, 5, 6, 2]])
    checkReconstruction(B)    
    print("Faces of A", [face(A, i) for i in range(4)])
    print("Faces of B", [face(B, i) for i in range(4)])
    