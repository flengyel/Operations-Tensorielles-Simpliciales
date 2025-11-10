# simplicial permutation group

import numpy as np
import itertools

def perm_matrix(perm):
    n = len(perm)
    M = np.zeros((n,n), dtype=int)
    for i,j in enumerate(perm):
        M[i,j] = 1
    return M

def face_matrix(M, i):
    # M is an (n+1)x(n+1) permutation matrix, 0 <= i <= n
    e_i = np.zeros((M.shape[0],1), dtype=int)
    e_i[i,0] = 1
    # rank-one update:
    A = M + (M @ e_i) @ (e_i.T @ M)
    # delete i-th row and column
    A2 = np.delete(np.delete(A, i, axis=0), i, axis=1)
    return A2

def is_perm(M):
    return (
        ((M.sum(axis=0)==1).all()) and
        ((M.sum(axis=1)==1).all()) and
        np.logical_or(M==0, M==1).all()
    )

# test for n=6 (i.e. S_5 → S_4 faces)
n = 6
for perm in itertools.permutations(range(n+1)):
    M = perm_matrix(perm)
    print(f"perm={perm}, M={M}")
    for i in range(n+1):
        F = face_matrix(M, i)
        print(F)
        assert is_perm(F), f"Failed at perm={perm}, i={i}"
        print(f"Checked face matrix for i={i} with result={is_perm(F)}")
print("All tests passed.")
