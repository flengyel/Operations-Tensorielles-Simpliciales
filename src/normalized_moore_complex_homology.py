import sympy as sp
import numpy as np
from collections import deque
from sagemath_compatible_tensor_ops import SymbolicTensor

# Generate a unique key for caching simplices
def key_tensor(t):
    data = t.tensor
    arr = np.array(data, dtype=object)
    try:
        flat = arr.flatten()
    except Exception:
        flat = arr if arr.ndim == 1 else [arr]
    entries = tuple(sorted(str(x) for x in flat))
    return (tuple(t.shape), entries)

# Build full simplicial object A_•(T)
def build_simplicial_object(T):
    d = min(T.shape) - 1
    visited = {k: {} for k in range(d + 2)}
    visited[d][key_tensor(T)] = T
    queue = deque([(T, d)])
    while queue:
        s, k = queue.popleft()
        if k > 0:
            for i in range(k + 1):
                f = s.face(i)
                fk = min(f.shape) - 1
                if fk < 0: continue
                kf = key_tensor(f)
                if kf not in visited[fk]:
                    visited[fk][kf] = f
                    queue.append((f, fk))
        if k < d + 1:
            for i in range(k + 1):
                g = s.degen(i)
                gk = min(g.shape) - 1
                if 0 <= gk <= d + 1:
                    kg = key_tensor(g)
                    if kg not in visited[gk]:
                        visited[gk][kg] = g
                        queue.append((g, gk))
    return {k: list(v.values()) for k, v in visited.items() if v}

# Build horn Λ^d_j simplicial object
def build_horn_object(T, j):
    d = min(T.shape) - 1
    facets = []
    for i in range(d + 1):
        if i == j:
            zero_shape = tuple(s - 1 for s in T.shape)
            zero_arr = np.zeros(zero_shape, dtype=object)
            facets.append(SymbolicTensor(zero_shape, tensor=zero_arr))
        else:
            facets.append(T.face(i))
    visited = {k: {} for k in range(d)}
    queue = deque()
    for f in facets:
        fk = min(f.shape) - 1
        if fk < 0: continue
        visited[fk][key_tensor(f)] = f
        queue.append((f, fk))
    while queue:
        s, k = queue.popleft()
        if k > 0:
            for i in range(k + 1):
                f = s.face(i)
                fk = min(f.shape) - 1
                if fk < 0: continue
                kf = key_tensor(f)
                if kf not in visited[fk]:
                    visited[fk][kf] = f
                    queue.append((f, fk))
        if k < d - 1:
            for i in range(k + 1):
                g = s.degen(i)
                gk = min(g.shape) - 1
                if 0 <= gk <= d - 1:
                    kg = key_tensor(g)
                    if kg not in visited[gk]:
                        visited[gk][kg] = g
                        queue.append((g, gk))
    return {k: list(v.values()) for k, v in visited.items() if v}

# Matrix of face map d_i: A[m] -> A[m-1]
def matrix_of_face(Am, Am1, i):
    M = sp.zeros(len(Am1), len(Am))
    lookup = {key_tensor(x): r for r, x in enumerate(Am1)}
    for j, x in enumerate(Am):
        y = x.face(i)
        ky = key_tensor(y)
        if ky in lookup:
            M[lookup[ky], j] = 1
    return M

# Build normalized chain basis N_k = ⋂_{i=1..k} ker(d_i)
def build_normalized(C, face_maps, d):
    N = {}
    C0 = C.get(0, [])
    N[0] = [sp.eye(len(C0)).col(i) for i in range(len(C0))]
    for k in range(1, d + 1):
        Ck = C.get(k, [])
        if not Ck:
            N[k] = []
            continue
        mats = [face_maps[(k, i)] for i in range(1, k + 1)]
        Mstack = sp.Matrix.vstack(*mats) if mats else sp.zeros(0, len(Ck))
        N[k] = Mstack.nullspace()
    return N

# Compute H_n relative via mapping cone
def relative_homology_dimension(shape, j):
    T = SymbolicTensor(shape)
    d = min(shape) - 1
    if d < 0: return 0
    A = build_simplicial_object(T)
    L = build_horn_object(T, j)
    fA = {(k, i): matrix_of_face(A.get(k, []), A.get(k-1, []), i)
          for k in A for i in range(k+1)}
    fL = {(k, i): matrix_of_face(L.get(k, []), L.get(k-1, []), i)
          for k in L for i in range(k+1)}
    NA = build_normalized(A, fA, d)
    NL = build_normalized(L, fL, d)
    # basis mats
    BAd   = sp.Matrix.hstack(*NA[d])   if NA.get(d)   else sp.zeros(len(A.get(d, [])), 0)
    BAdm1 = sp.Matrix.hstack(*NA[d-1]) if NA.get(d-1) else sp.zeros(len(A.get(d-1, [])), 0)
    BLm1  = sp.Matrix.hstack(*NL[d-1]) if NL.get(d-1) else sp.zeros(len(L.get(d-1, [])), 0)
    BLm2  = sp.Matrix.hstack(*NL[d-2]) if NL.get(d-2) else sp.zeros(len(L.get(d-2, [])), 0)
    # full diffs
    dA_full = fA.get((d,0), sp.zeros(BAdm1.rows, BAd.cols))
    dL_full = fL.get((d-1,0), sp.zeros(BLm2.rows, BLm1.cols))
    # pseudoinverse change-of-basis
    def pinv(B):
        return (B.T*B).inv()*B.T if B.cols>0 and B.rows>=B.cols else sp.zeros(B.cols, B.rows)
    dA_n = pinv(BAdm1) * dA_full * BAd
    dL_n = pinv(BLm2)   * dL_full * BLm1
    # inclusion N_{d-1}(L)->N_{d-1}(A)
    idxA = {key_tensor(x): i for i,x in enumerate(A.get(d-1, []))}
    f_full = sp.zeros(len(A.get(d-1, [])), len(L.get(d-1, [])))
    for i,x in enumerate(L.get(d-1, [])):
        kx = key_tensor(x)
        if kx in idxA: f_full[idxA[kx], i] = 1
    f_n = pinv(BAdm1) * f_full * BLm1
    # build cone: [dA_n, f_n; 0, -dL_n]
    top = sp.Matrix.hstack(dA_n, f_n)
    bot = sp.Matrix.hstack(sp.zeros(BLm1.cols, BAd.cols), -dL_n)
    D_cone = sp.Matrix.vstack(top, bot)
    return len(D_cone.nullspace())

if __name__ == '__main__':
    for shape in [(3,3),(4,4),(2,2,2),(3,3,3)]:
        n = min(shape)-1
        vals = [relative_homology_dimension(shape,j) for j in range(n+1)]
        print(f"Shape {shape}, n={n}: H_n rel dims = {vals}")
