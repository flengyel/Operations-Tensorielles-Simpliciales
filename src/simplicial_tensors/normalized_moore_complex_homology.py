import numpy as np
from collections import deque
from .tensor_ops import face, degen, bdry, is_degen, random_tensor
from scipy.linalg import null_space

# Generate a unique key for caching simplices
def key_tensor(t):
    arr = np.array(t, dtype=object)
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
                f = face(s, i)
                fk = min(f.shape) - 1
                if fk < 0: continue
                kf = key_tensor(f)
                if kf not in visited[fk]:
                    visited[fk][kf] = f
                    queue.append((f, fk))
        if k < d + 1:
            for i in range(k + 1):
                g = degen(s, i)
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
            facets.append(zero_arr)
        else:
            facets.append(face(T, i))
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
                f = face(s, i)
                fk = min(f.shape) - 1
                if fk < 0: continue
                kf = key_tensor(f)
                if kf not in visited[fk]:
                    visited[fk][kf] = f
                    queue.append((f, fk))
        if k < d - 1:
            for i in range(k + 1):
                g = degen(s, i)
                gk = min(g.shape) - 1
                if 0 <= gk <= d - 1:
                    kg = key_tensor(g)
                    if kg not in visited[gk]:
                        visited[gk][kg] = g
                        queue.append((g, gk))
    return {k: list(v.values()) for k, v in visited.items() if v}

# Matrix of face map d_i: A[m] -> A[m-1]
def matrix_of_face(Am, Am1, i):
    M = np.zeros((len(Am1), len(Am)))
    lookup = {key_tensor(x): r for r, x in enumerate(Am1)}
    for j, x in enumerate(Am):
        y = face(x, i)
        ky = key_tensor(y)
        if ky in lookup:
            M[lookup[ky], j] = 1
    return M

# Build normalized chain basis N_k = ⋂_{i=1..k} ker(d_i)
def build_normalized(C, face_maps, d):
    N = {}
    C0 = C.get(0, [])
    N[0] = [np.eye(len(C0))[:, i] for i in range(len(C0))]
    for k in range(1, d + 1):
        Ck = C.get(k, [])
        if len(Ck) == 0:
            N[k] = []
            continue
        mats = [face_maps[(k, i)] for i in range(1, k + 1)]
        Mstack = np.vstack(mats) if mats else np.zeros((0, len(Ck)))
        N[k] = null_space(Mstack)  # Using scipy.linalg.null_space
    return N

# Compute H_n relative via mapping cone
def relative_homology_dimension(shape, j):
    T = random_tensor(shape)
    d = min(shape) - 1
    if d < 0: return 0
    A = build_simplicial_object(T)
    L = build_horn_object(T, j)
    
    # Debugging step: check shapes of A and L
    print(f"A shape: {len(A)}, L shape: {len(L)}")
    
    fA = {(k, i): matrix_of_face(A.get(k, []), A.get(k-1, []), i)
          for k in A for i in range(k+1)}
    fL = {(k, i): matrix_of_face(L.get(k, []), L.get(k-1, []), i)
          for k in L for i in range(k+1)}
    
    # Debugging step: print the face map keys and shapes
    print(f"fA keys: {list(fA.keys())}")
    print(f"fL keys: {list(fL.keys())}")
    
    NA = build_normalized(A, fA, d)
    NL = build_normalized(L, fL, d)
    
    # basis mats with bounds checking
    BAd = np.hstack(NA[d]) if len(NA.get(d, [])) > 0 else np.zeros((len(A.get(d, [])), 0))
    BAdm1 = np.hstack(NA[d-1]) if len(NA.get(d-1, [])) > 0 else np.zeros((len(A.get(d-1, [])), 0))
    BLm1 = np.hstack(NL[d-1]) if len(NL.get(d-1, [])) > 0 else np.zeros((len(L.get(d-1, [])), 0))
    BLm2 = np.hstack(NL[d-2]) if len(NL.get(d-2, [])) > 0 else np.zeros((len(L.get(d-2, [])), 0))
    
    # full diffs with bounds checking
    dA_full = fA.get((d,0), np.zeros((BAdm1.shape[0], BAd.shape[1]))) if (d,0) in fA else np.zeros((BAdm1.shape[0], BAd.shape[1]))
    dL_full = fL.get((d-1,0), np.zeros((BLm2.shape[0], BLm1.shape[1]))) if (d-1,0) in fL else np.zeros((BLm2.shape[0], BLm1.shape[1]))
    
    # pseudoinverse change-of-basis
    def pinv(B):
        return np.linalg.pinv(B)
    
    dA_n = pinv(BAdm1) @ dA_full @ BAd
    dL_n = pinv(BLm2) @ dL_full @ BLm1
    
    # inclusion N_{d-1}(L)->N_{d-1}(A)
    idxA = {key_tensor(x): i for i,x in enumerate(A.get(d-1, []))}
    f_full = np.zeros((len(A.get(d-1, [])), len(L.get(d-1, []))))
    for i,x in enumerate(L.get(d-1, [])):
        kx = key_tensor(x)
        if kx in idxA: f_full[idxA[kx], i] = 1
    f_n = pinv(BAdm1) @ f_full @ BLm1
    
    # build cone: [dA_n, f_n; 0, -dL_n]
    top = np.hstack([dA_n, f_n])
    bot = np.hstack([np.zeros((BLm1.shape[0], BAd.shape[1])), -dL_n])
    D_cone = np.vstack([top, bot])
    
    return len(null_space(D_cone))

def main():
    for shape in [(3, 3), (4, 4), (2, 2, 2), (3, 3, 3)]:
        n = min(shape) - 1
        vals = [relative_homology_dimension(shape, j) for j in range(n + 1)]
        print(f"Shape {shape}, n={n}: H_n rel dims = {vals}")
