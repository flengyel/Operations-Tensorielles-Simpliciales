import sympy as sp
import numpy as np
from collections import deque
from sagemath_compatible_tensor_ops import SymbolicTensor

# Unique key for caching
def key_tensor(t):
    data = t.tensor
    try:
        arr = np.array(data, dtype=object).flatten().tolist()
    except:
        try:
            arr = list(data)
        except:
            arr = [data]
    return (tuple(t.shape), tuple(sorted(str(x) for x in arr)))

# Build full simplicial object A_•(T)
def build_simplicial_object(T):
    d = min(T.shape) - 1
    visited = {k: {} for k in range(d+2)}
    visited[d][key_tensor(T)] = T
    Q = deque([(T, d)])
    while Q:
        s, k = Q.popleft()
        if k>0:
            for i in range(k+1):
                f = s.face(i)
                fk = min(f.shape)-1
                if fk<0: continue
                kk = key_tensor(f)
                if kk not in visited[fk]:
                    visited[fk][kk] = f; Q.append((f,fk))
        if k<d+1:
            for i in range(k+1):
                g = s.degen(i)
                gk = min(g.shape)-1
                if not(0<=gk<=d+1): continue
                kk = key_tensor(g)
                if kk not in visited[gk]:
                    visited[gk][kk] = g; Q.append((g,gk))
    return {k:list(v.values()) for k,v in visited.items() if v}

# Build horn simplicial object Λ^d_j
def build_horn_object(T,j):
    d = min(T.shape)-1
    facets=[]
    for i in range(d+1):
        if i==j:
            zs = tuple(s-1 for s in T.shape)
            zero = np.zeros(zs,object)
            facets.append(SymbolicTensor(zs,tensor=zero))
        else:
            facets.append(T.face(i))
    visited={k:{} for k in range(d)};Q=deque()
    for f in facets:
        fk=min(f.shape)-1; visited[fk][key_tensor(f)]=f; Q.append((f,fk))
    while Q:
        s,k=Q.popleft()
        if k>0:
            for i in range(k+1):
                f=s.face(i); fk=min(f.shape)-1
                if fk<0: continue
                kk=key_tensor(f)
                if kk not in visited[fk]: visited[fk][kk]=f; Q.append((f,fk))
        if k<d-1:
            for i in range(k+1):
                g=s.degen(i); gk=min(g.shape)-1
                if not(0<=gk<=d-1): continue
                kk=key_tensor(g)
                if kk not in visited[gk]: visited[gk][kk]=g; Q.append((g,gk))
    return {k:list(v.values()) for k,v in visited.items() if v}

# Matrix of face map
def matrix_of_face(Am,Am1,i):
    M=sp.zeros(len(Am1),len(Am))
    lk={key_tensor(x):r for r,x in enumerate(Am1)}
    for j,x in enumerate(Am):
        y=x.face(i); ky=key_tensor(y)
        if ky in lk: M[lk[ky],j]=1
    return M

# Build normalized chain bases N_k
def build_normalized(C,face,d):
    N={}
    C0=C.get(0,[])
    N[0]=[sp.eye(len(C0)).col(i) for i in range(len(C0))]
    for k in range(1,d+2):
        Ck=C.get(k,[])
        if not Ck: N[k]=[]; continue
        mats=[face.get((k,i),sp.zeros(len(C.get(k-1,[])),len(Ck))) for i in range(1,k+1)]
        M=sp.Matrix.vstack(*mats)
        N[k]=M.nullspace()
    return N

# Compute relative homology via mapping cone
def relative_homology_dimension(shape,j):
    T=SymbolicTensor(shape,'range')
    d=min(shape)-1
    if d<0: return 0
    A=build_simplicial_object(T)
    L=build_horn_object(T,j)
    # face maps
    fA={(k,i):matrix_of_face(A.get(k,[]),A.get(k-1,[]),i) for k in A for i in range(k+1)}
    fL={(k,i):matrix_of_face(L.get(k,[]),L.get(k-1,[]),i) for k in L for i in range(k+1)}
    NA=build_normalized(A,fA,d)
    NL=build_normalized(L,fL,d)
    # basis mats
    Bn=sp.Matrix.hstack(*NA[d]) if NA[d] else sp.zeros(len(A.get(d,[])),0)
    Bn1=sp.Matrix.hstack(*NA[d-1]) if NA[d-1] else sp.zeros(len(A.get(d-1,[])),0)
    BL1=sp.Matrix.hstack(*NL[d-1]) if NL[d-1] else sp.zeros(len(L.get(d-1,[])),0)
    BL2=sp.Matrix.hstack(*NL[d-2]) if NL.get(d-2) else sp.zeros(len(L.get(d-2,[])),0)
    # full diffs
    dA=fA.get((d,0),sp.zeros(len(A.get(d-1,[])),len(A.get(d,[]))))
    dL=fL.get((d-1,0),sp.zeros(len(L.get(d-2,[])),len(L.get(d-1,[]))))
    # normalize
    def normalize(D_full,B_src,B_tgt):
        if B_tgt.shape[1] and B_src.shape[1]: return (B_src.T*B_src).inv()*B_src.T*D_full*B_tgt
        return sp.zeros(B_src.shape[1],B_tgt.shape[1])
    dA_n=normalize(dA,Bn1,Bn)
    f_full=sp.zeros(len(A.get(d-1,[])),len(L.get(d-1,[])))
    lkA={key_tensor(x):i for i,x in enumerate(A.get(d-1,[]))}
    for i,x in enumerate(L.get(d-1,[])):
        kx=key_tensor(x)
        if kx in lkA: f_full[lkA[kx],i]=1
    f_n=normalize(f_full,Bn1,BL1)
    dL_n=-normalize(dL,BL2,BL1)
    # mapping cone block
    from sympy import BlockMatrix
    top=BlockMatrix([[dA_n,f_n]]).as_explicit()
    bot=BlockMatrix([[sp.zeros(dL_n.shape[0],dA_n.shape[1]),dL_n]]).as_explicit()
    cone=BlockMatrix([[dA_n,f_n],[sp.zeros(dL_n.shape[0],dA_n.shape[1]),dL_n]]).as_explicit()
    return len(cone.nullspace())

if __name__=='__main__':
    for shape in [(3,3),(4,4),(2,2,2),(3,3,3)]:
        n=min(shape)-1
        print(f"Shape {shape}, n={n}:")
        for j in range(n+1): print(f" H_{n}(Λ^{j}) = {relative_homology_dimension(shape,j)}")
