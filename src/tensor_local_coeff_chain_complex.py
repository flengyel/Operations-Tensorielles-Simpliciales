"""
Tensor Local Coeff Chain Complex

This module computes normalized simplicial homology with *local tensor coefficients*.
Each simplex is assigned a genuine tensor (numeric or symbolic) rather than a formal symbol,
allowing the homology to detect relations in the tensor data itself. In local tensor coefficient homology,
each simplex is its own coefficient tensor.

 Chain groups:
      C_k = ⊕_{T ∈ generators[k]} R·T
      In particular, if generators[k] is empty then C_k = 0 (the zero module).

    Boundaries ∂_k : C_k → C_{k−1} are zero when C_k=0, and otherwise
    assembled by summing signed face‐tensors.
"""
import numpy as np
from typing import List, Tuple, Optional
from tensor_ops import face, is_generator_numeric, dimen
from numpy.linalg import matrix_rank
from sympy import Matrix, Symbol
from symbolic_tensor_ops import SymbolicTensor, is_generator_symbolic

class NumericLocalChainComplex:
    def __init__(self, data: List[np.ndarray]):
        if not data:
            raise ValueError("Must provide at least one top-dimensional tensor")
        shapes = {T.shape for T in data}
        if len(shapes) != 1:
            raise ValueError(f"All top-dimensional tensors must share the same shape, got {shapes}")
        # algebraic dimension from shared dimen()
        d = dimen(data[0])

        # gens[k] holds degree-k generators
        gens: List[List[np.ndarray]] = [[] for _ in range(d + 1)]
        gens[d] = data[:]

        # build lower-degree generators by independent numeric faces
        for k in range(d, 0, -1):
            faces: List[np.ndarray] = []
            for T in gens[k]:
                ℓ = dimen(T)
                for i in range(ℓ + 1):
                    F = face(T, i)
                    faces.append(F)

            # filter out zero/degenerate
            faces = [F for F in faces if is_generator_numeric(F)]
            # drop exact duplicates
            unique_faces: List[np.ndarray] = []
            seen = set()
            for F in faces:
                key = tuple(F.flatten().tolist())
                if key not in seen:
                    seen.add(key)
                    unique_faces.append(F)
            faces = unique_faces

            # keep only linearly independent faces
            independent: List[np.ndarray] = []
            mats = None
            for F in faces:
                vec = F.flatten().astype(float)
                if mats is None or matrix_rank(np.hstack([mats, vec[:, None]])) > matrix_rank(mats):
                    independent.append(F)
                    mats = vec[:, None] if mats is None else np.hstack([mats, vec[:, None]])
            gens[k - 1] = independent

        self.generators = gens
        self.boundaries: List[np.ndarray] = []

        # assemble boundary matrices ∂: C_k → C_{k-1}
        for k in range(1, d + 1):
            Ck = self.generators[k]
            Ckm1 = self.generators[k - 1]
            rows = sum(U.size for U in Ckm1)
            cols = len(Ck)
            B = np.zeros((rows, cols), dtype=int)
            offs = [0] + [sum(U.size for U in Ckm1[:i + 1]) for i in range(len(Ckm1) - 1)]

            for j, T in enumerate(Ck):
                ℓ = dimen(T)
                for i in range(ℓ + 1):
                    F = face(T, i)
                    sign = 1 if i % 2 == 0 else -1
                    for m, U in enumerate(Ckm1):
                        if np.array_equal(F, U):
                            start = offs[m]
                            B[start:start + U.size, j] = sign * F.flatten()
                            break
            self.boundaries.append(B)

    def betti_numbers(self) -> List[int]:
        # dims[k] = dim C_k
        dims = [len(layer) for layer in self.generators]
        # short‐circuit: only C_0
        if not self.boundaries:
            return [dims[0] if dims[0] > 0 else 0]

        # compute boundary ranks, treating empty boundaries as rank 0
        ranks = [0 if B.size == 0 else matrix_rank(B) for B in self.boundaries]

        bettis: List[int] = []
        for k in range(len(dims)):
            d_k = dims[k]
            if d_k == 0:
                # zero module ⇒ zero homology
                betti = 0
            else:
                # rank of ∂_k: that's ranks[k-1], except for k=0
                r_prev = ranks[k-1] if k-1 >= 0 else 0
                # rank of ∂_{k+1}: that's ranks[k], except at top
                r_next = ranks[k]   if k   < len(ranks) else 0
                betti = d_k - r_prev - r_next
                # clamp to ≥0
                if betti < 0:
                    betti = 0
            bettis.append(betti)
        return bettis

class SymbolicLocalChainComplex:
    def __init__(self, data: List[SymbolicTensor]):
        data = [T for T in data if is_generator_symbolic(T)]
        if not data:
            raise ValueError("Must provide at least one non-degenerate, nonzero top level symbolic tensor")
        shapes = {T.shape for T in data}
        if len(shapes) != 1:
            raise ValueError(f"All top-level symbolic tensors must share the same shape, got {shapes}")
        # algebraic dimension from tensor's own method
        d = data[0].dimen()

        gens: List[List[SymbolicTensor]] = [[] for _ in range(d + 1)]
        gens[d] = data[:]

        # build lower-degree generators by unique symbolic faces
        for k in range(d, 0, -1):
            faces: List[SymbolicTensor] = []
            for T in gens[k]:
                ℓ = T.dimen()
                for i in range(ℓ + 1):
                    faces.append(T.face(i))

            # filter out zero/degenerate
            faces = [F for F in faces if is_generator_symbolic(F)]
            # drop exact duplicates
            unique: List[SymbolicTensor] = []
            seen = set()
            for F in faces:
                flat = tuple(F.tensor.reshape(-1))
                if flat not in seen:
                    seen.add(flat)
                    unique.append(F)
            gens[k - 1] = unique

        self.generators = gens
        self.boundaries: List[Matrix] = []

        # assemble boundary matrices ∂: C_k → C_{k-1}
        for k in range(1, d + 1):
            Ck = self.generators[k]
            Ckm1 = self.generators[k - 1]
            rows = sum(U.tensor.size for U in Ckm1)
            cols = len(Ck)
            B = Matrix.zeros(rows, cols)
            offs = [0] + [sum(U.tensor.size for U in Ckm1[:i + 1]) for i in range(len(Ckm1) - 1)]

            for j, T in enumerate(Ck):
                ℓ = T.dimen()
                for i in range(ℓ + 1):
                    F = T.face(i)
                    sign = 1 if i % 2 == 0 else -1
                    flat_vals = list(F.tensor.reshape(-1))
                    for m, U in enumerate(Ckm1):
                        if (F.tensor == U.tensor).all():
                            start = offs[m]
                            for idx, val in enumerate(flat_vals):
                                B[start + idx, j] = sign * val
                            break
            self.boundaries.append(B)

    def betti_numbers(self, mod: Optional[int] = None) -> List[int]:
        # dims[k] = dim C_k
        dims = [len(layer) for layer in self.generators]
        # short‐circuit: only C_0
        if not self.boundaries:
            return [dims[0] if dims[0] > 0 else 0]

        # compute boundary ranks over Z or mod p, empty matrices → rank 0
        ranks: List[int] = []
        for B in self.boundaries:
            Bm = B if mod is None else B.applyfunc(lambda x: x % mod)
            ranks.append(int(Bm.rank()))

        bettis: List[int] = []
        for k in range(len(dims)):
            d_k = dims[k]
            if d_k == 0:
                betti = 0
            else:
                r_prev = ranks[k-1] if k-1 >= 0 else 0
                r_next = ranks[k]   if k   < len(ranks) else 0
                betti = d_k - r_prev - r_next
                if betti < 0:
                    betti = 0
            bettis.append(betti)
        return bettis

import numpy as np
import sympy as sp
from symbolic_tensor_ops import SymbolicTensor, is_generator_symbolic

class _HornDifferenceTensor(SymbolicTensor):
    """
    A wrapper around a base tensor Δ so that Δ.face(j) always returns zero,
    but for i!=j delegates to the real Δ.face(i).
    """
    def __init__(self, base: SymbolicTensor, missing_index: int):
        # we pretend shape is the same
        self.base = base
        self.shape = base.shape
        self.missing_index = missing_index

    def dimen(self):
        return self.base.dimen()

    def face(self, i: int):
        if i == self.missing_index:
            # build a zero tensor of the right “face‐shape”
            zero_shape = tuple(dim-1 for dim in self.base.shape)
            Z = np.empty(zero_shape, dtype=object)
            for idx in np.ndindex(zero_shape):
                Z[idx] = sp.S.Zero
            return SymbolicTensor(zero_shape, tensor=Z)
        else:
            return self.base.face(i)

    @property
    def tensor(self):
        # if anyone inspects tensor directly, get the true array
        return self.base.tensor



if __name__ == "__main__":
    import numpy as np
    from symbolic_tensor_ops import SymbolicTensor

    # four distinct 0‑simplices
    points = [np.array([[i]]) for i in (1, 2, 3, 4)]
    cl = NumericLocalChainComplex(points)
    print(f"Four disconnected points: {cl.betti_numbers()}")

    print("Symbolic results:")
    for shape in [(2,2), (3,3,3)]:
        t = SymbolicTensor(shape)
        print(f"symbolic tensor of shape {shape}: {t}")
        cl_sym = SymbolicLocalChainComplex([t])
        print(f"Betti numbers: {cl_sym.betti_numbers()}")

    print("Custom test - 0‑simplex symbolic tensor of shape (2,3):")
    ts0 = SymbolicTensor((2,3))
    print(f"symbolic tensor of shape {ts0.shape}: {ts0}")
    cl0 = SymbolicLocalChainComplex([ts0])
    print(f"Betti numbers: {cl0.betti_numbers()}")

    import numpy as np
    import networkx as nx

    # 1) Build the Petersen graph adjacency
    G = nx.petersen_graph()
    A = nx.to_numpy_array(G, dtype=int)

    # 2) Compute homology of A itself
    clA = NumericLocalChainComplex([A])
    print("Adjacency‐tensor Betti numbers:", clA.betti_numbers())

    # 3) Explicitly look at its principal‐minor “boundary” faces
    #    (removing row&col i for i=0…n-1)
    minors = [np.delete(np.delete(A, i, axis=0), i, axis=1)
            for i in range(A.shape[0])]
    clMin = NumericLocalChainComplex(minors)
    print("Principal‐minor Betti numbers:", clMin.betti_numbers())

    # tensor local coefficient graph homology
    # 1) Build the Petersen graph
    G = nx.petersen_graph()
    n = G.number_of_nodes()

    # 2) Turn each edge into a 1‑simplex tensor of shape (2, n)
    top_tensors = []
    for u, v in G.edges():
        T = np.zeros((2, n), dtype=int)
        T[0, u] = 1   # “source” row
        T[1, v] = 1   # “target” row
        top_tensors.append(T)

    # 3) Compute tensor‑local‑coefficient homology
    cl = NumericLocalChainComplex(top_tensors)
    print("Petersen edge‑tensor Betti numbers in matrix local coefficient homology:", cl.betti_numbers())

    from sympy import symbols, Matrix
    from symbolic_tensor_ops import SymbolicTensor
    import numpy as np

    t = SymbolicTensor((3,3))
    inner_horn = t.horn(1)
    print("Inner horn tensor list:")
    for fc in inner_horn:
        print(fc.tensor)
    chain_complex = SymbolicLocalChainComplex(inner_horn)
    print("Inner horn Betti numbers:", chain_complex.betti_numbers())
    

    # 1) build T and G′ as before
    T = SymbolicTensor((3,3))
    horn = T.horn(1)
    G_prime = T.filler(horn, 1)

    # 2) form the raw difference Δ = T – G′
    diff = T - G_prime

    # 3) wrap it so face(1) always returns 0
    hd = _HornDifferenceTensor(diff, missing_index=1)

    # 4) feed that single “fake” tensor into your existing homology engine
    cl = SymbolicLocalChainComplex([hd])
    print("Restricted‐boundary Betti numbers:", cl.betti_numbers())
    # → [0, 0, 1]
