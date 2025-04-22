"""
Tensor Local Coeff Chain Complex

This module computes normalized simplicial homology with *local tensor coefficients*.
Each simplex is assigned a genuine tensor (numeric or symbolic) rather than a formal symbol,
allowing the homology to detect relations in the tensor data itself.
"""
import numpy as np
from typing import List, Tuple, Optional
from tensor_ops import face, is_generator_numeric
from numpy.linalg import matrix_rank
from sympy import Matrix
from symbolic_tensor_ops import SymbolicTensor, is_generator_symbolic

class NumericLocalChainComplex:
    def __init__(self, data: np.ndarray):
        d = min(data.shape) - 1
        print(f"[DEBUG] Numeric: simplex dimension d={d}")
        gens: List[List[np.ndarray]] = [[] for _ in range(d+1)]
        gens[d] = [data]
        for k in range(d, 0, -1):
            print(f"[DEBUG] Numeric: building C_{k-1} from C_{k}")
            faces: List[np.ndarray] = []
            for T in gens[k]:
                for i in range(k+1):
                    faces.append(face(T, i))
            print(f"[DEBUG] Numeric: faces count (raw)={len(faces)}")
            faces = [F for F in faces if is_generator_numeric(F)]
            print(f"[DEBUG] Numeric: faces count (filtered)={len(faces)}")
            independent: List[np.ndarray] = []
            mats = None
            for F in faces:
                vec = F.flatten().astype(float)
                if mats is None:
                    independent.append(F)
                    mats = vec[:, None]
                else:
                    try:
                        M_new = np.hstack([mats, vec[:, None]])
                    except ValueError:
                        continue
                    if matrix_rank(M_new) > matrix_rank(mats):
                        independent.append(F)
                        mats = M_new
            print(f"[DEBUG] Numeric: independent count={len(independent)}")
            gens[k-1] = independent
        self.generators = gens
        self.boundaries: List[np.ndarray] = []
        for k in range(1, d+1):
            print(f"[DEBUG] Numeric: assemble boundary ∂_{k}: C_{k}→C_{k-1}")
            Ck, Ck1 = self.generators[k], self.generators[k-1]
            rows = sum(U.size for U in Ck1)
            cols = len(Ck)
            B = np.zeros((rows, cols), dtype=int)
            offs = [0] + [sum(U.size for U in Ck1[:i+1]) for i in range(len(Ck1)-1)]
            for j, T in enumerate(Ck):
                for i in range(k+1):
                    F = face(T, i)
                    sign = 1 if i % 2 == 0 else -1
                    for m, U in enumerate(Ck1):
                        if np.array_equal(F, U):
                            start = offs[m]
                            B[start:start+U.size, j] = sign * F.flatten()
                            break
            print(f"[DEBUG] Numeric: boundary ∂_{k} shape={B.shape}\n{B}")
            self.boundaries.append(B)

    def betti_numbers(self) -> List[int]:
        dims = [len(layer) for layer in self.generators]
        print(f"[DEBUG] betti_numbers: dims={dims}")
        if len(dims) == 1:
            print(f"[DEBUG] betti_numbers: no boundaries, returning [dims[0]]={[dims[0]]}")
            return [dims[0]]
        ranks = [0 if B.size == 0 else matrix_rank(B) for B in self.boundaries]
        print(f"[DEBUG] betti_numbers: ranks={ranks}")
        bettis: List[int] = []
        bettis.append(dims[0] - ranks[0])
        for k in range(1, len(dims)-1):
            bettis.append(dims[k] - ranks[k-1] - ranks[k])
        bettis.append(dims[-1] - ranks[-1])
        print(f"[DEBUG] betti_numbers: bettis={bettis}")
        return bettis

class SymbolicLocalChainComplex:
    def __init__(self, tensor: SymbolicTensor):
        d = min(tensor.shape) - 1
        print(f"[DEBUG] Symbolic: simplex dimension d={d}")
        gens: List[List[SymbolicTensor]] = [[] for _ in range(d+1)]
        gens[d] = [tensor]
        for k in range(d, 0, -1):
            print(f"[DEBUG] Symbolic: building C_{k-1} from C_{k}")
            faces: List[SymbolicTensor] = []
            for T in gens[k]:
                for i in range(k+1):
                    faces.append(T.face(i))
            print(f"[DEBUG] Symbolic: faces count (raw)={len(faces)}")
            faces = [F for F in faces if is_generator_symbolic(F)]
            print(f"[DEBUG] Symbolic: faces count (filtered)={len(faces)}")
            independent: List[SymbolicTensor] = []
            M: Optional[Matrix] = None
            for F in faces:
                vec = Matrix(F.tensor.reshape(-1,1))
                if M is None:
                    independent.append(F)
                    M = vec
                else:
                    M_new = M.row_join(vec)
                    if M_new.rank() > M.rank():
                        independent.append(F)
                        M = M_new
            print(f"[DEBUG] Symbolic: independent count={len(independent)}")
            gens[k-1] = independent
        self.generators = gens
        self.boundaries: List[Matrix] = []
        for k in range(1, d+1):
            print(f"[DEBUG] Symbolic: assemble boundary ∂_{k}: C_{k}→C_{k-1}")
            Ck, Ck1 = self.generators[k], self.generators[k-1]
            rows = sum(U.tensor.size for U in Ck1)
            cols = len(Ck)
            B = Matrix.zeros(rows, cols)
            offs = [0] + [sum(U.tensor.size for U in Ck1[:i+1]) for i in range(len(Ck1)-1)]
            for j, T in enumerate(Ck):
                for i in range(k+1):
                    F = T.face(i)
                    sign = 1 if i % 2 == 0 else -1
                    for m, U in enumerate(Ck1):
                        if F.tensor.tolist() == U.tensor.tolist():
                            start = offs[m]
                            B[start:start+U.tensor.size, j] = sign * Matrix(F.tensor.reshape(-1,1))
                            break
            print(f"[DEBUG] Symbolic: boundary ∂_{k} shape=({B.rows},{B.cols})\n{B}")
            self.boundaries.append(B)

    def betti_numbers(self, mod: Optional[int] = None) -> List[int]:
        dims = [len(layer) for layer in self.generators]
        print(f"[DEBUG] betti_numbers (symbolic): dims={dims}")
        if len(dims) == 1:
            print(f"[DEBUG] betti_numbers (symbolic): no boundaries, returning [dims[0]]={[dims[0]]}")
            return [dims[0]]
        ranks: List[int] = [int((B if mod is None else B.applyfunc(lambda x: x % mod)).rank()) for B in self.boundaries]
        print(f"[DEBUG] betti_numbers (symbolic): ranks={ranks}")
        bettis: List[int] = [dims[0] - ranks[0]]
        for k in range(1, len(dims)-1):
            bettis.append(dims[k] - ranks[k-1] - ranks[k])
        bettis.append(dims[-1] - ranks[-1])
        print(f"[DEBUG] betti_numbers (symbolic): bettis={bettis}")
        return bettis


def adj_from_edges(n: int, edges: List[Tuple[int,int]]) -> np.ndarray:
    A = np.zeros((n,n), dtype=int)
    for u,v in edges:
        A[u,v] = 1
    return A

if __name__ == "__main__":
    examples = {
        "single_edge": (2, [(0, 1)]),
        "directed_3_cycle": (3, [(0, 1), (1, 2), (2, 0)]),
        "transitive_tournament_3": (3, [(0, 1), (0, 2), (1, 2)]),
        "directed_path_4": (4, [(0, 1), (1, 2), (2, 3)]),
        "complete_digraph_3": (3, [(i, j) for i in range(3) for j in range(3) if i != j]),
        "triangle_plus_chord": (3, [(0, 1), (1, 2), (2, 0), (0, 2)])
    }
    print("Numeric results:")
    for name, (n, edges) in examples.items():
        cl = NumericLocalChainComplex(adj_from_edges(n, edges))
        print(f"{name}: {cl.betti_numbers()}")
    # Additional test: a single 0-simplex with non-zero tensor
    print("Custom test - 0-simplex with unit tensor [1]:")
    cl0 = NumericLocalChainComplex(np.array([[1]]))
    print(f"scalar_unit: {cl0.betti_numbers()}")

    print("Symbolic results:")
    for shape in [(2,2), (3,3,3), (4,4,4,4)]:
        cl = SymbolicLocalChainComplex(SymbolicTensor(shape))
        print(f"symbolic_{shape}: {cl.betti_numbers()}")
    # Symbolic test: 0-simplex with symbolic unit tensor
    print("Custom test - 0-simplex symbolic unit tensor:")
    cs0 = SymbolicLocalChainComplex(SymbolicTensor((1,1)))
    print(f"symbolic_scalar_unit: {cs0.betti_numbers()}")
