import numpy as np
from typing import List, Tuple, Optional
from tensor_ops import face, degen as num_degen
from numpy.linalg import matrix_rank
from sympy import Matrix
from symbolic_tensor_ops import SymbolicTensor


def is_degen_tensor(a: np.ndarray) -> bool:
    """
    Numeric degeneracy test:
    - d = min(a.shape) - 1 (simplicial dimension).
    - If d <= 0: never degenerate.
    - If d == 1: degenerate iff all entries equal.
    - If d > 1: degenerate if a == num_degen(face(a, i), i) for any 0 <= i <= d.
    """
    d = min(a.shape) - 1
    if d <= 0:
        return False
    flat = a.flatten()
    if d == 1:
        return bool(np.all(flat == flat[0]))
    for i in range(d + 1):
        try:
            B = face(a, i)
            C = num_degen(B, i)
        except Exception:
            continue
        if np.array_equal(a, C):
            return True
    return False


def is_degen_symbolic(T: SymbolicTensor) -> bool:
    """
    Symbolic degeneracy test:
    - d = min(T.shape) - 1 (simplicial dimension).
    - If d <= 0: never degenerate.
    - If d == 1: degenerate iff all entries equal.
    - If d > 1: degenerate if face+degen restores T for any 0 <= i <= d.
    """
    d = min(T.shape) - 1
    if d <= 0:
        return False
    entries = list(T.tensor.flatten())
    if d == 1:
        return all(entry == entries[0] for entry in entries)
    for i in range(d + 1):
        try:
            F = T.face(i)
            D = F.degen(i)
        except Exception:
            continue
        if D.tensor.tolist() == T.tensor.tolist():
            return True
    return False


class NumericLocalChainComplex:
    """
    Build the normalized simplicial chain complex for a numeric tensor (local coefficients),
    computing chain groups C_k by greedy independent face enumeration and degeneracy filtering,
    then assembling boundary maps ∂_k.
    """
    def __init__(self, data: np.ndarray):
        self.data = data
        d = min(data.shape) - 1  # simplicial dimension
        # Build chain groups from top dimension d down to 0
        gens: List[List[np.ndarray]] = [[data]]
        for k in range(d, 0, -1):
            faces: List[np.ndarray] = []
            for T in gens[-1]:
                for i in range(k + 1):
                    faces.append(face(T, i))
            # Greedy linear-independence reduction
            independent: List[np.ndarray] = []
            mats = None
            for F in faces:
                vec = F.flatten().astype(float)
                if mats is None:
                    independent.append(F)
                    mats = vec[:, None]
                else:
                    M = np.hstack([mats, vec[:, None]])
                    if matrix_rank(M) > matrix_rank(mats):
                        independent.append(F)
                        mats = M
            # Remove degeneracies
            layer = [F for F in independent if not is_degen_tensor(F)]
            gens.append(layer)
        # Reverse so that C_0 is first
        self.generators = list(reversed(gens))
        # Assemble boundary matrices ∂_k: C_k → C_{k-1}
        self.boundaries: List[np.ndarray] = []
        for k in range(1, len(self.generators)):
            Ck = self.generators[k]
            Ckm1 = self.generators[k - 1]
            total_rows = sum(U.size for U in Ckm1)
            B = np.zeros((total_rows, len(Ck)), dtype=int)
            for j, T in enumerate(Ck):
                offset = 0
                for U in Ckm1:
                    for i in range(min(T.shape)):
                        F = face(T, i)
                        if np.array_equal(F, U):
                            sign = 1 if i % 2 == 0 else -1
                            B[offset:offset + U.size, j] = sign * F.flatten()
                            break
                    offset += U.size
            self.boundaries.append(B)

    def betti_numbers(self) -> List[int]:
        ranks = [matrix_rank(B) for B in self.boundaries]
        dims = [len(layer) for layer in self.generators]
        betti: List[int] = []
        # β_0
        betti.append(dims[0] - (ranks[0] if ranks else 0))
        # β_k for 1 ≤ k ≤ n-1
        for k in range(1, len(dims) - 1):
            betti.append(dims[k] - ranks[k - 1] - ranks[k])
        # β_n
        betti.append(dims[-1] - (ranks[-1] if ranks else 0))
        return betti


class SymbolicLocalChainComplex:
    """
    Build the normalized simplicial chain complex for a SymbolicTensor (local coefficients),
    mirroring the numeric workflow using Sympy matrices.
    """
    def __init__(self, tensor: SymbolicTensor):
        self.tensor = tensor
        d = min(tensor.shape) - 1
        gens: List[List[SymbolicTensor]] = [[tensor]]
        for k in range(d, 0, -1):
            faces: List[SymbolicTensor] = []
            for T in gens[-1]:
                for i in range(k + 1):
                    faces.append(T.face(i))
            independent: List[SymbolicTensor] = []
            M: Optional[Matrix] = None
            for F in faces:
                vec = Matrix(F.tensor.reshape(-1, 1))
                if M is None:
                    independent.append(F)
                    M = vec
                else:
                    N = M.row_join(vec)
                    if N.rank() > M.rank():
                        independent.append(F)
                        M = N
            layer = [F for F in independent if not is_degen_symbolic(F)]
            gens.append(layer)
        self.generators = list(reversed(gens))
        self.boundaries: List[Matrix] = []
        for k in range(1, len(self.generators)):
            Ck = self.generators[k]
            Ckm1 = self.generators[k - 1]
            total_rows = sum(U.tensor.size for U in Ckm1)
            B = Matrix.zeros(total_rows, len(Ck))
            for j, T in enumerate(Ck):
                offset = 0
                for U in Ckm1:
                    for i in range(min(T.shape)):
                        F = T.face(i)
                        if F.tensor.tolist() == U.tensor.tolist():
                            sign = 1 if i % 2 == 0 else -1
                            flat = Matrix(F.tensor.reshape(-1, 1))
                            B[offset:offset + U.tensor.size, j] = sign * flat
                            break
                    offset += U.tensor.size
            self.boundaries.append(B)

    def betti_numbers(self, mod: Optional[int] = None) -> List[int]:
        ranks: List[int] = []
        for B in self.boundaries:
            M = B if mod is None else B.applyfunc(lambda x: x % mod)
            ranks.append(int(M.rank()))
        dims = [len(layer) for layer in self.generators]
        betti: List[int] = []
        betti.append(dims[0] - (ranks[0] if ranks else 0))
        for k in range(1, len(dims) - 1):
            betti.append(dims[k] - ranks[k - 1] - ranks[k])
        betti.append(dims[-1] - (ranks[-1] if ranks else 0))
        return betti


def adj_from_edges(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    A = np.zeros((n, n), dtype=int)
    for u, v in edges:
        A[u, v] = 1
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
        A = adj_from_edges(n, edges)
        cl = NumericLocalChainComplex(A)
        print(f"{name}: {cl.betti_numbers()}")
    print("Symbolic results:")
    for shape in [(2, 2), (3, 3, 3), (4, 4, 4, 4)]:
        T = SymbolicTensor(shape)
        cl = SymbolicLocalChainComplex(T)
        print(f"symbolic_{shape}: {cl.betti_numbers()}")
