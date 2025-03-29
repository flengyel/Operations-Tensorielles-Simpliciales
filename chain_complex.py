import sympy as sp
from symbolic_tensor_ops import SymbolicTensor

class ChainComplex:
    def __init__(self, shape, mod2=False):
        self.shape = shape
        self.mod2 = mod2

    def is_cycle(self, tensor):
        sym_tensor = SymbolicTensor(tensor.shape, tensor=tensor)
        boundary = sym_tensor.bdry().tensor
        flat = [sp.simplify(boundary[i, j]) for i in range(boundary.shape[0]) for j in range(boundary.shape[1])]
        if self.mod2:
            flat = [c % 2 for c in flat]
        return all(c == 0 for c in flat)

    def is_boundary(self, tensor):
        n = tensor.shape[0] + 1
        basis_vectors = []
        for i in range(n):
            for j in range(n):
                basis_tensor = sp.zeros(n, n)
                basis_tensor[i, j] = 1
                bdry_tensor = SymbolicTensor((n, n), tensor=basis_tensor).bdry().tensor
                vec = sp.Matrix([bdry_tensor[a, b] for a in range(n - 1) for b in range(n - 1)])
                if self.mod2:
                    vec = vec.applyfunc(lambda x: x % 2)
                basis_vectors.append(vec)

        B = sp.Matrix.hstack(*basis_vectors)
        target_vec = sp.Matrix([tensor[a, b] for a in range(n - 1) for b in range(n - 1)])
        if self.mod2:
            target_vec = target_vec.applyfunc(lambda x: x % 2)

        try:
            sol, params = B.gauss_jordan_solve(target_vec)
            residual = B @ sol - target_vec
            return all(r.is_zero for r in residual)
        except Exception:
            return False

if __name__ == "__main__":
    for n in range(3,5):
        A = SymbolicTensor((n, n))

        horn = A.horn(1)
        filler = A.filler(horn, 1)

        bdry_A = A.bdry().tensor
        bdry_filler = filler.bdry().tensor

        diff = bdry_A - bdry_filler

        complex = ChainComplex(shape=(n - 1, n - 1))
        print("n = 4 test")
        print("Is cycle:", complex.is_cycle(diff))
        print("Is boundary:", complex.is_boundary(diff))

