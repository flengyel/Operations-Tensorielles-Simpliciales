import sympy as sp
<<<<<<< HEAD
import numpy as np
from symbolic_tensor_ops import SymbolicTensor
from tensor_ops import bdry, standard_basis_matrix, dimen

class ChainComplex:
    def __init__(self, shape, mod2=False):
        """
        shape: shape of the tensor being tested for difference tensor)
        mod2: whether to compute over Z/2
        """
=======
from symbolic_tensor_ops import SymbolicTensor

class ChainComplex:
    def __init__(self, shape, mod2=False):
>>>>>>> ddc767ecb3cb817612baea2d1f4e82f9752866d5
        self.shape = shape
        self.mod2 = mod2

    def is_cycle(self, tensor):
<<<<<<< HEAD
        """
        Check whether the given tensor is a cycle:
        bdry(tensor) == 0
        """
=======
>>>>>>> ddc767ecb3cb817612baea2d1f4e82f9752866d5
        sym_tensor = SymbolicTensor(tensor.shape, tensor=tensor)
        boundary = sym_tensor.bdry().tensor
        flat = [sp.simplify(boundary[i, j]) for i in range(boundary.shape[0]) for j in range(boundary.shape[1])]
        if self.mod2:
            flat = [c % 2 for c in flat]
        return all(c == 0 for c in flat)

    def is_boundary(self, tensor):
<<<<<<< HEAD
        """
        Check whether the given tensor is a boundary:
        There exists X such that bdry(X) == tensor
        """
        return False  # Placeholder for actual implementation
    
    def __init__(self, shape, mod2=False):
        """
        shape: shape of the tensor being tested 
        mod2: whether to compute over Z/2
        """
        self.shape = shape
        self.mod2 = mod2

    def is_cocycle(self, tensor):
        """
        Check whether tensor defines a cocycle functional:
        tensor applied to any boundary is zero.
        """
        n = dimen(tensor) + 1

        # Build basis of (n x n) tensors
        basis_vectors = []
        for i in range(n):
            for j in range(n):
                basis = standard_basis_matrix(n, n, i, j)
                bdry_basis = bdry(basis)
                vec = sp.Matrix([bdry_basis[a, b] for a in range(n - 1) for b in range(n - 1)])
=======
        n = tensor.shape[0] + 1
        basis_vectors = []
        for i in range(n):
            for j in range(n):
                basis_tensor = sp.zeros(n, n)
                basis_tensor[i, j] = 1
                bdry_tensor = SymbolicTensor((n, n), tensor=basis_tensor).bdry().tensor
                vec = sp.Matrix([bdry_tensor[a, b] for a in range(n - 1) for b in range(n - 1)])
>>>>>>> ddc767ecb3cb817612baea2d1f4e82f9752866d5
                if self.mod2:
                    vec = vec.applyfunc(lambda x: x % 2)
                basis_vectors.append(vec)

<<<<<<< HEAD
        # Flatten input tensor
=======
        B = sp.Matrix.hstack(*basis_vectors)
>>>>>>> ddc767ecb3cb817612baea2d1f4e82f9752866d5
        target_vec = sp.Matrix([tensor[a, b] for a in range(n - 1) for b in range(n - 1)])
        if self.mod2:
            target_vec = target_vec.applyfunc(lambda x: x % 2)

<<<<<<< HEAD
        # Check pairing with each boundary vector
        for v in basis_vectors:
            pairing = sum(a * b for a, b in zip(target_vec, v))
            if self.mod2:
                pairing = pairing % 2
            if pairing != 0:
                return False
        return True

if __name__ == "__main__":
    n = 3
    print("\n=== Manual Example ===")

    # Define a symbolic tensor
    A = SymbolicTensor((n, n))

    # Create the boundary of A and check if it is a cycle (should be True)
    boundary_A = A.bdry().tensor
    chain = ChainComplex(shape=boundary_A.shape)
    print("Boundary of A is a cycle:", chain.is_cycle(boundary_A))

    # Difference of A and its filler
    horn = A.horn(1)
    filler = A.filler(horn, 1)
    diff = A.tensor - filler.tensor

    # Test for cycle and boundary
    print("\n=== Test on A - A' ===")
    print("Is cycle:", chain.is_cycle(diff))
    print("Is boundary:", chain.is_boundary(diff))

    # Cochain test
    cochain = ChainComplex(shape=diff.shape)
    print("\nIs cocycle (dual test):", cochain.is_cocycle(diff))
=======
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

>>>>>>> ddc767ecb3cb817612baea2d1f4e82f9752866d5
