import sympy as sp
import numpy as np
from symbolic_tensor_ops import SymbolicTensor
from tensor_ops import bdry, standard_basis_matrix, dimen

class ChainComplex:
    def __init__(self, shape, mod2=False):
        """
        shape: shape of the tensor being tested for difference tensor)
        mod2: whether to compute over Z/2
        """
        self.shape = shape
        self.mod2 = mod2

    def is_cycle(self, tensor):
        """
        Check whether the given tensor is a cycle:
        bdry(tensor) == 0
        """
        sym_tensor = SymbolicTensor(tensor.shape, tensor=tensor)
        boundary = sym_tensor.bdry().tensor
        flat = [sp.simplify(boundary[i, j]) for i in range(boundary.shape[0]) for j in range(boundary.shape[1])]
        if self.mod2:
            flat = [c % 2 for c in flat]
        return all(c == 0 for c in flat)

    def is_boundary(self, tensor):
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
                if self.mod2:
                    vec = vec.applyfunc(lambda x: x % 2)
                basis_vectors.append(vec)

        # Flatten input tensor
        target_vec = sp.Matrix([tensor[a, b] for a in range(n - 1) for b in range(n - 1)])
        if self.mod2:
            target_vec = target_vec.applyfunc(lambda x: x % 2)

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
