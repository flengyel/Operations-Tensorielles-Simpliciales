# horn_difference_tensor.py

import numpy as np
import sympy as sp
from symbolic_tensor_ops import SymbolicTensor
from tensor_local_coeff_chain_complex import SymbolicLocalChainComplex

# (Re-use your existing wrapper class)
class HornDifferenceTensor(SymbolicTensor):
    def __init__(self, base: SymbolicTensor, missing_index: int):
        self.base = base
        self.shape = base.shape
        self.missing_index = missing_index

    def dimen(self):
        return self.base.dimen()

    def face(self, i: int):
        if i == self.missing_index:
            # build a zero tensor for the missing face
            zero_shape = tuple(dim-1 for dim in self.base.shape)
            Z = np.empty(zero_shape, dtype=object)
            for idx in np.ndindex(zero_shape):
                Z[idx] = sp.S.Zero
            return SymbolicTensor(zero_shape, tensor=Z)
        else:
            return self.base.face(i)

    @property
    def tensor(self):
        return self.base.tensor

def restricted_homology(T: SymbolicTensor, j: int):
    """Compute β-vector for the j-th horn of T, returning zeros if Δ=0."""
    # 1) moore‐filler
    horn = T.horn(j)
    Gp   = T.filler(horn, j)
    # 2) raw difference
    Δ = T - Gp
    # 3) detect zero‐difference
    if all(Δ.tensor[idx] == 0 for idx in np.ndindex(Δ.shape)):
        # Δ=0 ⇒ unique filler ⇒ trivial homology
        d = T.dimen()
        return [0]*(d+1)
    # 4) wrap & dispatch
    hd = HornDifferenceTensor(Δ, missing_index=j)
    try:
        return SymbolicLocalChainComplex([hd]).betti_numbers()
    except ValueError:
        # no non-degenerate generators ⇒ trivial homology
        d = T.dimen()
        return [0]*(d+1)

def horn_difference_example(shape=(4, 4)):
    # Instantiate a (4,4) symbolic tensor
    T = SymbolicTensor(shape)
    d = T.dimen()  # should be 3
    print(f"Testing restricted homology on every horn of a {shape} tensor (d={d})\n")
    # Compute the restricted homology for each horn
    for j in range(d + 1):
        print(f"horn j={j}: β = {restricted_homology(T, j)}")

if __name__ == "__main__":
    horn_difference_example(shape=(3, 3))  # Example with a (4,4) tensor
    horn_difference_example(shape=(4, 4))  # Example with a (4,4) tensor
    horn_difference_example(shape=(5, 5))  # Example with a (5,5) tensor

    # instantiate a (4,4,4) symbolic tensor (nontrivial homology!)
    horn_difference_example(shape=(4,4,4))
    horn_difference_example(shape=(5,5,5,5))
    # instantiate a (6,6,6,6,6) symbolic tensor
    horn_difference_example(shape=(6,6,6,6,6))
    