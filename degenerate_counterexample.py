# separated degenerate_counterexample.py from tensor_ops.py for better readability
# In this example, the hypothesis that the boundary is non-degenerate is violated.
# -*- coding: utf-8 -*-

import numpy as np
from tensor_ops import bdry, is_degen, n_hypergroupoid_comparison, n_hypergroupoid_conjecture, horn, filler

if __name__ == '__main__':

    counterexample = np.array( [[8, 8, 7],
                                [3, 2, 1],
                                [6, 5, 4],
                                [8, 5, 4],
                                [4, 4, 1],
                                [1, 4, 8],
                                [6, 5, 6],
                                [6, 2, 5]] ).transpose()
    print(f"Counterexample with degenerate boundary: {counterexample}")
    print(f"bdry(counterexample): {bdry(counterexample)}")
    print(f"is_degen(bdry(counterexample)): {is_degen(bdry(counterexample))}")
    comparison = n_hypergroupoid_comparison(counterexample, verbose=True)
    conjecture = n_hypergroupoid_conjecture(counterexample.shape, verbose=True)
    print("Conjecture:", conjecture, "Comparison:", comparison)
    h = horn(counterexample, 1)
    print("Horn:", h)
    f = filler(h, 1)
    print("Filler:", f)
    print("Counterexample and filler agree:", np.array_equal(counterexample,f))

