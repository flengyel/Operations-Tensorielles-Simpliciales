from typing import Tuple # , List, Union, Any
from operations_tensorielles_simpliciales import tensor_inner_horn_rank_dimension_conjecture
from operations_tensorielles_simpliciales import tensor_inner_horn_rank_dimension_comparison
from operations_tensorielles_simpliciales import horn, filler, bdry
import numpy as np
import random

def random_shape(n: int) -> Tuple[int]:
    length = random.randint(2, n/2)  # Length of at least two and bounded by 10
    return tuple(random.randint(2, n) for _ in range(length))  # Positive integers at least two and bounded by n

def test_rank_dim_conjecture(tests: int, maxdim:int) -> None:    
    non_unique_horns = 0
    unique_horns = 0
    for i in range(tests):
        shape = random_shape(maxdim)
        print(i+1 ,": Shape = ", shape)
        # create a random non-zero tensor of the given shape
        A = np.random.randint(low=1, high=10, size=shape, dtype=np.int16)
        # check if the boundary of A is a degeneracy
        comparison = tensor_inner_horn_rank_dimension_comparison(A, verbose=True)
        conjecture = tensor_inner_horn_rank_dimension_conjecture(shape, verbose=True)
        if comparison != conjecture:
            print(f"Counterexample of shape: {A.shape} with tensor: {A}")
            print(f"Conjecture: {conjecture} Comparison: {comparison}")
            print(f"bdry(A): {bdry(A)}")
            print(f"matrix rank of boundary: {np.linalg.matrix_rank(bdry(A))}")
            exit(0)
        if comparison:
            unique_horns += 1
        else:
            non_unique_horns += 1
    print(f"Conjecture verified for {tests} random shapes. Unique horns: {unique_horns} non-unique horns: {non_unique_horns}")

if __name__ == "__main__":
    test_rank_dim_conjecture(500, 4) 
    counterexample = np.array( [[8, 8, 7],
                                [3, 2, 1],
                                [6, 5, 4],
                                [8, 5, 4],
                                [4, 4, 1],
                                [1, 4, 8],
                                [6, 5, 6],
                                [6, 2, 5]] ).transpose()
    print(f"Counterexample: {counterexample}")
    print(f"bdry(counterexample): {bdry(counterexample)}")
    print(f"matrix rank of boundary: {np.linalg.matrix_rank(bdry(counterexample))}")     
    comparison = tensor_inner_horn_rank_dimension_comparison(counterexample, verbose=True)
    conjecture = tensor_inner_horn_rank_dimension_conjecture(counterexample.shape, verbose=True)
    print("Conjecture:", conjecture, "Comparison:", comparison)
    h = horn(counterexample, 1)
    print("Horn:", h)
    f = filler(h, 1)
    print("Filler:", f)
    print("Counterexample and filler agree:", np.array_equal(counterexample,f))
    counterexample2 = np.array([[6, 4, 7, 2, 4, 7, 5, 6],
                                [4, 3, 6, 3, 9, 5, 3, 4],
                                [6, 5, 7, 7, 1, 8, 4, 9]])
    print("Counterexample:", counterexample2)
    print("bdry(counterexample):", bdry(counterexample2))
    print("matrix rank of boundary:", np.linalg.matrix_rank(bdry(counterexample2)))
    comparison = tensor_inner_horn_rank_dimension_comparison(counterexample2, verbose=True)
    conjecture = tensor_inner_horn_rank_dimension_conjecture(counterexample2.shape, verbose=True)
    print("Conjecture:", conjecture, "Comparison:", comparison)
    h = horn(counterexample2, 1)
    print("Horn:", h)
    f = filler(h, 1)
    print("Filler:", f)
    print("Counterexample and filler agree:", np.array_equal(counterexample2,f))
    