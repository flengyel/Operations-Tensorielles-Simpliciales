from typing import Tuple # , List, Union, Any
from operations_tensorielles_simpliciales import tensor_inner_horn_rank_dimension_conjecture
from operations_tensorielles_simpliciales import tensor_inner_horn_rank_dimension_comparison
from operations_tensorielles_simpliciales import horn, filler, bdry, degen, isDegeneracy    
import numpy as np
import random

def random_shape(n: int) -> Tuple[int]:
    length = random.randint(2, n/2)  # Length of at least two and bounded by 10
    return tuple(random.randint(2, n) for _ in range(length))  # Positive integers at least two and bounded by n


def rank_dim_conjecture(tests: int, maxdim:int, force_degeneracy:bool=False, skip_degeneracies:bool=False) -> bool:    
    non_unique_horns = 0
    unique_horns = 0
    for i in range(tests):
        shape = random_shape(maxdim)
        A = np.random.randint(low=1, high=10, size=shape, dtype=np.int16)
        if force_degeneracy:
            A = degen(A, 0) # force A to be a degeneracy
        if skip_degeneracies:
            while isDegeneracy(A) or isDegeneracy(bdry(A)):
                A = np.random.randint(low=1, high=10, size=shape, dtype=np.int16)    
        print(i+1 ,": Shape = ", shape)
        comparison = tensor_inner_horn_rank_dimension_comparison(A, verbose=True)
        conjecture = tensor_inner_horn_rank_dimension_conjecture(shape, verbose=True)
        if comparison != conjecture:
            print(f"Counterexample of shape: {A.shape} with tensor: {A}")
            print(f"Conjecture: {conjecture} Comparison: {comparison}")
            print(f"A is a degeneracy: {isDegeneracy(A)}")
            print(f"bdry(A): {bdry(A)}")
            print(f"boundary is a degeneracy: {isDegeneracy(bdry(A))}")
            return False
        if comparison:
            unique_horns += 1
        else:
            non_unique_horns += 1
    print(f"Conjecture verified for {tests} random shapes. Unique horns: {unique_horns} non-unique horns: {non_unique_horns}")
    return True

if __name__ == "__main__":
    rank_dim_conjecture(500, 6, force_degeneracy=False, skip_degeneracies=True)
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
    print(f"isDegeneracy(bdry(counterexample)): {isDegeneracy(bdry(counterexample))}")
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
    print("isDegeneracy(bdry(counterexample)):", isDegeneracy(bdry(counterexample2)))
    comparison = tensor_inner_horn_rank_dimension_comparison(counterexample2, verbose=True)
    conjecture = tensor_inner_horn_rank_dimension_conjecture(counterexample2.shape, verbose=True)
    print("Conjecture:", conjecture, "Comparison:", comparison)
    h = horn(counterexample2, 1)
    print("Horn:", h)
    f = filler(h, 1)
    print("Filler:", f)
    print("Counterexample and filler agree:", np.array_equal(counterexample2,f))
    
