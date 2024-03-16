#    Opérations Tensorielles Simpliciales
#    Simplicial Operations on Matrices and Hypermatrices
#
#    Copyright (C) 2021-2024 Florian Lengyel
#    Email: florian.lengyel at cuny edu, florian.lengyel at gmail
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Tuple # , List, Union, Any
from operations_tensorielles_simpliciales import tensor_inner_horn_rank_dimension_conjecture
from operations_tensorielles_simpliciales import tensor_inner_horn_rank_dimension_comparison
from operations_tensorielles_simpliciales import horn, filler, bdry, degen, is_degen    
import numpy as np
import random

def random_shape(n: int) -> Tuple[int]:
    length = random.randint(2, n // 2)  # Length of at least two and bounded by 10
    return tuple(random.randint(2, n) for _ in range(length))  # Positive integers at least two and bounded by n

# force_degeneracy: if True, force random tensor A to be a degeneracy
# skip_degeneracies: if True, skip degeneracies. No effect if force_degeneracy is True.
def rank_dim_conjecture(tests: int, maxdim:int, force_degeneracy:bool=False, skip_degeneracies:bool=False) -> bool:    
    non_unique_horns = 0
    unique_horns = 0
    for i in range(tests):
        print(f"{i+1}: generating random hypermatrix")
        shape = random_shape(maxdim)
        A = np.random.randint(low=1, high=10, size=shape, dtype=np.int16)
        if force_degeneracy:
            A = degen(A, 0) # force A to be a degeneracy
        if not force_degeneracy and skip_degeneracies:
            while is_degen(A) or is_degen(bdry(A)):
                A = np.random.randint(low=1, high=10, size=shape, dtype=np.int16)    
        conjecture = tensor_inner_horn_rank_dimension_conjecture(shape, verbose=True)
        comparison = tensor_inner_horn_rank_dimension_comparison(A, verbose=True)
        if comparison != conjecture:
            print(f"Counterexample of shape: {A.shape} with tensor: {A}")
            print(f"Conjecture: {conjecture} Comparison: {comparison}")
            print(f"A is a degeneracy: {is_degen(A)}")
            print(f"bdry(A): {bdry(A)}")
            print(f"boundary is a degeneracy: {is_degen(bdry(A))}")
            return False
        if comparison:
            unique_horns += 1
        else:
            non_unique_horns += 1
    print(f"Conjecture verified for {tests} shapes.") 
    print(f"Inner horns w/ unique fillers: {unique_horns}") 
    print(f"Inner horns w/ non-unique fillers: {non_unique_horns}")
    return True

if __name__ == "__main__":
    rank_dim_conjecture(750, 14, force_degeneracy=False, skip_degeneracies=True)
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
    print(f"isDegeneracy(bdry(counterexample)): {is_degen(bdry(counterexample))}")
    comparison = tensor_inner_horn_rank_dimension_comparison(counterexample, verbose=True)
    conjecture = tensor_inner_horn_rank_dimension_conjecture(counterexample.shape, verbose=True)
    print("Conjecture:", conjecture, "Comparison:", comparison)
    h = horn(counterexample, 1)
    print("Horn:", h)
    f = filler(h, 1)
    print("Filler:", f)
    print("Counterexample and filler agree:", np.array_equal(counterexample,f))

