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

import numpy as np
import random
from typing import Tuple # , List, Union, Any
from tensor_ops import n_hypergroupoid_conjecture, n_hypergroupoid_comparison
from tensor_ops import horn, filler, bdry, degen, is_degen    

def random_shape(n: int) -> Tuple[int]:
    # The degree of a hypermatrix is its number of dimensions. Defined as len(S.shape). 
    degree = random.randint(2, n // 2)  # Length of at least two and bounded by n // 2
    return tuple(random.randint(3, n) for _ in range(degree))  # number of elements per axis >= 3 and bounded by n

# force_degeneracy: if True, force random tensor A to be a degeneracy
# skip_degeneracies: if True, skip degeneracies. No effect if force_degeneracy is True.
def run_n_hypergroupoid_conjecture(tests: int, 
                        maxdim:int, 
                        force_degeneracy:bool=False, 
                        skip_degeneracies:bool=False, 
                        outer_horns:bool = False) -> bool:    
    non_unique_horns = 0
    unique_horns = 0
    seed = 123  # Set the seed value
    rng = np.random.default_rng(seed=seed)  # Create a random number generator with the seed

    for i in range(tests):
        print(f"{i+1}: generating random hypermatrix")
        shape = random_shape(maxdim)
        A = rng.integers(low=1, high=10, size=shape, dtype=np.int16)  # Generate random integers
        if force_degeneracy:
            A = degen(A, 0) # force A to be a degeneracy
        if not force_degeneracy and skip_degeneracies:
            while is_degen(A) or is_degen(bdry(A)):
                A = rng.integers(low=1, high=10, size=shape, dtype=np.int16)  # Generate random integers    
        conjecture = n_hypergroupoid_conjecture(shape, verbose=True)
        comparison = n_hypergroupoid_comparison(A, outer_horns=outer_horns, verbose=True)
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
    print(f"Horns w/ unique fillers: {unique_horns}") 
    print(f"Horns w/ non-unique fillers: {non_unique_horns}")
    return True

if __name__ == "__main__":
    run_n_hypergroupoid_conjecture(750, 12, force_degeneracy=False, skip_degeneracies=True)
