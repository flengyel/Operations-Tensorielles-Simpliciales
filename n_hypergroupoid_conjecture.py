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
from tensor_ops import bdry, degen, is_degen, ___SEED___, random_tensor    
from tensor_ops import degree, s_dim, face 


random.seed(___SEED___) # Set seed for reproducibility

# random_shape: generates a random shape for a hypermatrix
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

    for i in range(tests):
        print(f"{i+1}: generating random hypermatrix")
        shape = random_shape(maxdim)
        A = random_tensor(shape, low=1, high=10)  # Generate random integers
        if force_degeneracy:
            A = degen(A, 0) # force A to be a degeneracy
        if not force_degeneracy and skip_degeneracies:
            while is_degen(A) or is_degen(bdry(A)):
                A = random_tensor(shape, low=1, high=10)  # Generate random integers    
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

def verify_unique_down_to_degree(shape: Tuple[int] = (19, 18, 17, 19), outer_horns:bool = False) -> bool:
    
    A = random_tensor(shape, low=1, high=10)  
    
    # ensure nondegeneracy of A and bdry(A)
    while is_degen(A) or is_degen(bdry(A)):
        A = random_tensor(shape, low=1, high=10)  # Generate random integers

    A = degen(A, 0) # force A to be a degeneracy
    A = degen(A, 0) # force A to be a degeneracy of a degeneracy
    A = degen(A, 0) # force A to be a degeneracy of a degeneracy of a degeneracy

    print(f"A is a degeneracy: {is_degen(A)}")

    initial_shape = A.shape
    shape = A.shape # update shape of the tensor after adding degeneracies
  
    deg = initial_degree = degree(A)
    sdim = top_sdim = s_dim(A)

    print(f"Initial tensor has shape: {initial_shape}, degree: {initial_degree}, and s-dimension: {top_sdim}.")

    conjecture = n_hypergroupoid_conjecture(shape, verbose=True)
    print(f"Unique filler for degree {deg} and s-dimension {sdim}")
            
    comparison = n_hypergroupoid_comparison(A, outer_horns=outer_horns, verbose=True, allow_degen=True)
    print(f"Checking comparison for tensor of shape: {A.shape}: {comparison}")
    
    if not conjecture:
        print(f"Random tensor has non-unique filler for shape: {A.shape}. Start over.")
        return False

    if comparison != conjecture:
        print(f"Counterexample of shape: {A.shape} with tensor: {A}")
        print(f"Conjecture: {conjecture} Comparison: {comparison}")
        print(f"A is a degeneracy: {is_degen(A)}")
        print(f"bdry(A): {bdry(A)}")
        print(f"boundary is a degeneracy: {is_degen(bdry(A))}")
        return False

    # the comparison and conjecture are both True
    # so that degree < sdim and the s-diomension can be reduced 
    while deg < sdim:
        A = face(A, 0)
        deg = degree(A)
        sdim = s_dim(A) # reduce s-dimension
        shape = A.shape

        conjecture = n_hypergroupoid_conjecture(shape, verbose=True)
        if conjecture:
            print(f"Unique filler for degree {deg} and s-dimension {sdim}")
            comparison = n_hypergroupoid_comparison(A, outer_horns=outer_horns, verbose=True, allow_degen=True)
            print(f"Checking comparison for tensor of shape: {A.shape}: {comparison}")
            if not comparison:
                print(f"CRAP! Comparison failed for tensor of shape: {A.shape}")
                return False
        else:
            print(f"non-unique filler for degree {deg} and s-dimension {sdim}")
            print(f"Tensor of shape {initial_shape} generates unique fillers from s-dimension {deg+1} to s-dimension {top_sdim} and beyond.")    
            return True
        

if __name__ == "__main__":
#   run_n_hypergroupoid_conjecture(750, 12, force_degeneracy=False, skip_degeneracies=True)
    #verify_unique_down_to_degree()
    #verify_unique_down_to_degree(shape=(5,5))
    #verify_unique_down_to_degree(shape=(6,6))
    #verify_unique_down_to_degree(shape=(7,7))
    #verify_unique_down_to_degree(shape=(5,5,5))
    #verify_unique_down_to_degree(shape=(6,6,6))
    verify_unique_down_to_degree(shape=(7,11,23,43))
