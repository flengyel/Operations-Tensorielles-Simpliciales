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
from tensor_ops import (n_hypergroupoid_conjecture, n_hypergroupoid_comparison, 
                        bdry, degen, is_degen, ___SEED___, random_tensor, order, dimen, face) 


random.seed(___SEED___) # Set seed for reproducibility

# random_shape: generates a random shape for a hypermatrix
def random_shape(n: int) -> Tuple[int]:
    # The order of a hypermatrix is its number of dimensions. Defined as len(S.shape). 
    order = random.randint(2, n // 2)  # Length of at least two and bounded by n // 2
    return tuple(random.randint(3, n) for _ in range(order))  # number of elements per axis >= 3 and bounded by n


def generate_tensor(shape: Tuple[int], force_degeneracy: bool, skip_degeneracies: bool) -> np.ndarray:
    A = random_tensor(shape, low=1, high=10)  # Generate random integers
    if force_degeneracy:
        A = degen(A, 0)
        shape = A.shape
    if not force_degeneracy and skip_degeneracies:
        while is_degen(bdry(A)):
            print(f"Regenerating random tensor of shape: {A.shape}")
            A = random_tensor(shape, low=1, high=10)  # Generate random integers
    return A

# force_degeneracy: if True, force random tensor A to be a degeneracy
# skip_degeneracies: if True, skip degeneracies. No effect if force_degeneracy is True.
def run_n_hypergroupoid_conjecture(tests: int, 
                        maxdim:int, 
                        force_degeneracy:bool=False, 
                        skip_degeneracies:bool=False, 
                        outer_horns:bool = False,
                        single_shape: Tuple[int] = (0,0)) -> bool:    
    non_unique_horns = 0
    unique_horns = 0

    shape = single_shape
    for i in range(tests):
        print(f"{i+1}: generating random hypermatrix")    
        if single_shape == (0,0):
            shape = random_shape(maxdim)

        A = generate_tensor(shape, force_degeneracy, skip_degeneracies)

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

def verify_unique_down_to_order(shape: Tuple[int] = (19, 18, 17, 19), outer_horns:bool = False) -> bool:
    
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
  
    _order = initial_order = order(A)
    dim = top_dim = dimen(A)

    print(f"Initial tensor has shape: {initial_shape}, order: {initial_order}, and s-dimension: {top_dim}.")

    conjecture = n_hypergroupoid_conjecture(shape, verbose=True)
    print(f"Unique filler for order {_order}and s-dimension {dim}")
            
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
    # so that _order < dim and the dimension can be reduced 
    while _order < dim:
        A = face(A, 0)
        _order = order(A)
        dim = dimen(A) # reduce dimension
        shape = A.shape

        conjecture = n_hypergroupoid_conjecture(shape, verbose=True)
        if conjecture:
            print(f"Unique filler for order {_order} and dimension {dim}")
            comparison = n_hypergroupoid_comparison(A, outer_horns=outer_horns, verbose=True, allow_degen=True)
            print(f"Checking comparison for tensor of shape: {A.shape}: {comparison}")
            if not comparison:
                print(f"CRAP! Comparison failed for tensor of shape: {A.shape}")
                return False
        else:
            print(f"non-unique filler for order {_order} and s-dimension {dim}")
            print(f"Tensor of shape {initial_shape} generates unique fillers from dimension {_order+1} to dimension {top_dim} and beyond.")    
            return True
        

if __name__ == "__main__":
    run_n_hypergroupoid_conjecture(750, 12, 
                                   force_degeneracy=False, 
                                   skip_degeneracies=True, 
                                   outer_horns=False,
                                   single_shape=(8,10, 12))
    #run_n_hypergroupoid_conjecture(750, 12, 
    #                               force_degeneracy=False, 
    #                               skip_degeneracies=False, 
    #                               outer_horns=True,
    #                               single_shape=(6,6,6))

    verify_unique_down_to_order(shape=(8,10,12), outer_horns=False)
    verify_unique_down_to_order(shape=(8,10,12), outer_horns=True)
    