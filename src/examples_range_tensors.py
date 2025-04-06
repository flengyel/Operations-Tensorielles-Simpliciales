# examples of range tensors
import numpy as np
from tensor_ops import range_tensor, n_hypergroupoid_conjecture, horn, face 
from tensor_ops import reconstruct_range_tensor_from_horn, filler, is_degen, bdry  
from typing import Tuple

def print_range_tensor(shape: Tuple[int]) -> None:
    relation = "<" if n_hypergroupoid_conjecture(shape) else ">="
    print(f"shape = {shape}, degree = {len(shape)} {relation} s_dim = {min(shape)-1}")
    print(f"range_tensor(shape={shape}) \n{range_tensor(shape)}\n")

if __name__ == "__main__":
    print("Examples of range tensors")

shape = (2,2)
print_range_tensor(shape)

shape = (3,3)
print_range_tensor(shape)
reconstructed = reconstruct_range_tensor_from_horn(shape, proceed_anyway=True, verbose=True)
print(f"Range tensor of shape {shape} can be reconstructed from any of its horns: {reconstructed}\n")


X = np.array([[-6,  5,  7],
              [-5,  4,  8],
              [ 1,  9,  0]])
H = horn(X, 1)
Y = filler(H, 1)
print(f"X == Y: {np.array_equal(X,Y)}\n")
print(f"is_degen(X): {is_degen(X)}\n")
print(f"bdry(X): \n{bdry(X)}\n")
print(f"is_degen(bdry(X)): {is_degen(bdry(X))}\n")

# AHA! The boundary of the tensor X is non-degenerate.
# what about a range tensor of shape (4,4)?
shape = (4,4)
reconstruct_range_tensor_from_horn(shape, proceed_anyway=True, verbose=True)
print(f"Range tensor of shape {shape} can be reconstructed from any of its horns: {reconstructed}\n")


shape = (5,5,5)
print_range_tensor(shape)
reconstructed = reconstruct_range_tensor_from_horn(shape, proceed_anyway=True, verbose=True)  
print(f"Range tensor of shape {shape} can be reconstructed from any of its horns: {reconstructed}\n")

shape = (6,6,6)
print_range_tensor(shape)
reconstructed = reconstruct_range_tensor_from_horn(shape, proceed_anyway=True, verbose=True)
print(f"Range tensor of shape {shape} can be reconstructed from any of its horns: {reconstructed}\n")

shape = (7,7,7)
print_range_tensor(shape)
reconstructed = reconstruct_range_tensor_from_horn(shape, proceed_anyway=True, verbose=True)
print(f"Range tensor of shape {shape} can be reconstructed from any of its horns: {reconstructed}\n")