# Google Gemini hallucinated two "theorems" in combinatorial tensor theory.
#
# **Hallucinated Inner Horn Uniqueness Theorem**:
# "An inner horn is unique if and only if the tensor has rank at most 2 and 
# its shape is a partition of a number.This theorem was first proven by Thomas Brylawski in 1975. 
# It is a powerful tool in the study of combinatorial tensor theory and has been used to prove many 
# other results in the field." 
# 
# This unmitigated horseshit was followed by an even more egregious claim:
#
# **Hallucinated Hypermatrix Uniqueness Theorem**:
# There is a version of the Inner Horn Uniqueness Theorem that depends on the shape and rank of 
# a multi-dimensional matrix. It was first proven by Ezra Getzler and Jacob Lurie in 2009, and 
# it states as follows: An inner horn of a multi-dimensional matrix is unique if and only if the 
# multi-dimensional matrix has rank at most 2 and its shape is a partition of a number.
# 
# Bard, now Gemini, claimed that [0,1,2]⊗[3,4]⊗[5,6] does not have a unique inner horn.
# This is vacuously true, because it does not have an inner horn at all. This tensor has shape (3,2,2)
# which has simplicial dimension min(shape) - 1 = 1, which means there are no outer horns. 
# 
# Let's check with our code. Gemini produced the following Python code to construct the hypermatrix.

import numpy as np
from tensor_ops import n_hypergroupoid_conjecture, n_hypergroupoid_comparison, is_degen, bdry, horn
from tensor_ops import face, kan_condition, filler


a = np.array([0, 1, 2])
b = np.array([3, 4])
c = np.array([5, 6])

# Create an empty 3D array to store the tensor product
tensor = np.zeros((3, 2, 2), dtype=int)

# Fill the tensor by taking the pairwise product of elements from the 1D arrays
for i in range(3):
    for j in range(2):
        for k in range(2):
            tensor[i, j, k] = a[i] * b[j] * c[k]

# Now, let's check if the inner horn is unique according to our conjecture
print(f"Gemini's tensor:\n {tensor}")
print(f"Does Gemini's tensor satisfy the n-hypergroupoid conjecture? {n_hypergroupoid_conjecture(tensor.shape, verbose=True)}")
print(f"Is Gemini's tensor degenerate? {is_degen(tensor)}")
print(f"Is the boundary of Gemini's tensor degenerate? {is_degen(bdry(tensor))}")
print(f"Does Gemini's tensor have a unique outer horn: {n_hypergroupoid_comparison(tensor, outer_horns=True, verbose=True)}")

# compute the outer horn that omits the 0-face
h0 = horn(tensor, 0)
f0 = face(tensor, 0)
print(f"0-face:\n{f0}")
f1 = face(tensor, 1)
print(f"1-face:\n{f1}")
print(f"Outer horn that omits the first axis:\n{h0}")
# does the outer horn satisfy the Kan condition?
print(f"Does the outer horn satisfy the Kan condition? {kan_condition(h0, 0)}")
# fill in the missing face of the 0-horn (an outer horn)
ftensor = filler(h0, 0)
print(f"Filler tensor:\n{ftensor}")
# what about the 0-horn of the filler tensor? Does this agree with the 0-horn of the original tensor?
h1 = horn(ftensor, 0)
print(f"Do the 0-horns of the filler tensor and the original tensor agree? {np.array_equal(h0, h1)}")   
# are the filler tensor and the original tensor equal?
print(f"Are the filler tensor and the original tensor equal? {np.array_equal(tensor, ftensor)}")

# The output of the code shows that the tensor [0,1,2]⊗[3,4]⊗[5,6] does not satisfy 
# the n-hypergroupoid conjecture, and hence does not have a unique inner horn.
# This does not contradict Gemini's claim.

# Let's now check the tensor [0,1,2,3,4,5]⊗[3,4,5,6,7,8,9]⊗[5,6,7,8,9]⊗[7,8,9,10,11,12] 
# which has shape (6, 7, 6, 7) (i.e., 4D tensor with shape (6, 7, 6, 7) and simplicial dimension
# min(shape) - 1 = 4 - 1 = 3. Hence len(shape) = 4 < s_dim(shape) - 1 = 5. This tensor should satisfy 
# the n-hypergroupoid conjecture and have a unique inner horn unless the boundary is degenerate

a = np.array([0, 1, 2, 3, 4, 5])
b = np.array([3, 4, 5, 6, 7, 8, 9])
c = np.array([5, 6, 7, 8, 9, 10])
d = np.array([7, 8, 9, 10, 11, 12, 13])

# Create an empty 4D array to store the tensor product
tensor = np.zeros((a.size, b.size, c.size, d.size), dtype=int)

# Fill the tensor by taking the pairwise product of elements from the 1D arrays
for i in range(a.size):
    for j in range(b.size):
        for k in range(c.size):
            for l in range(d.size):
                tensor[i, j, k, l] = a[i] * b[j] * c[k] * d[l]

print(f"Does the tensor product satisfy the n-hypergroupoid conjecture? {n_hypergroupoid_conjecture(tensor.shape, verbose=True)}")
print(f"Is the tensor product degenerate? {is_degen(tensor)}")
print(f"Is the boundary of the tensor product degenerate? {is_degen(bdry(tensor))}")
print(f"Does the tensor product have a unique inner  horn: {n_hypergroupoid_comparison(tensor, outer_horns=False, verbose=True)}")

# compute an inner horn 
h0 = horn(tensor, 2)
# does the outer horn satisfy the Kan condition?
print(f"Does the 2-horn of the tensor product satisfy the Kan condition? {kan_condition(h0, 2)}")
# fill in the missing face of the 0-horn (an outer horn)
ftensor = filler(h0, 2)
# what about the 0-horn of the filler tensor? Does this agree with the 0-horn of the original tensor?
h1 = horn(ftensor, 2)
print(f"Do the 2-horns of the filler tensor and the original tensor agree? {np.array_equal(h0, h1)}")
# are the filler tensor and the original tensor equal?
print(f"Are the filler tensor and the original tensor equal? {np.array_equal(tensor, ftensor)}")


