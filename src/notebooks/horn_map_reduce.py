import dask.bag as db
import itertools
from typing import Set, Tuple
from functools import reduce

def compute_missing_indices_dask(shape: Tuple[int, ...], horn_j: int) -> Set[Tuple[int, ...]]:
    """
    Computes the set of missing multi-indices for a given tensor horn using Dask.

    Args:
        shape: A tuple representing the tensor's shape.
        horn_j: The index of the missing face in the horn.

    Returns:
        A set of tuples, where each tuple is a missing multi-index.
    """
    # 1. Calculate dimension N and the set of face indices present in the horn.
    if not shape:
        return set()
    n = min(shape) - 1
    if not (0 <= horn_j <= n):
        raise ValueError(f"horn_j must be between 0 and {n}, but got {horn_j}")

    horn_faces = [k for k in range(n + 1) if k != horn_j]

    if not horn_faces:
        # If there are no faces in the horn, all indices are "missing".
        all_indices = set(itertools.product(*(range(s) for s in shape)))
        return all_indices

    # 2. Create a dask.bag representing all possible index tuples for the shape.
    #    (Hint: use itertools.product and dask.bag.from_sequence)
    all_indices_iterator = itertools.product(*(range(s) for s in shape))
    # It's more efficient to start with a filtered bag than to create bags for all E_k
    # and then intersect them. We can start with E_k for the first face.
    first_face = horn_faces[0]
    initial_bag = db.from_sequence(all_indices_iterator).filter(lambda idx: first_face in idx)

    # 3. For each face index 'k' present in the horn, create the set E_k in parallel.
    #    This is the "map" step. You can use dask.bag.filter for this.
    # 4. Compute the intersection of all the E_k sets. This is the "reduce" step.
    # We can combine the map and reduce steps by iteratively filtering the bag.
    
    def intersection_reducer(bag, face_k):
        return bag.filter(lambda idx: face_k in idx)

    # Start with the initial bag and apply the filter for the rest of the faces.
    # The reduce function here is a high-level concept; in Dask, it's a sequence of transformations.
    final_bag = reduce(intersection_reducer, horn_faces[1:], initial_bag)

    # 5. Return the final set of missing indices.
    # The .compute() method triggers the actual Dask computation.
    missing_indices = set(final_bag.compute())

    return missing_indices

if __name__ == '__main__':
    # Example from the problem description
    shape_ex = (3, 3)
    horn_j_ex = 1
    missing_indices_ex = compute_missing_indices_dask(shape_ex, horn_j_ex)
    print(f"For shape = {shape_ex} and horn_j = {horn_j_ex}:")
    print(f"The missing indices are: {missing_indices_ex}")
    expected_ex = {(0, 2), (2, 0)}
    print(f"Expected: {expected_ex}")
    assert missing_indices_ex == expected_ex
    print("Example assertion passed!")

    # A more complex example
    shape_complex = (4, 4, 4)
    horn_j_complex = 0
    # N = 3. Horn faces are k=1, 2, 3.
    # We need indices (i1, i2, i3) where 1, 2, AND 3 are present.
    # e.g., (1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1)
    missing_indices_complex = compute_missing_indices_dask(shape_complex, horn_j_complex)
    print(f"\nFor shape = {shape_complex} and horn_j = {horn_j_complex}:")
    print(f"The missing indices are: {missing_indices_complex}")
    expected_complex = set(itertools.permutations([1, 2, 3]))
    print(f"Expected: {expected_complex}")
    assert missing_indices_complex == expected_complex
    print("Complex example assertion passed!")

    # Example with a larger shape to demonstrate potential for parallelism
    shape_large = (10, 10, 10)
    horn_j_large = 2
    # N = 9. Horn faces are 0,1,3,4,5,6,7,8,9
    # This would be slow to compute manually.
    print(f"\nComputing for large shape = {shape_large} and horn_j = {horn_j_large}...")
    missing_indices_large = compute_missing_indices_dask(shape_large, horn_j_large)
    print(f"Found {len(missing_indices_large)} missing indices.")
    
    # A simple check: one such index must contain all faces present in the horn.
    # This is not a full check, but a sanity check.
    if missing_indices_large:
        # **FIX:** Define horn_faces in the local scope for the assertion.
        n_large = min(shape_large) - 1
        horn_faces_large = [k for k in range(n_large + 1) if k != horn_j_large]
        
        sample_index = next(iter(missing_indices_large))
        print(f"A sample missing index is: {sample_index}")
        
        all_faces_present = all(face in sample_index for face in horn_faces_large)
        assert all_faces_present
        print(f"Sanity check passed: A sample index indeed contains all faces from the horn.")

    shapes = [(3,3), (3,3,3), (3,3,3,3), (3,3,3,3,3), 
              (3,3,3,3,3,3),(3,3,3,3,3,3,3), (3,3,3,3,3,3,3,3), 
              (3,3,3,3,3,3,3,3,3),(3,3,3,3,3,3,3,3,3,3)]
    shapes = [(3,5), (3,4,5), (3,5,5)]
    shapes = [(3,5), (3,3,5), (3,3,3,5), (3,3,3,3,5), 
              (3,3,3,3,3,5),(3,3,3,3,3,3,5), (3,3,3,3,3,3,3,5)] 

   
    horn_j = 1
    for shape in shapes:
        missing_indices = compute_missing_indices_dask(shape, horn_j)
        print(f"\nFor shape = {shape} and horn_j = {horn_j}:")
        print(f"There are {len(missing_indices)} missing indices.")