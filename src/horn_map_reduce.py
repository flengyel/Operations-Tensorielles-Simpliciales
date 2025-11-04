import dask.bag as db
import itertools
from typing import Set, Tuple
from functools import reduce

def compute_missing_indices_dask(shape: Tuple[int, ...], horn_j: int) -> Set[Tuple[int, ...]]:
    """
    FIXED: Computes the set of missing multi-indices using Dask.
    This version now correctly handles the k < N case.
    """
    if not shape:
        return set()
    
    order_k = len(shape)
    dim_n = min(shape) - 1

    # If k < n, the number of missing indices must be 0.
    if order_k < dim_n:
        return set()

    if not (0 <= horn_j <= dim_n):
        raise ValueError(f"horn_j must be between 0 and {dim_n}, but got {horn_j}")

    horn_faces = [k for k in range(dim_n + 1) if k != horn_j]

    if not horn_faces:
        return set(itertools.product(*(range(s) for s in shape)))

    all_indices_iterator = itertools.product(*(range(s) for s in shape))
    first_face = horn_faces[0]
    initial_bag = db.from_sequence(all_indices_iterator).filter(lambda idx: first_face in idx)

    def intersection_reducer(bag, face_k):
        return bag.filter(lambda idx: face_k in idx)

    final_bag = reduce(intersection_reducer, horn_faces[1:], initial_bag)
    missing_indices = set(final_bag.compute())
    return missing_indices

if __name__ == '__main__':
    # Example from the problem description
    shape_ex = (2, 2)
    horn_j_ex = 1
    missing_indices_ex = compute_missing_indices_dask(shape_ex, horn_j_ex)
    print(f"For shape = {shape_ex} and horn_j = {horn_j_ex}:")
    print(f"The missing indices are: {missing_indices_ex}")
    expected_ex = {(0, 1), (1, 0), (0, 0)}
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

    shapes = [(2,2), (3,3,3), (3,5), (3,3,5), (3,3,3,5), (3,3,3,3,5), (3,4,5,6),
              (3,3,3,3,3,5),(3,3,3,3,3,3,5), (3,3,3,3,3,3,3,5)] 
#    shapes = [(2,2), (3,3,3), (3,3,3,3), (3,4,5,6)] 
   
    horn_j = 1
    for shape in shapes:
        missing_indices = compute_missing_indices_dask(shape, horn_j)
        print(f"\nFor shape = {shape} and horn_j = {horn_j}:")
        print(f"There are {len(missing_indices)} missing indices.")