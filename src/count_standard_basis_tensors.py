# count_standard_basis_tensors.py
"""
This script counts the number of degenerate and non-degenerate standard basis tensors
for given tensor shapes. A standard basis tensor has a single '1' at a specified index
and '0's everywhere else. The degeneracy is determined by the `is_degen` function from
the `tensor_ops` module.
"""

import numpy as np
import itertools
from tensor_ops import is_degen
import logging
import datetime

# --- Setup Logging ---
# This will create a log file in the same directory as the script.
# The file will be overwritten each time the script is run (filemode='w').
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'logs/standard_basis_tensors_{timestamp}.log',
    filemode='w'
)
# Also log to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s') # Keep console output clean
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

def standard_basis_tensor(shape: tuple, index: tuple) -> np.ndarray:
    """
    Creates a standard basis tensor of a given shape.

    The tensor will have a single '1' at the specified index and
    '0's everywhere else.

    Args:
        shape: A tuple representing the tensor shape (e.g., (3, 4, 2)).
        index: A tuple for the location of the '1'.

    Returns:
        A NumPy array representing the standard basis tensor.
    """
    tensor = np.zeros(shape, dtype=int)
    tensor[index] = 1
    return tensor

def count_basis_tensors_for_shape(s: tuple):
    """
    Counts the degenerate and non-degenerate standard basis tensors for a shape.

    Args:
        s: A tuple representing the tensor shape.
    """
    degen_count = 0
    non_degen_count = 0
    
    # Generate all possible valid indices for the given shape
    index_ranges = [range(dim) for dim in s]
    all_indices = itertools.product(*index_ranges)

    # Iterate through each index, create the corresponding basis tensor, and check it
    non_degen_indices = set()
    for index in all_indices:
        t = standard_basis_tensor(s, index)
        if is_degen(t):
            degen_count += 1
        else:
            non_degen_count += 1
            non_degen_indices.add(index)

    # Report the results
    logging.info(f"Shape: {s}")
    logging.info(f"  - Degenerate standard basis tensors: {degen_count}")
    logging.info(f"  - Non-degenerate standard basis tensors: {non_degen_count}")
    # print non-degenerate indices for verification
    if non_degen_count > 0:
        logging.info(f"  - Non-degenerate indices: {sorted(non_degen_indices)}")   


if __name__ == "__main__":
    shapes = [
        (3, 3),       # Example shape 1
        (3, 3, 3),    # Example shape 2
        (4, 4, 4),     # Example shape 3
        (4,4),
        (4,5),
        (5,5),
        (6,6),
        (7,7),
        (8,8),
        (5,5,5),
        (5,5,6),
    ]
    for shape in shapes:
        count_basis_tensors_for_shape(shape)

    