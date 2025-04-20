import random
from typing import Tuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal

# Import existing implementations from tensor_ops.py
from tensor_ops import (
    face, 
    degen, 
    bdry, 
    random_tensor, 
    permute_tensor, 
    random_axis_permutation
)

# ---------------------------------------------------------------
# Equivariance Tests for Face and Degeneracy Maps under Permutation
# ---------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 42])
def test_face_equivariance(seed: int) -> None:
    """
    Verify that face operations commute with any axis permutation.
    For each random tensor T and permutation sigma,
    d_i(sigma(T)) == sigma(d_i(T)) holds for all valid i.
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. Random tensor order k ≥ 2
    k = random.randint(2, 6)

    # 2. Random shape with each dimension ≥ 2
    shape = tuple(random.randint(2, 7) for _ in range(k))

    # 3. Generate random integer tensor T
    T = random_tensor(shape, low=0, high=100, seed=seed)

    # 4. Random permutation σ of axes
    sigma = random_axis_permutation(k)
    T_perm = permute_tensor(T, sigma)

    # 5. Test all valid face indices 0 ≤ i < min(shape)
    for i in range(min(shape)):
        # Compute d_i(σ(T))
        f_after = face(T_perm, i)
        # Compute σ(d_i(T))
        f_before = permute_tensor(face(T, i), sigma)
        # Assert equivariance
        assert_array_equal(
            f_after, f_before,
            f"Face equivariance failed: shape={shape}, sigma={sigma}, i={i}"
        )


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_degeneracy_equivariance(seed: int) -> None:
    """
    Verify that degeneracy operations commute with any axis permutation.
    For each random tensor T and permutation sigma,
    s_i(sigma(T)) == sigma(s_i(T)) holds for all valid i.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Random order and shape
    k = random.randint(2, 6)
    shape = tuple(random.randint(2, 7) for _ in range(k))
    T = random_tensor(shape, low=0, high=100, seed=seed)

    sigma = random_axis_permutation(k)
    T_perm = permute_tensor(T, sigma)

    # Test all valid degeneracy indices 0 ≤ i < min(shape)
    for i in range(min(shape)):
        # Compute s_i(σ(T))
        d_after = degen(T_perm, i)
        # Compute σ(s_i(T))
        d_before = permute_tensor(degen(T, i), sigma)
        assert_array_equal(
            d_after, d_before,
            f"Degeneracy equivariance failed: shape={shape}, sigma={sigma}, i={i}"
        )

@pytest.mark.parametrize("seed", [0, 1, 42])
def test_boundary_equivariance(seed: int) -> None:
    """
    Verify that boundary operations commute with any axis permutation.
    For each random tensor T and permutation sigma,
    bdry(sigma(T)) == sigma(bdry(T)) holds.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Random order and shape
    k = random.randint(2, 6)
    shape = tuple(random.randint(2, 7) for _ in range(k))
    T = random_tensor(shape, low=0, high=100, seed=seed)

    sigma = random_axis_permutation(k)
    T_perm = permute_tensor(T, sigma)

    # Compute boundary of permuted tensor
    b_after = bdry(T_perm)
    # Compute permuted boundary of original tensor
    b_before = permute_tensor(bdry(T), sigma)

    assert_array_equal(
        b_after, b_before,
        f"Boundary equivariance failed: shape={shape}, sigma={sigma}"
    )
