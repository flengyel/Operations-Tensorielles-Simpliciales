import numpy as np
import pytest
import random
from numpy.testing import assert_array_equal
from cyclic_tensor_ops import face_axis, degen_axis, cyclic, cyclic_signed

# helper to build a small random tensor
def random_tensor(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 100, size=shape)


@pytest.mark.parametrize("shape,axis", [
    ((4, 3, 2), 0),
    ((4, 3, 2), 1),
    ((4, 3, 2), 2),
])
def test_face_axis_simple(shape, axis):
    """face_axis should match np.delete along the given axis."""
    T = random_tensor(shape, seed=42)
    for idx in range(shape[axis]):
        expected = np.delete(T, idx, axis=axis)
        got      = face_axis(T, axis, idx)
        assert_array_equal(got, expected,
            f"face_axis failed for shape={shape}, axis={axis}, idx={idx}")


@pytest.mark.parametrize("shape,axis", [
    ((3, 2), 0),
    ((3, 2), 1),
    ((4, 3, 2), 0),
    ((4, 3, 2), 1),
    ((4, 3, 2), 2),
])
def test_degen_axis_simple(shape, axis):
    """degen_axis should match np.insert of a single‐slice along the axis."""
    T = random_tensor(shape, seed=7)
    for idx in range(shape[axis]):
        # extract slice
        slicer = [slice(None)] * T.ndim
        slicer[axis] = idx
        slice_i = T[tuple(slicer)]
        expected = np.insert(T, idx, slice_i, axis=axis)
        got      = degen_axis(T, axis, idx)
        assert_array_equal(got, expected,
            f"degen_axis failed for shape={shape}, axis={axis}, idx={idx}")


@pytest.mark.parametrize("shape,axis", [
    ((3, 4), 0),
    ((3, 4), 1),
    ((2, 3, 4), 0),
    ((2, 3, 4), 1),
    ((2, 3, 4), 2),
])
def test_cyclic_rolls_axis(shape, axis):
    """cyclic should roll the given axis by +1 (last→first)."""
    T = random_tensor(shape, seed=13)
    got      = cyclic(T, axis=axis)
    expected = np.roll(T, shift=1, axis=axis)
    assert_array_equal(got, expected,
        f"cyclic failed to roll axis {axis} for shape={shape}")


@pytest.mark.parametrize("shape,axis", [
    ((2, 3), 0),  # d=1 → sign = -1
    ((3, 2), 0),  # d=2 → sign = +1
    ((3, 4, 5), 1),  # d=3 → sign = -1
    ((5, 3, 2), 2),  # d=1 → sign = -1
])
def test_cyclic_signed_sign_and_roll(shape, axis):
    """
    cyclic_signed should multiply cyclic by (-1)^d, 
    where d = tensor.shape[axis]-1.
    """
    T = random_tensor(shape, seed=99)
    d = shape[axis] - 1
    sign = -1 if (d % 2) else 1

    got      = cyclic_signed(T, axis=axis)
    expected = sign * np.roll(T, shift=1, axis=axis)
    assert_array_equal(got, expected,
        f"cyclic_signed failed for shape={shape}, axis={axis}, expected sign={sign}")


def test_face_axis_then_cyclic_commutes():
    """
    Pure cyclic‐module face axiom on axis=0:
      for i>0: face_axis(τ(T),0,i) == τ(face_axis(T,0,i-1))
      for i=0: face_axis(τ(T),0,0) == face_axis(T,0,d)
    """
    shape = (4,5,6)    # axis 0 has length 4 ⇒ d=3
    T     = random_tensor(shape, seed=123)
    axis  = 0
    d     = shape[axis] - 1

    for i in range(d+1):
        lhs = face_axis(cyclic(T, axis=axis), axis, i)
        if i > 0:
            rhs = cyclic(face_axis(T, axis=axis, index=i-1), axis=axis)
        else:
            rhs = face_axis(T, axis=axis, index=d)
        assert_array_equal(
            lhs, rhs,
            f"cyclic‐module face identity failed at i={i}, axis={axis}"
        )



@pytest.mark.parametrize("seed", [0, 1, 42])
def test_degen_axis_then_cyclic_commutes(seed):
    """
    Pure cyclic‐module degeneracy axiom on axis=1:
      for i>0: s_i τ = τ s_{i-1}
      for i=0: s_0 τ = σ_d τ^2
    """
    # reseed both Python and NumPy
    random.seed(seed)
    np.random.seed(seed)

    shape = (3, 4, 5)    # axis 1 has length 4 ⇒ d = 3
    T     = random_tensor(shape, seed=seed)
    axis  = 1
    d     = shape[axis] - 1

    for i in range(d + 1):
        lhs = degen_axis(cyclic(T, axis=axis), axis, i)
        if i > 0:
            # s_i τ == τ s_{i-1}
            rhs = cyclic(degen_axis(T, axis, index=i-1), axis=axis)
        else:
            # s_0 τ == σ_d τ^2
            g   = degen_axis(T, axis, index=d)                  # σ_d(T)
            rhs = cyclic(cyclic(g, axis=axis), axis=axis)       # τ^2 (σ_d T)
        assert_array_equal(
            lhs, rhs,
            f"cyclic‐module degeneracy identity failed at seed={seed}, i={i}, axis={axis}"
        )
