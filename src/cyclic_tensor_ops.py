# cyclic_tensor_ops.py
# -*- coding: utf-8 -*-
import numpy as np
from typing import Any, Union

def face_axis(tensor: np.ndarray, axis: int, index: int) -> np.ndarray:
    """
    Remove the slice at position `index` along the given `axis`.

    This is the per‐axis simplicial face:
      d_i^{(axis)} : Tensor[..., shape[axis], ...] → Tensor[..., shape[axis]-1, ...]

    Parameters:
    -----------
    tensor : np.ndarray
        The input tensor of arbitrary order.
    axis : int
        Which axis to delete (0 ≤ axis < tensor.ndim).
    index : int
        Which coordinate along that axis to remove
        (0 ≤ index < tensor.shape[axis]).

    Returns:
    --------
    np.ndarray
        A new array with the same number of dimensions,
        but with size along `axis` decreased by one.
    """
    return np.delete(tensor, index, axis=axis)


def degen_axis(tensor: np.ndarray, axis: int, index: int) -> np.ndarray:
    """
    Insert a duplicate of the slice at position `index` along the given `axis`.

    This is the per‐axis simplicial degeneracy:
      s_i^{(axis)} : Tensor[..., shape[axis], ...] → Tensor[..., shape[axis]+1, ...]

    Parameters:
    -----------
    tensor : np.ndarray
        The input tensor of arbitrary order.
    axis : int
        Which axis to duplicate (0 ≤ axis < tensor.ndim).
    index : int
        The coordinate along `axis` whose slice gets duplicated
        (0 ≤ index < tensor.shape[axis]).

    Returns:
    --------
    np.ndarray
        A new array with the same number of dimensions,
        but with size along `axis` increased by one.
    """
    # Extract the slice at position `index` along `axis`
    slicer: list[Union[int, slice]] = [slice(None)] * tensor.ndim        
    slicer[axis] = index
    slice_i = tensor[tuple(slicer)]
    # Insert it back at position `index` along the same axis
    return np.insert(tensor, index, slice_i, axis=axis)


def cyclic(tensor: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    The simplicial-cycle operator τ on a chosen axis:
      τ : C_n → C_n,   τ(x_0, x_1, …, x_n) = (x_n, x_0, x_1, …, x_{n-1}).

    In array‐terms, it right‐rotates the entries along `axis` by one step
    (moving the last slice to the front).

    Parameters:
    -----------
    tensor : np.ndarray
      The input array whose `axis`‐th dimension has length d+1.
    axis : int, optional
      Which axis to cycle (default 0).

    Returns:
    --------
    np.ndarray
      A new array with the same shape, but with the `axis`‐th coordinate
      rolled so that index d→0, 0→1, …, d−1→d.
    """
    return np.roll(tensor, shift=1, axis=axis)


def cyclic_signed(tensor: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Connes’ signed cycle λ = (−1)^d τ, where d = dimen(tensor) along `axis`.

    Parameters:
    -----------
    tensor : np.ndarray
    axis   : int, optional
      Which axis to treat as the simplicial axis (default 0).

    Returns:
    --------
    np.ndarray
      The same as `cyclic(tensor, axis)` but multiplied by (−1)^d, where
      d = tensor.shape[axis] − 1.
    """
    d = tensor.shape[axis] - 1
    sign = -1 if (d % 2) else 1
    return sign * cyclic(tensor, axis)
