"""
Tensor Module — SKELETON
Version: v0.5

Wraps C tensor operations via ctypes.
"""

from typing import List, Optional, Tuple, Union
import ctypes


class Tensor:
    """Multi-dimensional array backed by C ArixTensor."""

    def __init__(self, shape: Tuple[int, ...], dtype: str = "float32"):
        self._shape = shape
        self._dtype = dtype
        self._data = None  # TODO(v0.5): allocate via arix_tensor_create

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    def numpy(self):
        """Convert to NumPy array (requires cast)."""
        raise NotImplementedError("v0.5")

    @staticmethod
    def zeros(shape: Tuple[int, ...], dtype: str = "float32") -> "Tensor":
        return Tensor(shape, dtype)

    @staticmethod
    def ones(shape: Tuple[int, ...], dtype: str = "float32") -> "Tensor":
        return Tensor(shape, dtype)

    @staticmethod
    def randn(shape: Tuple[int, ...], dtype: str = "float32") -> "Tensor":
        return Tensor(shape, dtype)

    def __add__(self, other: "Tensor") -> "Tensor":
        raise NotImplementedError("v0.5")

    def __mul__(self, other: "Tensor") -> "Tensor":
        raise NotImplementedError("v0.5")

    def __matmul__(self, other: "Tensor") -> "Tensor":
        raise NotImplementedError("v0.5")
