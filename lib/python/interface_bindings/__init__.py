"""
ARIX-Algo Python Library — SKELETON
Version: v0.5

High-level Python API for tensor operations, model construction,
and training.  Wraps the C library via ctypes/cffi.
"""

from .core import Tensor
from .nn import Module, Linear, ReLU, Sequential
from .optim import SGD, Adam

__all__ = ["Tensor", "Module", "Linear", "ReLU", "Sequential", "SGD", "Adam"]
