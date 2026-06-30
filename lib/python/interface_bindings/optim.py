"""
Optimizer Module — SKELETON
Version: v0.5

Optimizers for training neural networks.
"""

from typing import Callable, List, Optional
from .core import Tensor


class Optimizer:
    """Base optimizer (v0.5)."""

    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self._params = params
        self._lr = lr

    def step(self):
        raise NotImplementedError("v0.5")

    def zero_grad(self):
        raise NotImplementedError("v0.5")


class SGD(Optimizer):
    """Stochastic gradient descent (v0.5)."""

    def __init__(self, params: List[Tensor], lr: float = 0.01, momentum: float = 0.0):
        super().__init__(params, lr)
        self._momentum = momentum

    def step(self):
        raise NotImplementedError("v0.5")


class Adam(Optimizer):
    """Adam optimizer (v0.5)."""

    def __init__(self, params: List[Tensor], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        super().__init__(params, lr)
        self._betas = betas
        self._eps = eps

    def step(self):
        raise NotImplementedError("v0.5")
