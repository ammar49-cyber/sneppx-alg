"""
Neural Network Module — SKELETON
Version: v0.5

High-level neural network layer abstractions.
"""

from typing import Callable, List, Optional
from .core import Tensor


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        self._parameters: List[Tensor] = []
        self._training = True

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        return self._parameters

    def train(self):
        self._training = True

    def eval(self):
        self._training = False


class Linear(Module):
    """Fully connected layer (v0.5)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self._in = in_features
        self._out = out_features
        self._use_bias = bias

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("v0.5")


class ReLU(Module):
    """Rectified linear unit (v0.5)."""

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("v0.5")


class Sequential(Module):
    """Sequential container of modules (v0.5)."""

    def __init__(self, *modules: Module):
        super().__init__()
        self._modules = list(modules)

    def forward(self, x: Tensor) -> Tensor:
        for m in self._modules:
            x = m(x)
        return x
