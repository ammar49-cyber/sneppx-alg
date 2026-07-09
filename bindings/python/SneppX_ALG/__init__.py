"""SneppX-ALG — Python bindings for the cognitive processing system.

Usage:
    from SneppX_ALG import _sneppx_c as ax
    t = ax._Tensor.zeros([4, 8], ax.FLOAT32)
    ax.crypto.sha3_256(b"data")
"""

from . import _SNEPPX_c as _neural_engine_bridge
from ._SNEPPX_c import *
from .interface_bindings import *

__all__ = ["_neural_engine_bridge"]
