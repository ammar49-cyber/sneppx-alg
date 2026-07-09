"""SneppX-ALG — Python bindings for the cognitive processing system.

Usage:
    from SneppX_ALG import _SNEPPX_c as ax
    t = ax._Tensor.zeros([4, 8], ax.FLOAT32)
    ax.crypto.sha3_256(b"data")
"""

from SneppX_ALG._SNEPPX_c import *

__all__ = []
