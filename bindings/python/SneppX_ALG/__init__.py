"""SneppX-ALG — Python bindings for the cognitive processing system.

Usage:
    from SneppX_ALG import Tensor, Linear, AdamW, Trainer
    from SneppX_ALG import _neural_engine_bridge as ax

    t = Tensor.zeros([4, 8], ax.SNEPPXDtype.FLOAT32)
"""

try:
    from . import _arix_c as _neural_engine_bridge
except ImportError:
    try:
        from . import _SNEPPX_c as _neural_engine_bridge
    except ImportError:
        import types
        _neural_engine_bridge = types.ModuleType("_neural_engine_bridge_fallback")

from .interface_bindings import *

__all__ = ["_neural_engine_bridge"]
