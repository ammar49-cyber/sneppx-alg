"""
ARIX-Algo Python Library

This path is deprecated. Use bindings/python/ instead:
    from arix_algo import _arix_c as ax
"""

import warnings
warnings.warn(
    "lib/python/ is deprecated. Use 'from arix_algo import _arix_c' from bindings/python/.",
    DeprecationWarning,
    stacklevel=2,
)
