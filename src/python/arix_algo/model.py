from .arix_algo_core import _Model, _ArchConfig as ArchConfig
from .tensor import Tensor
import numpy as np


class Model:
    def __init__(self, config=None):
        if config is None:
            cfg = ArchConfig()
            cfg.input_dim = 16
            cfg.output_dim = 16
        else:
            cfg = ArchConfig()
            cfg.input_dim = config.get('input_dim', 16)
            cfg.output_dim = config.get('output_dim', 16)
        self._model = _Model(cfg)

    def forward(self, input):
        if isinstance(input, Tensor):
            arr = input.numpy()
        elif isinstance(input, np.ndarray):
            arr = input.astype(np.float32)
        else:
            arr = np.array(input, dtype=np.float32)
        result = self._model.forward(arr)
        if result is None:
            return None
        return Tensor(result)

    def parameters(self):
        return []
