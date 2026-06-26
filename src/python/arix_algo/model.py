from .arix_algo_core import _Model, _ArchConfig as ArchConfig
from .tensor import Tensor
import numpy as np


class Model:
    def __init__(self, config=None):
        cfg = ArchConfig()
        cfg.input_dim = 16
        cfg.output_dim = 16
        if config:
            for k in ('input_dim', 'output_dim', 'seed', 
                       'hss_state_dim', 'hss_num_layers',
                       'hss_input_dim', 'hss_output_dim'):
                if k in config:
                    setattr(cfg, k, config[k])
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
        return [Tensor(p) for p in self._model.parameters()]

    def num_params(self):
        return self._model.num_params()
