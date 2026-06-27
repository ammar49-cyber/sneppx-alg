from .arix_algo_core import _Model, _ArchConfig as ArchConfig
from .arix_algo_core import _ARCLayer, _ARCConfig
from .arix_algo_core import _NPEVM, _NPEConfig
from .arix_algo_core import _FMController, _FMConfig
from .arix_algo_core import _SERModel, _SERConfig
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
        self._training = True

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

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def __call__(self, input):
        return self.forward(input)


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = Tensor.randn(out_features, in_features)
        self.bias = Tensor.randn(out_features) if bias else None

    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def __call__(self, x):
        return self.forward(x)


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()

    def __call__(self, x):
        return self.forward(x)
