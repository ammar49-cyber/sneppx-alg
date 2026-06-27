from .arix_algo_core import _Trainer, _TrainConfig, _OptimizerConfig, _Optimizer, _OptimizerType
from .arix_algo_core import _optimizer_config_default
import numpy as np


OPTIMIZER_MAP = {
    'sgd': _OptimizerType.SGD,
    'adam': _OptimizerType.ADAM,
    'adamw': _OptimizerType.ADAMW,
    'adamax': _OptimizerType.ADAMAX,
    'rmsprop': _OptimizerType.RMSPROP,
    'adagrad': _OptimizerType.ADAGRAD,
    'adadelta': _OptimizerType.ADADELTA,
}


class Optimizer:
    def __init__(self, params, lr=0.01, optimizer_type='adam', **kwargs):
        self.params = list(params)
        cfg = _optimizer_config_default()
        cfg.learning_rate = lr
        cfg.type = OPTIMIZER_MAP.get(optimizer_type.lower(), _OptimizerType.ADAM)
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        self._opt = _Optimizer(cfg)
        self._lr = lr

    def step(self):
        param_arrays = [p.numpy() for p in self.params]
        if not param_arrays:
            return
        zero_grads = [np.zeros_like(p) for p in param_arrays]
        self._opt.step(param_arrays, zero_grads)
        for i, p in enumerate(self.params):
            p._data[:] = param_arrays[i]

    def zero_grad(self):
        pass


class Trainer:
    def __init__(self, model, config=None):
        cfg = _TrainConfig()
        if config:
            cfg.learning_rate = config.get('learning_rate', 0.01)
            cfg.num_epochs = config.get('num_epochs', 10)
            cfg.batch_size = config.get('batch_size', 32)
            cfg.log_interval = config.get('log_interval', 10)
        self._trainer = _Trainer(model._model, cfg) if hasattr(model, '_model') else None
        self._model = model
        self._config = config or {}

    def train_step(self, input_data, target_data):
        if isinstance(input_data, (list, tuple)):
            input_data = np.array(input_data, dtype=np.float32)
        if isinstance(target_data, (list, tuple)):
            target_data = np.array(target_data, dtype=np.float32)
        if self._trainer:
            return self._trainer.train_step(input_data, target_data)
        return 0.0

    def evaluate(self, input_data, target_data):
        if isinstance(input_data, (list, tuple)):
            input_data = np.array(input_data, dtype=np.float32)
        if isinstance(target_data, (list, tuple)):
            target_data = np.array(target_data, dtype=np.float32)
        if self._trainer:
            return self._trainer.evaluate(input_data, target_data)
        return 0.0

    def save(self, path):
        if self._trainer:
            return self._trainer.save(path)
        return False

    def load(self, path):
        if self._trainer:
            return self._trainer.load(path)
        return False

    @property
    def learning_rate(self):
        return self._trainer.learning_rate if self._trainer else 0.0
