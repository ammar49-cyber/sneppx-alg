from .arix_algo_core import _Trainer, _TrainConfig
import numpy as np


class Trainer:
    def __init__(self, model, config=None):
        cfg = _TrainConfig()
        if config:
            cfg.learning_rate = config.get('learning_rate', 0.01)
            cfg.num_epochs = config.get('num_epochs', 10)
            cfg.batch_size = config.get('batch_size', 32)
            cfg.log_interval = config.get('log_interval', 10)
        self._trainer = _Trainer(model._model, cfg)

    def train_step(self, input_data, target_data):
        if isinstance(input_data, (list, tuple)):
            input_data = np.array(input_data, dtype=np.float32)
        if isinstance(target_data, (list, tuple)):
            target_data = np.array(target_data, dtype=np.float32)
        return self._trainer.train_step(input_data, target_data)

    def evaluate(self, input_data, target_data):
        if isinstance(input_data, (list, tuple)):
            input_data = np.array(input_data, dtype=np.float32)
        if isinstance(target_data, (list, tuple)):
            target_data = np.array(target_data, dtype=np.float32)
        return self._trainer.evaluate(input_data, target_data)

    def save(self, path):
        return self._trainer.save(path)

    def load(self, path):
        return self._trainer.load(path)
