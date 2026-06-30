from typing import Optional, Callable, List, Tuple, Iterator
from . import _neural_engine_bridge
from .tensor import Tensor, Device
from .model import Model


class TrainConfig:
    def __init__(self):
        self._c = _neural_engine_bridge.ArixTrainConfig()
        self._c.default()

    @property
    def num_epochs(self):
        return self._c.num_epochs
    @num_epochs.setter
    def num_epochs(self, v):
        self._c.num_epochs = v

    @property
    def batch_size(self):
        return self._c.batch_size
    @batch_size.setter
    def batch_size(self, v):
        self._c.batch_size = v

    @property
    def learning_rate(self):
        return self._c.learning_rate
    @learning_rate.setter
    def learning_rate(self, v):
        self._c.learning_rate = v

    @property
    def log_interval(self):
        return self._c.log_interval
    @log_interval.setter
    def log_interval(self, v):
        self._c.log_interval = v

    @property
    def save_interval(self):
        return self._c.save_interval
    @save_interval.setter
    def save_interval(self, v):
        self._c.save_interval = v

    @property
    def device(self):
        return self._c.device
    @device.setter
    def device(self, v):
        if isinstance(v, int):
            v = Device(v)
        self._c.device = v


class Optimizer:
    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self._params = params
        self._grads = [Tensor.zeros(p.shape) for p in params]

    def zero_grad(self):
        for g in self._grads:
            g.fill_(0.0)

    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.01, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        super().__init__(params, lr)
        self._opt = _neural_engine_bridge._Optimizer.sgd_create(lr, momentum, weight_decay)

    def step(self):
        self._opt.step(self._params, self._grads)


class Adam(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self._opt = _neural_engine_bridge._Optimizer.adam_create(lr, betas[0], betas[1], eps, weight_decay)

    def step(self):
        self._opt.step(self._params, self._grads)


class AdamW(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(params, lr)
        self._opt = _neural_engine_bridge._Optimizer.adamw_create(lr, betas[0], betas[1], eps, weight_decay)

    def step(self):
        self._opt.step(self._params, self._grads)


class LRScheduler:
    def __init__(self, optimizer: Optimizer, *args, **kwargs):
        self._optimizer = optimizer

    def step(self, current_loss: float = 0.0):
        pass


class StepLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self._sched = _neural_engine_bridge._LRScheduler.step_lr(0.01, gamma, step_size)

    def step(self, current_loss: float = 0.0):
        self._sched.step()


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, gamma: float = 0.9):
        super().__init__(optimizer)
        self._sched = _neural_engine_bridge._LRScheduler.exponential(0.01, gamma)

    def step(self, current_loss: float = 0.0):
        self._sched.step()


class CosineLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, min_lr: float = 0.0, max_lr: float = 0.01,
                 total_steps: int = 100):
        super().__init__(optimizer)
        self._sched = _neural_engine_bridge._LRScheduler.cosine(0.01, min_lr, max_lr, total_steps)

    def step(self, current_loss: float = 0.0):
        self._sched.step()


class ReduceLROnPlateau(LRScheduler):
    def __init__(self, optimizer: Optimizer, factor: float = 0.5, patience: int = 10,
                 mode: str = 'min'):
        super().__init__(optimizer)
        mode_min = 1 if mode == 'min' else 0
        self._sched = _neural_engine_bridge._LRScheduler.reduce_on_plateau(0.01, factor, patience, mode_min)

    def step(self, current_loss: float = 0.0):
        self._sched.step(current_loss)


# ---- Loss functions ----
class MSELoss:
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return pred.mse_loss(target)


class CrossEntropyLoss:
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return pred.cross_entropy(target)


class MAELoss:
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return pred.mae_loss(target)


class NLLLoss:
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return pred.nll_loss(target)


class KLDivLoss:
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return pred.kl_div(target)


class BCELoss:
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return pred.binary_cross_entropy(target)


# ---- Dataset ----
class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors
        if tensors:
            self._len = len(tensors[0])
        else:
            self._len = 0

    def __getitem__(self, idx):
        return tuple(t[idx] if isinstance(t, Tensor) else t for t in self.tensors)

    def __len__(self):
        return self._len


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        import numpy as np
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = self.dataset[batch_indices[0]]
            if isinstance(batch, tuple):
                batches = [[] for _ in range(len(batch))]
                for i in batch_indices:
                    item = self.dataset[i]
                    for j in range(len(item)):
                        batches[j].append(item[j] if isinstance(item[j], Tensor)
                                          else Tensor.from_numpy(item[j]))
                yield tuple(Tensor.concat(b) if b else Tensor.zeros((0,)) for b in batches)
            else:
                items = [self.dataset[i] if isinstance(self.dataset[i], Tensor)
                         else Tensor.from_numpy(self.dataset[i]) for i in batch_indices]
                yield Tensor.concat(items)


# ---- Trainer ----
class Trainer:
    def __init__(self, model: Model, config: Optional[TrainConfig] = None):
        if config is None:
            config = TrainConfig()
        self._config = config
        self._model = model
        self._trainer = _neural_engine_bridge._Trainer.create(model, config._c)
        self._optimizer = None
        self._loss_fn = None

    def train_step(self, input: Tensor, target: Tensor) -> float:
        return float(self._trainer.train_step(input._t, target._t))

    def evaluate(self, val_input: Tensor, val_target: Tensor) -> float:
        return float(self._trainer.evaluate(val_input._t, val_target._t))

    def save_checkpoint(self, path: str):
        self._trainer.save_checkpoint(path)

    def load_checkpoint(self, path: str):
        self._trainer.load_checkpoint(path)

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
            epochs: Optional[int] = None):
        if epochs is not None:
            self._config.num_epochs = epochs
        epochs = self._config.num_epochs

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                if isinstance(batch, tuple):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, batch

                loss = self.train_step(inputs, targets)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)

            if val_loader is not None:
                val_loss = 0.0
                val_batches = 0
                for batch in val_loader:
                    if isinstance(batch, tuple):
                        inputs, targets = batch[0], batch[1]
                    else:
                        inputs, targets = batch, batch
                    val_loss += self.evaluate(inputs, targets)
                    val_batches += 1
                avg_val_loss = val_loss / max(val_batches, 1)

            if (epoch + 1) % self._config.log_interval == 0:
                msg = f"Epoch {epoch + 1}/{epochs}  train_loss={avg_loss:.6f}"
                if val_loader is not None:
                    msg += f"  val_loss={avg_val_loss:.6f}"
                print(msg)

            if (epoch + 1) % self._config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.ckpt")
