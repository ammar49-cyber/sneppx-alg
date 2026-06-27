import numpy as np


class Device:
    CPU = 'cpu'
    CUDA = 'cuda'


class Tensor:
    def __init__(self, data, dtype='float32', device=Device.CPU):
        if isinstance(data, np.ndarray):
            self._data = data.astype(np.float32)
        elif isinstance(data, (list, tuple)):
            self._data = np.array(data, dtype=np.float32)
        elif isinstance(data, Tensor):
            self._data = data._data.copy()
        else:
            raise TypeError(f"Unsupported type: {type(data)}")
        self._shape = list(self._data.shape)
        self._device = device

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return 'float32'

    def numpy(self):
        return self._data.copy()

    def item(self):
        return float(self._data.item())

    def to(self, device):
        return Tensor(self._data, device=device)

    def cpu(self):
        return Tensor(self._data)

    def cuda(self):
        return Tensor(self._data, device=Device.CUDA)

    @staticmethod
    def from_numpy(arr):
        return Tensor(arr)

    @staticmethod
    def zeros(*shape):
        return Tensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def ones(*shape):
        return Tensor(np.ones(shape, dtype=np.float32))

    @staticmethod
    def randn(*shape):
        return Tensor(np.random.randn(*shape).astype(np.float32))

    @staticmethod
    def arange(start, stop=None, step=1):
        return Tensor(np.arange(start, stop, step, dtype=np.float32))

    def reshape(self, *shape):
        return Tensor(self._data.reshape(shape))

    def transpose(self, *axes):
        return Tensor(self._data.transpose(*axes))

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self._data + other)
        return Tensor(self._data + (other._data if isinstance(other, Tensor) else other))

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self._data - other)
        return Tensor(self._data - (other._data if isinstance(other, Tensor) else other))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self._data * other)
        return Tensor(self._data * (other._data if isinstance(other, Tensor) else other))

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self._data / other)
        return Tensor(self._data / (other._data if isinstance(other, Tensor) else other))

    def __matmul__(self, other):
        return Tensor(self._data @ (other._data if isinstance(other, Tensor) else other))

    def __neg__(self):
        return Tensor(-self._data)

    def __pow__(self, exp):
        return Tensor(self._data ** exp)

    def __getitem__(self, key):
        return Tensor(self._data[key])

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype=float32, device={self._device})"
