import numpy as np


class Tensor:
    def __init__(self, data, dtype='float32'):
        if isinstance(data, np.ndarray):
            self._data = data.astype(np.float32)
        elif isinstance(data, (list, tuple)):
            self._data = np.array(data, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported type: {type(data)}")
        self._shape = list(self._data.shape)

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def ndim(self):
        return len(self._shape)

    def numpy(self):
        return self._data.copy()

    @staticmethod
    def from_numpy(arr):
        return Tensor(arr)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype=float32)"
