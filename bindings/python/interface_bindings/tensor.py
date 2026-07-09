import numpy as np
from . import _neural_engine_bridge
from typing import List, Tuple, Optional, Union, Sequence

Dtype = _neural_engine_bridge.SNEPPXDtype
Device = _neural_engine_bridge.SNEPPXDevice
Layout = _neural_engine_bridge.SNEPPXLayout

_NP_TO_SNEPPX = {
    np.dtype('float32'): Dtype.FLOAT32,
    np.dtype('float64'): Dtype.FLOAT64,
    np.dtype('float16'): Dtype.FLOAT16,
    np.dtype('int32'): Dtype.INT32,
    np.dtype('int64'): Dtype.INT64,
    np.dtype('int16'): Dtype.INT16,
    np.dtype('int8'): Dtype.INT8,
    np.dtype('uint8'): Dtype.UINT8,
    np.dtype('bool'): Dtype.BOOL,
    np.dtype('complex64'): Dtype.COMPLEX64,
    np.dtype('complex128'): Dtype.COMPLEX128,
}

_SNEPPX_TO_NP = {v: k for k, v in _NP_TO_SNEPPX.items()}

def _resolve_dtype(dtype) -> Dtype:
    if isinstance(dtype, Dtype):
        return dtype
    if isinstance(dtype, np.dtype):
        return _NP_TO_SNEPPX.get(dtype, Dtype.FLOAT32)
    if isinstance(dtype, str):
        return _NP_TO_SNEPPX.get(np.dtype(dtype), Dtype.FLOAT32)
    if dtype is None:
        return Dtype.FLOAT32
    return Dtype.FLOAT32

def _to_shape(shape) -> np.ndarray:
    if isinstance(shape, np.ndarray):
        return shape.astype(np.uint64)
    return np.array(shape, dtype=np.uint64)


class Tensor:
    __slots__ = ('_t',)

    def __init__(self):
        self._t = _neural_engine_bridge._Tensor()

    @classmethod
    def _from_ptr(cls, ptr):
        t = cls.__new__(cls)
        t._t = ptr
        return t

    def _to_tensor_ptr(self):
        return self._t._to_tensor_ptr()

    # ---- Factory methods ----
    @classmethod
    def empty(cls, shape, dtype=None):
        return cls._from_ptr(_neural_engine_bridge._Tensor.empty(_to_shape(shape), _resolve_dtype(dtype)))

    @classmethod
    def zeros(cls, shape, dtype=None):
        return cls._from_ptr(_neural_engine_bridge._Tensor.zeros(_to_shape(shape), _resolve_dtype(dtype)))

    @classmethod
    def ones(cls, shape, dtype=None):
        return cls._from_ptr(_neural_engine_bridge._Tensor.ones(_to_shape(shape), _resolve_dtype(dtype)))

    @classmethod
    def full(cls, shape, value, dtype=None):
        return cls._from_ptr(_neural_engine_bridge._Tensor.full(_to_shape(shape), _resolve_dtype(dtype), value))

    @classmethod
    def arange(cls, start, stop, step=1.0, dtype=None):
        return cls._from_ptr(_neural_engine_bridge._Tensor.arange(float(start), float(stop), float(step), _resolve_dtype(dtype)))

    @classmethod
    def linspace(cls, start, stop, steps, dtype=None):
        return cls._from_ptr(_neural_engine_bridge._Tensor.linspace(float(start), float(stop), int(steps), _resolve_dtype(dtype)))

    @classmethod
    def eye(cls, n, dtype=None):
        return cls._from_ptr(_neural_engine_bridge._Tensor.eye(int(n), _resolve_dtype(dtype)))

    @classmethod
    def randn(cls, shape, dtype=None):
        return cls._from_ptr(_neural_engine_bridge._Tensor.randn(_to_shape(shape), _resolve_dtype(dtype)))

    @classmethod
    def from_numpy(cls, arr: np.ndarray):
        arr = np.ascontiguousarray(arr)
        dtype = _resolve_dtype(arr.dtype)
        t = cls.empty(arr.shape, dtype)
        t._t.set_data(arr)
        return t

    @classmethod
    def load(cls, path: str):
        return cls._from_ptr(_neural_engine_bridge._Tensor.load(path))

    # ---- Properties ----
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(s) for s in self._t.shape)

    @property
    def ndim(self) -> int:
        return int(self._t.ndim)

    @property
    def dtype(self) -> Dtype:
        return self._t.dtype

    @property
    def size(self) -> int:
        return int(self._t.size_)

    @property
    def device(self) -> Device:
        return self._t.device

    @property
    def layout(self) -> Layout:
        return self._t.layout

    @property
    def dtype_name(self) -> str:
        return self._t.dtype_name

    @property
    def numel(self) -> int:
        return int(self._t.numel)

    @property
    def is_contiguous(self) -> bool:
        return bool(self._t.is_contiguous)

    @property
    def data(self) -> np.ndarray:
        return self._t.data()

    @data.setter
    def data(self, arr: np.ndarray):
        self._t.set_data(np.ascontiguousarray(arr))

    @property
    def T(self):
        if self.ndim < 2:
            return self
        return self.transpose(0, 1)

    # ---- Item access ----
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self._t.__getitem__(*idx)
        return self._t.__getitem__(idx)

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple):
            self._t.__setitem__(*idx, value)
        else:
            self._t.__setitem__(idx, value)

    def __repr__(self):
        return self._t.__repr__()

    def __len__(self):
        return int(self._t.shape[0]) if self.ndim > 0 else 0

    # ---- Fill ----
    def fill_(self, value):
        if self.dtype == Dtype.FLOAT64:
            self._t.fill_f64(float(value))
        else:
            self._t.fill_f32(float(value))
        return self

    # ---- Unary math ----
    def neg(self):
        return Tensor._from_ptr(self._t.neg())

    def abs(self):
        return Tensor._from_ptr(self._t.abs())

    def sign(self):
        return Tensor._from_ptr(self._t.sign())

    def floor(self):
        return Tensor._from_ptr(self._t.floor())

    def ceil(self):
        return Tensor._from_ptr(self._t.ceil())

    def round(self):
        return Tensor._from_ptr(self._t.round())

    def trunc(self):
        return Tensor._from_ptr(self._t.trunc())

    def exp(self):
        return Tensor._from_ptr(self._t.exp())

    def log(self):
        return Tensor._from_ptr(self._t.log())

    def sqrt(self):
        return Tensor._from_ptr(self._t.sqrt())

    def sin(self):
        return Tensor._from_ptr(self._t.sin())

    def cos(self):
        return Tensor._from_ptr(self._t.cos())

    def tan(self):
        return Tensor._from_ptr(self._t.tan())

    def asin(self):
        return Tensor._from_ptr(self._t.asin())

    def acos(self):
        return Tensor._from_ptr(self._t.acos())

    def atan(self):
        return Tensor._from_ptr(self._t.atan())

    def sinh(self):
        return Tensor._from_ptr(self._t.sinh())

    def cosh(self):
        return Tensor._from_ptr(self._t.cosh())

    def tanh(self):
        return Tensor._from_ptr(self._t.tanh())

    # ---- Arithmetic operators ----
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor.full(self.shape, other, self.dtype)
        return Tensor._from_ptr(self._t.__add__(other._t))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor.full(self.shape, other, self.dtype)
        return Tensor._from_ptr(self._t.__sub__(other._t))

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor.full(self.shape, other, self.dtype)
        return Tensor._from_ptr(other._t.__sub__(self._t))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor.full(self.shape, other, self.dtype)
        return Tensor._from_ptr(self._t.__mul__(other._t))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor.full(self.shape, other, self.dtype)
        return Tensor._from_ptr(self._t.__truediv__(other._t))

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor.full(self.shape, other, self.dtype)
        return Tensor._from_ptr(other._t.__truediv__(self._t))

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor.full(self.shape, other, self.dtype)
        return Tensor._from_ptr(self._t.__pow__(other._t))

    def __neg__(self):
        return Tensor._from_ptr(self._t.__neg__())

    # ---- Comparison ----
    def __eq__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor.full(self.shape, other, self.dtype)
        return Tensor._from_ptr(self._t.__eq__(other._t))

    def __ne__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor.full(self.shape, other, self.dtype)
        return Tensor._from_ptr(self._t.__ne__(other._t))

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor.full(self.shape, other, self.dtype)
        return Tensor._from_ptr(self._t.__lt__(other._t))

    def __le__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor.full(self.shape, other, self.dtype)
        return Tensor._from_ptr(self._t.__le__(other._t))

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor.full(self.shape, other, self.dtype)
        return Tensor._from_ptr(self._t.__gt__(other._t))

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor.full(self.shape, other, self.dtype)
        return Tensor._from_ptr(self._t.__ge__(other._t))

    # ---- Activations ----
    def relu(self):
        return Tensor._from_ptr(self._t.relu())

    def gelu(self):
        return Tensor._from_ptr(self._t.gelu())

    def silu(self):
        return Tensor._from_ptr(self._t.silu())

    def sigmoid(self):
        return Tensor._from_ptr(self._t.sigmoid())

    def softmax(self, dim=-1):
        if dim < 0:
            dim = self.ndim + dim
        return Tensor._from_ptr(self._t.softmax(dim))

    def log_softmax(self, dim=-1):
        if dim < 0:
            dim = self.ndim + dim
        return Tensor._from_ptr(self._t.log_softmax(dim))

    def tanh_act(self):
        return Tensor._from_ptr(self._t.tanh_act())

    # ---- Reductions ----
    def sum(self, dim=None):
        if dim is None:
            dim = 0
        return Tensor._from_ptr(self._t.sum(dim))

    def mean(self, dim=None):
        if dim is None:
            dim = 0
        return Tensor._from_ptr(self._t.mean(dim))

    def var(self, dim=None):
        if dim is None:
            dim = 0
        return Tensor._from_ptr(self._t.var(dim))

    def std(self, dim=None):
        if dim is None:
            dim = 0
        return Tensor._from_ptr(self._t.std(dim))

    def min(self):
        return float(self._t.min())

    def max(self):
        return float(self._t.max())

    def argmin(self):
        return int(self._t.argmin())

    def argmax(self):
        return int(self._t.argmax())

    def cumsum(self, dim=0):
        return Tensor._from_ptr(self._t.cumsum(dim))

    def cumprod(self, dim=0):
        return Tensor._from_ptr(self._t.cumprod(dim))

    # ---- Linear algebra ----
    def dot(self, other):
        if not isinstance(other, Tensor):
            other = Tensor.from_numpy(np.asarray(other))
        return float(self._t.dot(other._t))

    def matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor.from_numpy(np.asarray(other))
        return Tensor._from_ptr(self._t.matmul(other._t))

    def transpose(self, dim1=0, dim2=1):
        return Tensor._from_ptr(self._t.transpose(dim1, dim2))

    def inverse(self):
        return Tensor._from_ptr(self._t.inverse())

    def det(self):
        return float(self._t.det())

    # ---- Shape ops ----
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, np.ndarray)):
            shape = tuple(shape[0])
        return Tensor._from_ptr(self._t.reshape(_to_shape(shape)))

    def permute(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list, np.ndarray)):
            axes = tuple(axes[0])
        return Tensor._from_ptr(self._t.permute(_to_shape(axes)))

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, np.ndarray)):
            shape = tuple(shape[0])
        return Tensor._from_ptr(self._t.expand(_to_shape(shape)))

    def squeeze(self, dim=None):
        if dim is None:
            for d in range(self.ndim):
                if self.shape[d] == 1:
                    return Tensor._from_ptr(self._t.squeeze(d))
            return self
        return Tensor._from_ptr(self._t.squeeze(dim))

    def unsqueeze(self, dim):
        return Tensor._from_ptr(self._t.unsqueeze(dim))

    def slice(self, dim, start, end):
        return Tensor._from_ptr(self._t.slice(dim, start, end))

    @staticmethod
    def concat(tensors, dim=0):
        if not tensors:
            raise ValueError("empty tensor list")
        tlist = [t._t for t in tensors]
        return Tensor._from_ptr(_neural_engine_bridge._Tensor.concat(tlist, dim))

    def split(self, num_splits, dim=0):
        splits = self._t.split(num_splits, dim)
        return [Tensor._from_ptr(s) for s in splits]

    def tile(self, *reps):
        if len(reps) == 1 and isinstance(reps[0], (tuple, list, np.ndarray)):
            reps = tuple(reps[0])
        return Tensor._from_ptr(self._t.tile(_to_shape(reps)))

    def repeat(self, repeats, dim=0):
        return Tensor._from_ptr(self._t.repeat(repeats, dim))

    def gather(self, dim, indices):
        if isinstance(indices, np.ndarray):
            indices = Tensor.from_numpy(indices)
        return Tensor._from_ptr(self._t.gather(dim, indices._t))

    def scatter(self, dim, indices, src):
        if isinstance(indices, np.ndarray):
            indices = Tensor.from_numpy(indices)
        if isinstance(src, np.ndarray):
            src = Tensor.from_numpy(src)
        return Tensor._from_ptr(self._t.scatter(dim, indices._t, src._t))

    def masked_select(self, mask):
        if isinstance(mask, np.ndarray):
            mask = Tensor.from_numpy(mask)
        return Tensor._from_ptr(self._t.masked_select(mask._t))

    def masked_fill(self, mask, value):
        if isinstance(mask, np.ndarray):
            mask = Tensor.from_numpy(mask)
        return Tensor._from_ptr(self._t.masked_fill(mask._t, value))

    @staticmethod
    def where(condition, x, y):
        if isinstance(condition, np.ndarray):
            condition = Tensor.from_numpy(condition)
        if isinstance(x, np.ndarray):
            x = Tensor.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = Tensor.from_numpy(y)
        return Tensor._from_ptr(_neural_engine_bridge._Tensor.where(condition._t, x._t, y._t))

    # ---- Cast / Device / Layout ----
    def cast(self, dtype):
        return Tensor._from_ptr(self._t.cast(_resolve_dtype(dtype)))

    def to_device(self, device):
        if isinstance(device, int):
            device = Device(device)
        return Tensor._from_ptr(self._t.to_device(device))

    def to_layout(self, layout):
        if isinstance(layout, int):
            layout = Layout(layout)
        return Tensor._from_ptr(self._t.to_layout(layout))

    # ---- I/O ----
    def save(self, path: str):
        self._t.save(path)

    def copy(self):
        return Tensor._from_ptr(self._t.copy())

    def clone(self):
        return Tensor._from_ptr(self._t.clone())

    # ---- NN ops ----
    def conv1d(self, kernel, stride=1, padding=0):
        if isinstance(kernel, np.ndarray):
            kernel = Tensor.from_numpy(kernel)
        return Tensor._from_ptr(self._t.conv1d(kernel._t, stride, padding))

    def conv2d(self, kernel, stride_h=1, stride_w=1, pad_h=0, pad_w=0):
        if isinstance(kernel, np.ndarray):
            kernel = Tensor.from_numpy(kernel)
        return Tensor._from_ptr(self._t.conv2d(kernel._t, stride_h, stride_w, pad_h, pad_w))

    def pool1d(self, kernel_size, stride=None):
        stride = stride or kernel_size
        return Tensor._from_ptr(self._t.pool1d(kernel_size, stride))

    def pool2d(self, kernel_h, kernel_w, stride_h=None, stride_w=None):
        stride_h = stride_h or kernel_h
        stride_w = stride_w or kernel_w
        return Tensor._from_ptr(self._t.pool2d(kernel_h, kernel_w, stride_h, stride_w))

    def dropout(self, rate=0.5, seed=42):
        return Tensor._from_ptr(self._t.dropout(rate, seed))

    def layer_norm(self, gamma, beta, eps=1e-5):
        if isinstance(gamma, np.ndarray):
            gamma = Tensor.from_numpy(gamma)
        if isinstance(beta, np.ndarray):
            beta = Tensor.from_numpy(beta)
        return Tensor._from_ptr(self._t.layer_norm(gamma._t, beta._t, eps))

    def batch_norm(self, gamma, beta, running_mean, running_var, eps=1e-5):
        if isinstance(gamma, np.ndarray): gamma = Tensor.from_numpy(gamma)
        if isinstance(beta, np.ndarray): beta = Tensor.from_numpy(beta)
        if isinstance(running_mean, np.ndarray): running_mean = Tensor.from_numpy(running_mean)
        if isinstance(running_var, np.ndarray): running_var = Tensor.from_numpy(running_var)
        return Tensor._from_ptr(self._t.batch_norm(gamma._t, beta._t, running_mean._t, running_var._t, eps))

    def group_norm(self, gamma, beta, num_groups, eps=1e-5):
        if isinstance(gamma, np.ndarray): gamma = Tensor.from_numpy(gamma)
        if isinstance(beta, np.ndarray): beta = Tensor.from_numpy(beta)
        return Tensor._from_ptr(self._t.group_norm(gamma._t, beta._t, num_groups, eps))

    def instance_norm(self, gamma, beta, eps=1e-5):
        if isinstance(gamma, np.ndarray): gamma = Tensor.from_numpy(gamma)
        if isinstance(beta, np.ndarray): beta = Tensor.from_numpy(beta)
        return Tensor._from_ptr(self._t.instance_norm(gamma._t, beta._t, eps))

    def embedding(self, indices):
        if isinstance(indices, np.ndarray):
            indices = Tensor.from_numpy(indices)
        return Tensor._from_ptr(self._t.embedding(indices._t))

    # ---- Loss ----
    def mse_loss(self, target):
        if isinstance(target, np.ndarray):
            target = Tensor.from_numpy(target)
        return Tensor._from_ptr(self._t.mse_loss(target._t))

    def cross_entropy(self, target):
        if isinstance(target, np.ndarray):
            target = Tensor.from_numpy(target)
        return Tensor._from_ptr(self._t.cross_entropy(target._t))

    def mae_loss(self, target):
        if isinstance(target, np.ndarray):
            target = Tensor.from_numpy(target)
        return Tensor._from_ptr(self._t.mae_loss(target._t))

    def nll_loss(self, target):
        if isinstance(target, np.ndarray):
            target = Tensor.from_numpy(target)
        return Tensor._from_ptr(self._t.nll_loss(target._t))

    def kl_div(self, target):
        if isinstance(target, np.ndarray):
            target = Tensor.from_numpy(target)
        return Tensor._from_ptr(self._t.kl_div(target._t))

    def binary_cross_entropy(self, target):
        if isinstance(target, np.ndarray):
            target = Tensor.from_numpy(target)
        return Tensor._from_ptr(self._t.binary_cross_entropy(target._t))


# Union type for functions that accept either Tensor or ndarray
Tensorable = Union[Tensor, np.ndarray]
