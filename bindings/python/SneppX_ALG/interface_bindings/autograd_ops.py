"""Autograd Ops � differentiable operation subclasses of Function.

Each op defines forward(ctx, *args) -> Tensor and the corresponding
backward(ctx, grad_output) -> list of gradients.

Consumed by tensor.py via lazy imports inside the op methods.
"""

import numpy as np

from .tensor import Tensor, _numpy_dtype, _resolve_dtype
from .autograd import Function, Context

# ===========================================================================
#  Arithmetic Ops
# ===========================================================================


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        if isinstance(b, (int, float)):
            return Tensor(a.data + b, dtype=a.dtype)
        return Tensor(a.data + b.data, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return [grad_output, grad_output]


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        if isinstance(b, (int, float)):
            return Tensor(a.data - b, dtype=a.dtype)
        return Tensor(a.data - b.data, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return [grad_output, -grad_output]


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        if isinstance(b, (int, float)):
            ctx.save_attr(b_val=b)
            return Tensor(a.data * b, dtype=a.dtype)
        ctx.save_for_backward(a=a, b=b)
        return Tensor(a.data * b.data, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        b_val = ctx.get_attr("b_val")
        if b_val is not None:
            return [grad_output * b_val, None]
        a = ctx.get_saved_tensor("a")
        b = ctx.get_saved_tensor("b")
        return [grad_output * b.data, grad_output * a.data]


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        if isinstance(b, (int, float)):
            ctx.save_attr(b_val=b)
            return Tensor(a.data / b, dtype=a.dtype)
        ctx.save_for_backward(a=a, b=b)
        return Tensor(a.data / b.data, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        b_val = ctx.get_attr("b_val")
        if b_val is not None:
            return [grad_output / b_val, None]
        a = ctx.get_saved_tensor("a")
        b = ctx.get_saved_tensor("b")
        g = grad_output.data
        return [
            Tensor(g / b.data, dtype=a.dtype),
            Tensor(-g * a.data / (b.data**2), dtype=b.dtype),
        ]


class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        return Tensor(-a.data, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return [-grad_output]


class Pow(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a)
        ctx.save_attr(b_val=b)
        return Tensor(a.data**b, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.get_saved_tensor("a")
        b = ctx.get_attr("b_val")
        return [Tensor(b * (a.data ** (b - 1)) * grad_output.data, dtype=a.dtype)]


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return Tensor(a.data @ b.data, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.get_saved_tensor("a")
        b = ctx.get_saved_tensor("b")
        g = grad_output.data
        a_data = a.data
        b_data = b.data
        if a_data.ndim == 1:
            grad_a = g @ b_data.T
            grad_b = np.outer(a_data, g)
        elif b_data.ndim == 1:
            grad_a = np.outer(g, b_data)
            grad_b = a_data.T @ g
        else:
            grad_a = g @ b_data.T
            grad_b = a_data.T @ g
        return [Tensor(grad_a, dtype=a.dtype), Tensor(grad_b, dtype=b.dtype)]


# ===========================================================================
#  Reduction Ops
# ===========================================================================


class Sum(Function):
    @staticmethod
    def forward(ctx, a, dim=None):
        ctx.save_attr(shape=a.shape, dim=dim)
        if dim is None:
            return Tensor(float(a.data.sum()), dtype=a.dtype)
        return Tensor(a.data.sum(axis=dim), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.get_attr("shape")
        dim = ctx.get_attr("dim")
        g = grad_output.data
        if dim is None:
            return [
                Tensor(
                    np.full(
                        shape, g.flatten()[0], dtype=_numpy_dtype(grad_output.dtype)
                    ),
                    dtype=grad_output.dtype,
                )
            ]
        expand_shape = list(shape)
        for d in ([dim] if isinstance(dim, int) else dim):
            expand_shape[d] = 1
        g_br = g.reshape(expand_shape)
        return [
            Tensor(
                np.broadcast_to(g_br, shape).astype(_numpy_dtype(grad_output.dtype)),
                dtype=grad_output.dtype,
            )
        ]


class Mean(Function):
    @staticmethod
    def forward(ctx, a, dim=None):
        ctx.save_attr(shape=a.shape, numel=a.numel, dim=dim)
        if dim is None:
            return Tensor(float(a.data.mean()), dtype=a.dtype)
        return Tensor(a.data.mean(axis=dim), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.get_attr("shape")
        n = ctx.get_attr("numel")
        dim = ctx.get_attr("dim")
        g = grad_output.data
        if dim is None:
            return [
                Tensor(
                    np.full(
                        shape, g.flatten()[0] / n, dtype=_numpy_dtype(grad_output.dtype)
                    ),
                    dtype=grad_output.dtype,
                )
            ]
        dims = [dim] if isinstance(dim, int) else dim
        nelem = 1
        for d in dims:
            nelem *= shape[d]
        expand_shape = list(shape)
        for d in dims:
            expand_shape[d] = 1
        g_br = g.reshape(expand_shape)
        return [
            Tensor(
                np.broadcast_to(g_br, shape).astype(_numpy_dtype(grad_output.dtype))
                / nelem,
                dtype=grad_output.dtype,
            )
        ]


# ===========================================================================
#  Activation Ops
# ===========================================================================


class Relu(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a=a)
        return Tensor(np.maximum(0, a.data), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.get_saved_tensor("a")
        return [
            Tensor(
                (a.data > 0).astype(grad_output.data.dtype) * grad_output.data,
                dtype=grad_output.dtype,
            )
        ]


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        out = 1.0 / (1.0 + np.exp(-a.data))
        ctx.save_attr(out=out)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.get_attr("out")
        return [Tensor(out * (1 - out) * grad_output.data, dtype=grad_output.dtype)]


class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        out = np.tanh(a.data)
        ctx.save_attr(out=out)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.get_attr("out")
        return [Tensor((1 - out**2) * grad_output.data, dtype=grad_output.dtype)]


class Gelu(Function):
    @staticmethod
    def forward(ctx, a):
        x = a.data
        c = 0.79788456
        tanh_arg = c * (x + 0.044715 * x**3)
        out = 0.5 * x * (1.0 + np.tanh(tanh_arg))
        ctx.save_for_backward(a=a)
        ctx.save_attr(out_val=out, tanh_arg=tanh_arg)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.get_saved_tensor("a")
        x = a.data
        tanh_arg = ctx.get_attr("tanh_arg")
        sech2 = 1 - np.tanh(tanh_arg) ** 2
        c = 0.79788456
        grad_x = 0.5 * (1 + np.tanh(tanh_arg)) + 0.5 * x * sech2 * c * (
            1 + 0.134145 * x**2
        )
        return [Tensor(grad_x * grad_output.data, dtype=grad_output.dtype)]


class Silu(Function):
    @staticmethod
    def forward(ctx, a):
        x = a.data
        sig = 1.0 / (1.0 + np.exp(-x))
        out = x * sig
        ctx.save_attr(sig=sig, out=out)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        sig = ctx.get_attr("sig")
        out = ctx.get_attr("out")
        return [
            Tensor((sig + out * (1 - sig)) * grad_output.data, dtype=grad_output.dtype)
        ]


# ===========================================================================
#  Unary Math Ops
# ===========================================================================


class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        out = np.sqrt(a.data)
        ctx.save_attr(out=out)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.get_attr("out")
        return [Tensor(grad_output.data / (2 * out + 1e-10), dtype=grad_output.dtype)]


class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        out = np.exp(a.data)
        ctx.save_attr(out=out)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.get_attr("out")
        return [Tensor(out * grad_output.data, dtype=grad_output.dtype)]


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a=a)
        return Tensor(np.log(a.data + 1e-10), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.get_saved_tensor("a")
        return [Tensor(grad_output.data / (a.data + 1e-10), dtype=grad_output.dtype)]


class Abs(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a=a)
        return Tensor(np.abs(a.data), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.get_saved_tensor("a")
        return [Tensor(np.sign(a.data) * grad_output.data, dtype=grad_output.dtype)]


# ===========================================================================
#  Softmax / LogSoftmax
# ===========================================================================


class Softmax(Function):
    @staticmethod
    def forward(ctx, a, dim=-1):
        x = a.data - a.data.max(axis=dim, keepdims=True)
        e = np.exp(x)
        out = e / e.sum(axis=dim, keepdims=True)
        ctx.save_attr(out=out, dim=dim)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.get_attr("out")
        dim = ctx.get_attr("dim")
        g = grad_output.data
        s = out * (g - (out * g).sum(axis=dim, keepdims=True))
        return [Tensor(s, dtype=grad_output.dtype)]


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, a, dim=-1):
        x = a.data - a.data.max(axis=dim, keepdims=True)
        e = np.exp(x)
        sm = e / e.sum(axis=dim, keepdims=True)
        out = np.log(sm + 1e-10)
        ctx.save_attr(sm=sm, dim=dim)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        sm = ctx.get_attr("sm")
        dim = ctx.get_attr("dim")
        g = grad_output.data
        s = g - sm * g.sum(axis=dim, keepdims=True)
        return [Tensor(s, dtype=grad_output.dtype)]


# ===========================================================================
#  Shape Ops
# ===========================================================================


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.save_attr(orig_shape=a.shape)
        return Tensor(a.data.reshape(shape), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        orig_shape = ctx.get_attr("orig_shape")
        return [Tensor(grad_output.data.reshape(orig_shape), dtype=grad_output.dtype)]


class Transpose(Function):
    @staticmethod
    def forward(ctx, a, dim1=0, dim2=1):
        ctx.save_attr(dim1=dim1, dim2=dim2)
        return Tensor(a.data.swapaxes(dim1, dim2), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        dim1 = ctx.get_attr("dim1")
        dim2 = ctx.get_attr("dim2")
        return [Tensor(grad_output.data.swapaxes(dim1, dim2), dtype=grad_output.dtype)]


class Expand(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.save_attr(orig_shape=a.shape)
        return Tensor(np.broadcast_to(a.data, shape), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        orig_shape = ctx.get_attr("orig_shape")
        g = grad_output.data
        if len(orig_shape) < g.ndim:
            axes = tuple(range(g.ndim - len(orig_shape)))
            g = g.sum(axis=axes)
        axes = tuple(
            i for i in range(len(orig_shape)) if orig_shape[i] == 1 and g.shape[i] > 1
        )
        if axes:
            g = g.sum(axis=axes, keepdims=True)
        return [Tensor(g.reshape(orig_shape), dtype=grad_output.dtype)]


class Squeeze(Function):
    @staticmethod
    def forward(ctx, a, dim=None):
        ctx.save_attr(orig_shape=a.shape, dim=dim)
        if dim is None:
            return Tensor(np.squeeze(a.data), dtype=a.dtype)
        return Tensor(np.squeeze(a.data, axis=dim), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        orig_shape = ctx.get_attr("orig_shape")
        return [Tensor(grad_output.data.reshape(orig_shape), dtype=grad_output.dtype)]


class Unsqueeze(Function):
    @staticmethod
    def forward(ctx, a, dim):
        ctx.save_attr(orig_shape=a.shape)
        return Tensor(np.expand_dims(a.data, dim), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        orig_shape = ctx.get_attr("orig_shape")
        return [Tensor(grad_output.data.reshape(orig_shape), dtype=grad_output.dtype)]


# ===========================================================================
#  Indexing
# ===========================================================================


class GetItem(Function):
    @staticmethod
    def forward(ctx, a, key):
        ctx.save_attr(key=key, orig_shape=a.shape)
        return Tensor(a.data[key], dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        key = ctx.get_attr("key")
        g = np.zeros(ctx.get_attr("orig_shape"), dtype=_numpy_dtype(grad_output.dtype))
        g[key] = grad_output.data
        return [Tensor(g, dtype=grad_output.dtype)]


# ===========================================================================
#  Normalization Ops
# ===========================================================================


class LayerNorm(Function):
    @staticmethod
    def forward(ctx, a, gamma, beta, eps=1e-5):
        x = a.data
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        g = gamma.data if isinstance(gamma, Tensor) else gamma
        b = beta.data if isinstance(beta, Tensor) else beta
        out = x_norm * g + b
        ctx.save_attr(
            mean=mean, var=var, x_norm=x_norm, gamma_val=g, eps=eps, shape=x.shape
        )
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x_norm = ctx.get_attr("x_norm")
        gamma_val = ctx.get_attr("gamma_val")
        shape = ctx.get_attr("shape")
        g = grad_output.data
        n = shape[-1]

        dgamma = (g * x_norm).sum(axis=tuple(range(g.ndim - 1)), keepdims=False)
        dbeta = g.sum(axis=tuple(range(g.ndim - 1)), keepdims=False)
        dx_norm = g * gamma_val
        dx = (
            dx_norm
            - dx_norm.mean(axis=-1, keepdims=True)
            - x_norm * (dx_norm * x_norm).mean(axis=-1, keepdims=True)
        )
        dx = dx / np.sqrt(ctx.get_attr("var") + ctx.get_attr("eps"))
        return [
            Tensor(dx, dtype="float32"),
            Tensor(dgamma, dtype="float32"),
            Tensor(dbeta, dtype="float32"),
        ]


class RMSNorm(Function):
    @staticmethod
    def forward(ctx, a, gamma, eps=1e-6):
        x = a.data
        rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)
        x_norm = x / rms
        g = gamma.data if isinstance(gamma, Tensor) else gamma
        out = x_norm * g
        ctx.save_attr(rms=rms, x_norm=x_norm, gamma_val=g)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.get_saved_tensor("a")
        rms = ctx.get_attr("rms")
        x_norm = ctx.get_attr("x_norm")
        gamma_val = ctx.get_attr("gamma_val")
        g = grad_output.data
        n = a.shape[-1]

        dgamma = (g * x_norm).sum(axis=tuple(range(g.ndim - 1)), keepdims=False)
        dx_norm = g * gamma_val
        drms = -(dx_norm * x_norm / rms).sum(axis=-1, keepdims=True)
        dx = dx_norm / rms + drms * x_norm / (n * rms)
        return [Tensor(dx, dtype=a.dtype), Tensor(dgamma, dtype=a.dtype)]


class DropoutFn(Function):
    @staticmethod
    def forward(ctx, a, rate=0.5, seed=42):
        rng = np.random.RandomState(seed)
        mask = rng.binomial(1, 1.0 - rate, a.shape).astype(np.float32)
        mask /= 1.0 - rate
        ctx.save_attr(mask=mask)
        return Tensor(a.data * mask, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.get_attr("mask")
        return [Tensor(grad_output.data * mask, dtype=grad_output.dtype)]


# ===========================================================================
#  Neural Network Ops
# ===========================================================================


class LinearFn(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias=None):
        ctx.save_for_backward(inp=inp, weight=weight)
        out = inp.data @ weight.data.T
        has_b = bias is not None
        ctx.save_attr(has_bias=has_b)
        if has_b:
            b = bias.data if isinstance(bias, Tensor) else bias
            ctx.save_for_backward(bias=bias)
            out = out + b
        return Tensor(out, dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        inp = ctx.get_saved_tensor("inp")
        w = ctx.get_saved_tensor("weight")
        g = grad_output.data
        grads = [
            Tensor(g @ w.data, dtype=inp.dtype),
            Tensor(g.T @ inp.data, dtype=w.dtype),
        ]
        if ctx.get_attr("has_bias"):
            grads.append(Tensor(g.sum(axis=0), dtype="float32"))
        return grads


class EmbeddingFn(Function):
    @staticmethod
    def forward(ctx, weight, indices):
        idx = (
            indices.data.astype(np.int64)
            if isinstance(indices, Tensor)
            else np.array(indices, dtype=np.int64)
        )
        ctx.save_attr(indices=idx)
        ctx.save_for_backward(weight=weight)
        return Tensor(weight.data[idx], dtype=weight.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        weight = ctx.get_saved_tensor("weight")
        indices = ctx.get_attr("indices")
        g = grad_output.data
        grad_w = np.zeros(
            (weight.shape[0], weight.shape[1]), dtype=_numpy_dtype(weight.dtype)
        )
        for i in range(grad_w.shape[0]):
            mask = indices == i
            if mask.any():
                grad_w[i] = g[mask].sum(axis=0)
        return [Tensor(grad_w, dtype=weight.dtype), None]


# ===========================================================================
#  Convolution & Pooling Ops
# ===========================================================================


class Conv1d(Function):
    @staticmethod
    def forward(ctx, inp, kernel, stride=1, padding=0):
        from scipy import signal

        arr = inp.data
        k = kernel.data if isinstance(kernel, Tensor) else kernel
        ctx.save_attr(stride=stride, padding=padding, k_shape=k.shape)
        ctx.save_for_backward(
            inp=inp, kernel=kernel if isinstance(kernel, Tensor) else Tensor(kernel)
        )
        if padding > 0:
            arr = np.pad(
                arr, [(0, 0), (padding,), (0,)] if arr.ndim == 3 else [(padding,)]
            )
        out = signal.correlate(arr, k, mode="valid")[..., ::stride]
        return Tensor(out, dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        from scipy import signal

        inp = ctx.get_saved_tensor("inp")
        kernel = ctx.get_saved_tensor("kernel")
        stride = ctx.get_attr("stride")
        padding = ctx.get_attr("padding")
        k = kernel.data
        g = grad_output.data

        if stride > 1:
            if g.ndim == 3:
                g_dil = np.zeros(
                    (g.shape[0], 1 + (g.shape[1] - 1) * stride, g.shape[2])
                )
                g_dil[:, ::stride, :] = g
            else:
                g_dil = np.zeros((1 + (g.shape[0] - 1) * stride,))
                g_dil[::stride] = g
            g = g_dil

        if padding > 0:
            grad_pad = np.pad(
                g, [(0, 0), (padding,), (0,)] if g.ndim == 3 else [(padding,)]
            )
        else:
            grad_pad = g
        k_rot = k[..., ::-1, :] if k.ndim == 3 else k[::-1]
        grad_inp = signal.correlate(grad_pad, k_rot, mode="full")
        slices = tuple(slice(0, s) for s in inp.shape)
        grad_inp = grad_inp[slices]

        inp_pad = np.pad(
            inp.data, [(0, 0), (padding,), (0,)] if inp.ndim == 3 else [(padding,)]
        )
        g_flip = g[..., ::-1, :] if g.ndim == 3 else g[::-1]
        grad_k = signal.correlate(inp_pad, g_flip, mode="valid")
        grad_k = grad_k.reshape(k.shape)

        return [Tensor(grad_inp, dtype=inp.dtype), Tensor(grad_k, dtype=kernel.dtype)]


class MaxPool2d(Function):
    @staticmethod
    def forward(ctx, inp, kernel_h, kernel_w, stride_h=None, stride_w=None):
        stride_h = stride_h or kernel_h
        stride_w = stride_w or kernel_w
        arr = inp.data
        ctx.save_attr(k_h=kernel_h, k_w=kernel_w, s_h=stride_h, s_w=stride_w)
        ctx.save_for_backward(inp=inp)
        out = np.array(
            [
                [
                    arr[..., i : i + kernel_h, j : j + kernel_w].max(axis=(-2, -1))
                    for j in range(0, arr.shape[-1] - kernel_w + 1, stride_w)
                ]
                for i in range(0, arr.shape[-2] - kernel_h + 1, stride_h)
            ]
        )
        out = out.transpose(2, 3, 0, 1) if arr.ndim == 4 else out
        return Tensor(out, dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        inp = ctx.get_saved_tensor("inp")
        k_h = ctx.get_attr("k_h")
        k_w = ctx.get_attr("k_w")
        s_h = ctx.get_attr("s_h")
        s_w = ctx.get_attr("s_w")
        arr = inp.data
        g = grad_output.data
        grad = np.zeros_like(arr)
        for i in range(0, arr.shape[-2] - k_h + 1, s_h):
            for j in range(0, arr.shape[-1] - k_w + 1, s_w):
                patch = arr[..., i : i + k_h, j : j + k_w]
                max_val = patch.max(axis=(-2, -1), keepdims=True)
                max_mask = (patch == max_val).astype(g.dtype)
                gi = i // s_h
                gj = j // s_w
                if g.ndim == 4:
                    g_val = g[..., gi, gj, None, None]
                else:
                    g_val = g
                grad[..., i : i + k_h, j : j + k_w] += max_mask * g_val
        return [Tensor(grad, dtype=inp.dtype)]


class AvgPool2d(Function):
    @staticmethod
    def forward(ctx, inp, kernel_h, kernel_w, stride_h=None, stride_w=None):
        stride_h = stride_h or kernel_h
        stride_w = stride_w or kernel_w
        arr = inp.data
        ctx.save_attr(k_h=kernel_h, k_w=kernel_w, s_h=stride_h, s_w=stride_w)
        ctx.save_for_backward(inp=inp)
        out = np.array(
            [
                [
                    arr[..., i : i + kernel_h, j : j + kernel_w].mean(axis=(-2, -1))
                    for j in range(0, arr.shape[-1] - kernel_w + 1, stride_w)
                ]
                for i in range(0, arr.shape[-2] - kernel_h + 1, stride_h)
            ]
        )
        out = out.transpose(2, 3, 0, 1) if arr.ndim == 4 else out
        return Tensor(out, dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        inp = ctx.get_saved_tensor("inp")
        k_h = ctx.get_attr("k_h")
        k_w = ctx.get_attr("k_w")
        s_h = ctx.get_attr("s_h")
        s_w = ctx.get_attr("s_w")
        arr = inp.data
        g = grad_output.data
        grad = np.zeros_like(arr)
        norm = k_h * k_w
        for i in range(0, arr.shape[-2] - k_h + 1, s_h):
            for j in range(0, arr.shape[-1] - k_w + 1, s_w):
                gi = i // s_h
                gj = j // s_w
                if g.ndim == 4:
                    g_val = g[..., gi, gj, None, None]
                else:
                    g_val = g
                grad[..., i : i + k_h, j : j + k_w] += g_val / norm
        return [Tensor(grad, dtype=inp.dtype)]


# ===========================================================================
#  Loss Ops
# ===========================================================================


class MSELoss(Function):
    @staticmethod
    def forward(ctx, inp, target):
        diff = inp.data - target.data
        ctx.save_for_backward(inp=inp, target=target)
        ctx.save_attr(n=inp.numel)
        return Tensor(np.array(float((diff**2).mean())), dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        inp = ctx.get_saved_tensor("inp")
        target = ctx.get_saved_tensor("target")
        n = ctx.get_attr("n")
        g = grad_output.data.flat[0]
        grad_inp = Tensor(2.0 * (inp.data - target.data) * g / n, dtype=inp.dtype)
        return [grad_inp, -grad_inp]


class CrossEntropyLoss(Function):
    @staticmethod
    def forward(ctx, inp, target):
        x = inp.data
        x_max = x.max(axis=-1, keepdims=True)
        logits = x - x_max
        e = np.exp(logits)
        sm = e / e.sum(axis=-1, keepdims=True)
        t = target.data if isinstance(target, Tensor) else target
        if t.ndim == sm.ndim:
            t = t.argmax(axis=-1)
        loss = -np.mean(np.log(sm[np.arange(sm.shape[0]), t.astype(np.int64)] + 1e-10))
        ctx.save_attr(sm=sm, t=t.astype(np.int64))
        return Tensor(np.array([loss]), dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        sm = ctx.get_attr("sm")
        t = ctx.get_attr("t")
        g = grad_output.data.flat[0]
        grad = sm.copy()
        grad[np.arange(sm.shape[0]), t] -= 1.0
        grad /= sm.shape[0]
        return [Tensor(grad * g, dtype="float32"), None]


__all__ = [
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Neg",
    "Pow",
    "MatMul",
    "Sum",
    "Mean",
    "Relu",
    "Sigmoid",
    "Tanh",
    "Gelu",
    "Silu",
    "Sqrt",
    "Exp",
    "Log",
    "Abs",
    "Softmax",
    "LogSoftmax",
    "Reshape",
    "Transpose",
    "Expand",
    "Squeeze",
    "Unsqueeze",
    "GetItem",
    "LayerNorm",
    "RMSNorm",
    "DropoutFn",
    "LinearFn",
    "EmbeddingFn",
    "Conv1d",
    "MaxPool2d",
    "AvgPool2d",
    "MSELoss",
    "CrossEntropyLoss",
]
