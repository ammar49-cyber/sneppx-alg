"""Neural Network Module — layers, activations, containers with CUDA kernel support."""

from typing import Callable, List, Optional, Tuple, Union
from .tensor import Tensor, Dtype
import numpy as np
import math

# Try to import CUDA kernels
try:
    from .cuda_kernels import (
        gemm_kernel,
        layernorm_kernel,
        rmsnorm_kernel,
        flash_attention_v2_kernel,
        dropout_kernel,
        gelu_kernel,
        silu_kernel,
        relu_kernel,
        _HAS_CUDA_BACKEND,
    )

    _HAS_CUDA_KERNELS = True
except ImportError:
    _HAS_CUDA_KERNELS = False
    _HAS_CUDA_BACKEND = False


def _is_cuda_tensor(x: Tensor) -> bool:
    """Check if tensor is on CUDA device and CUDA kernels are available."""
    return _HAS_CUDA_KERNELS and x.device.startswith("cuda")


class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._training = True
        self._name = self.__class__.__name__

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self):
        params = []
        for p in self._parameters.values():
            if isinstance(p, Tensor):
                params.append(p)
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def named_parameters(self):
        named = []
        for name, p in self._parameters.items():
            if isinstance(p, Tensor):
                named.append((name, p))
        for name, m in self._modules.items():
            for n, p in m.named_parameters():
                named.append((f"{name}.{n}", p))
        return named

    def state_dict(self) -> dict:
        sd = {}
        for name, p in self.named_parameters():
            sd[name] = p.data.copy()
        return sd

    def load_state_dict(self, state_dict: dict):
        for name, p in self.named_parameters():
            if name in state_dict:
                p.data = state_dict[name]

    def to(self, device: str):
        for p in self.parameters():
            if p.device != device:
                p.data = p.data.astype(p.dtype_name)  # Ensure correct dtype
                p._data = p._data  # Keep the same data buffer
                p._data._is_cuda = device.startswith("cuda")
                p.device = device
        return self

    def train(self):
        self._training = True
        for m in self._modules.values():
            m.train()

    def eval(self):
        self._training = False
        for m in self._modules.values():
            m.eval()

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor):
            self._parameters[name] = value
        super().__setattr__(name, value)


class Linear(Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor.randn(
            (out_features, in_features), dtype=dtype
        ) / math.sqrt(in_features)
        self.bias = Tensor.zeros((out_features,), dtype=dtype) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        if _is_cuda_tensor(x):
            # Use CUDA GEMM kernel
            # Reshape for batched GEMM if needed
            if x.ndim > 2:
                # Flatten batch dimensions
                batch_shape = x.shape[:-1]
                batch_size = np.prod(batch_shape)
                x_2d = x.reshape((batch_size, x.shape[-1]))
                out_data = np.empty((batch_size, self.out_features), dtype=np.float32)
                gemm_kernel(
                    x_2d.data,
                    self.weight.data.T,
                    out_data,
                    batch_size,
                    self.out_features,
                    x_2d.shape[1],
                )
                out_data = out_data.reshape(batch_shape + (self.out_features,))
            else:
                out_data = np.empty((x.shape[0], self.out_features), dtype=np.float32)
                gemm_kernel(
                    x.data,
                    self.weight.data.T,
                    out_data,
                    x.shape[0],
                    self.out_features,
                    x.shape[1],
                )
            out = Tensor(out_data, dtype=x.dtype_name, device=x.device)
            if self.bias is not None:
                # Ensure bias is on same device
                bias = (
                    self.bias.to(x.device)
                    if self.bias.device != x.device
                    else self.bias
                )
                out = out + bias
            return out
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype="float32"):
        super().__init__()
        self.weight = Tensor.randn((num_embeddings, embedding_dim), dtype=dtype) * 0.01

    def forward(self, indices: Tensor) -> Tensor:
        idx = indices.data.astype(np.int64)
        w = self.weight.data
        return Tensor.from_numpy(w[idx])


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self._training or self.p == 0:
            return x
        if _is_cuda_tensor(x):
            out_data = np.empty_like(x.data)
            mask = dropout_kernel(x.data, out_data, self.p, True)
            return Tensor.from_numpy(out_data, dtype=x.dtype_name)
        mask = np.random.binomial(1, 1.0 - self.p, x.shape).astype(np.float32)
        mask /= 1.0 - self.p
        return Tensor.from_numpy(x.data * mask, dtype=x.dtype_name)


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Tensor.ones(normalized_shape, dtype=dtype)
        self.bias = Tensor.zeros(normalized_shape, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if _is_cuda_tensor(x):
            out_data = np.empty_like(x.data)
            layernorm_kernel(
                x.data,
                self.weight.data,
                self.bias.data,
                out_data,
                self.normalized_shape,
                self.eps,
            )
            return Tensor(out_data, dtype=x.dtype_name, device=x.device)
        arr = x.data
        axis = tuple(range(-len(self.normalized_shape), 0))
        mean = arr.mean(axis=axis, keepdims=True)
        var = arr.var(axis=axis, keepdims=True)
        out = (arr - mean) / np.sqrt(var + self.eps)
        out = out * self.weight.data + self.bias.data
        return Tensor(out, dtype=x.dtype_name, device=x.device)


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Tensor.ones((dim,), dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if _is_cuda_tensor(x):
            out_data = np.empty_like(x.data)
            rmsnorm_kernel(x.data, self.weight.data, out_data, (self.dim,), self.eps)
            return Tensor(out_data, dtype=x.dtype_name, device=x.device)
        arr = x.data
        rms = np.sqrt((arr**2).mean(axis=-1, keepdims=True) + self.eps)
        return Tensor(arr / rms * self.weight.data, dtype=x.dtype_name, device=x.device)


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.silu()


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh_act()


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        for i, m in enumerate(modules):
            self._modules[str(i)] = m

    def forward(self, x: Tensor) -> Tensor:
        for m in self._modules.values():
            x = m(x)
        return x


class MultiheadAttention(Module):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = 0.0, dtype="float32"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, dtype=dtype)
        self.k_proj = Linear(embed_dim, embed_dim, dtype=dtype)
        self.v_proj = Linear(embed_dim, embed_dim, dtype=dtype)
        self.o_proj = Linear(embed_dim, embed_dim, dtype=dtype)
        self.dropout = Dropout(dropout)

    def forward(
        self, q: Tensor, k: Tensor = None, v: Tensor = None, mask: Tensor = None
    ) -> Tensor:
        if k is None:
            k = q
        if v is None:
            v = q
        B, T, C = q.shape
        H = self.num_heads
        D = self.head_dim

        # Check if we can use flash attention (CUDA tensors)
        use_flash = _is_cuda_tensor(q) and _is_cuda_tensor(k) and _is_cuda_tensor(v)

        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        if use_flash:
            # Flash Attention: (B, H, T, D)
            q_proj = q_proj.reshape(B, T, H, D).permute(0, 2, 1, 3)
            k_proj = k_proj.reshape(B, -1, H, D).permute(0, 2, 1, 3)
            v_proj = v_proj.reshape(B, -1, H, D).permute(0, 2, 1, 3)

            out = np.empty_like(q_proj.data)
            flash_attention_v2_kernel(
                q_proj.data,
                k_proj.data,
                v_proj.data,
                out,
                B,
                H,
                T,
                D,
                causal=(mask is not None),
            )
            out_t = Tensor(out, dtype=q.dtype_name, device=q.device)
            out_t = out_t.permute(0, 2, 1, 3).reshape(B, T, C)
            return self.o_proj(out_t)
        else:
            q_np = q_proj.data.reshape(B, T, H, D).transpose(0, 2, 1, 3)
            k_np = k_proj.data.reshape(B, -1, H, D).transpose(0, 2, 1, 3)
            v_np = v_proj.data.reshape(B, -1, H, D).transpose(0, 2, 1, 3)
            scores = (q_np @ k_np.transpose(0, 1, 3, 2)) / math.sqrt(D)
            if mask is not None:
                scores = scores + mask.data
            attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = attn / attn.sum(axis=-1, keepdims=True)
            out_np = attn @ v_np
            out_np = out_np.transpose(0, 2, 1, 3).reshape(B, T, C)

        return self.o_proj(Tensor(out_np, dtype=q.dtype_name, device=q.device))


class TransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        dtype="float32",
    ):
        super().__init__()
        self.attention = MultiheadAttention(dim, num_heads, dropout, dtype)
        self.norm1 = LayerNorm(dim, dtype=dtype)
        self.norm2 = LayerNorm(dim, dtype=dtype)
        self.ffn = Sequential(
            Linear(dim, ffn_dim, dtype=dtype),
            GELU(),
            Linear(ffn_dim, dim, dtype=dtype),
            Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        dtype="float32",
    ):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, dim, dtype)
        self.pos_embedding = Embedding(max_seq_len, dim, dtype)
        self.blocks = Sequential(
            *[
                TransformerBlock(dim, num_heads, ffn_dim, dropout, dtype)
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(dim, dtype=dtype)
        self.lm_head = Linear(dim, vocab_size, dtype=dtype)

    def forward(self, tokens: Tensor) -> Tensor:
        B, T = tokens.shape
        tok_emb = self.token_embedding(tokens)
        pos = Tensor.arange(0, T, dtype="int64").unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.norm(x)
        return self.lm_head(x)
