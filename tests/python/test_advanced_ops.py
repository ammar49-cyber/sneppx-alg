"""Tests for Advanced Tensor Operations."""

import numpy as np
from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings.advanced_ops import (
    conv2d, conv1d, max_pool2d, avg_pool2d, adaptive_avg_pool2d,
    rnn_cell, lstm_cell, gru_cell,
    multi_head_attention, softmax, silu,
    layernorm, linear, dropout, embedding,
    cat, stack, split, flatten,
    sum, mean, max, min, norm,
    clamp, pad, permute, transpose,
    relu, leaky_relu, sigmoid,
)


def test_conv2d_basic():
    x = Tensor.ones((1, 1, 4, 4))
    w = Tensor.ones((1, 1, 3, 3))
    out = conv2d(x, w, stride=(1, 1), padding=(0, 0))
    assert out.shape == (1, 1, 2, 2)
    assert np.allclose(out.data, 9.0)


def test_conv2d_with_padding():
    x = Tensor.ones((1, 1, 4, 4))
    w = Tensor.ones((1, 1, 3, 3))
    out = conv2d(x, w, padding=(1, 1))
    assert out.shape == (1, 1, 4, 4)


def test_conv2d_with_bias():
    x = Tensor.ones((1, 1, 4, 4))
    w = Tensor.ones((1, 1, 3, 3))
    b = Tensor.ones((1,)) * 2.0
    out = conv2d(x, w, bias=b, stride=(1, 1), padding=(0, 0))
    assert np.allclose(out.data, 11.0)


def test_conv2d_stride2():
    x = Tensor.ones((1, 1, 6, 6))
    w = Tensor.ones((1, 1, 2, 2))
    out = conv2d(x, w, stride=(2, 2))
    assert out.shape == (1, 1, 3, 3)


def test_conv1d_basic():
    x = Tensor.ones((1, 1, 8))
    w = Tensor.ones((1, 1, 3))
    out = conv1d(x, w, stride=1, padding=0)
    assert out.shape == (1, 1, 6)
    assert np.allclose(out.data, 3.0)


def test_max_pool2d():
    x = Tensor.from_numpy(np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32))
    out = max_pool2d(x, kernel_size=2, stride=1)
    assert out.shape == (1, 1, 2, 2)
    assert np.allclose(out.data, [[[[5, 6], [8, 9]]]])


def test_avg_pool2d():
    x = Tensor.ones((1, 1, 4, 4))
    out = avg_pool2d(x, kernel_size=2)
    assert out.shape == (1, 1, 2, 2)
    assert np.allclose(out.data, 1.0)


def test_adaptive_avg_pool2d():
    x = Tensor.ones((1, 1, 8, 8))
    out = adaptive_avg_pool2d(x, (4, 4))
    assert out.shape == (1, 1, 4, 4)


def test_rnn_cell():
    x = Tensor.ones((2, 4))
    w_ih = Tensor.ones((8, 4))
    w_hh = Tensor.ones((8, 8))
    h = rnn_cell(x, None, w_ih, w_hh, nonlinearity="tanh")
    assert h.shape == (2, 8)


def test_lstm_cell():
    x = Tensor.ones((2, 4))
    w_ih = Tensor.ones((16, 4))
    w_hh = Tensor.ones((16, 4))
    h, c = lstm_cell(x, None, None, w_ih, w_hh)
    assert h.shape == (2, 4)
    assert c.shape == (2, 4)


def test_gru_cell():
    x = Tensor.ones((2, 4))
    w_ih = Tensor.ones((12, 4))
    w_hh = Tensor.ones((12, 4))
    h = gru_cell(x, None, w_ih, w_hh)
    assert h.shape == (2, 4)


def test_multi_head_attention():
    q = Tensor.ones((1, 4, 8))
    k = Tensor.ones((1, 4, 8))
    v = Tensor.ones((1, 4, 8))
    out = multi_head_attention(q, k, v, num_heads=2)
    assert out.shape == (1, 4, 8)


def test_softmax():
    x = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]]))
    out = softmax(x.data)
    assert np.allclose(out.sum(), 1.0, atol=1e-6)


def test_linear():
    x = Tensor.ones((2, 4))
    w = Tensor.ones((3, 4))
    out = linear(x, w)
    assert out.shape == (2, 3)
    assert np.allclose(out.data, 4.0)


def test_dropout():
    x = Tensor.ones((100, 100))
    out = dropout(x, p=0.5, training=True)
    assert out.shape == (100, 100)
    assert abs(out.data.mean() - 1.0) < 0.1


def test_embedding():
    indices = Tensor.from_numpy(np.array([[0, 1], [2, 0]]))
    w = Tensor.from_numpy(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    out = embedding(indices, w)
    assert out.shape == (2, 2, 2)


def test_cat():
    a = Tensor.ones((2, 3))
    b = Tensor.ones((2, 3)) * 2
    out = cat([a, b], dim=1)
    assert out.shape == (2, 6)


def test_stack():
    a = Tensor.ones((2, 3))
    b = Tensor.ones((2, 3)) * 2
    out = stack([a, b], dim=0)
    assert out.shape == (2, 2, 3)


def test_flatten():
    x = Tensor.ones((2, 3, 4))
    out = flatten(x, start_dim=1)
    assert out.shape == (2, 12)


def test_clamp():
    x = Tensor.from_numpy(np.array([-1.0, 0.0, 1.0, 2.0]))
    out = clamp(x, min=0.0, max=1.0)
    assert np.allclose(out.data, [0.0, 0.0, 1.0, 1.0])


def test_gelu_nn():
    from SneppX_ALG.interface_bindings.nn import GELU
    x = Tensor.ones((2, 3))
    out = GELU()(x)
    assert out.shape == (2, 3)
    assert np.allclose(out.data, 0.8413, atol=1e-3)


def test_silu():
    x = Tensor.ones((2, 3))
    out = silu(x)
    assert out.shape == (2, 3)


def test_relu():
    x = Tensor.from_numpy(np.array([-1.0, 0.0, 1.0]))
    out = relu(x)
    assert np.allclose(out.data, [0.0, 0.0, 1.0])


def test_leaky_relu():
    x = Tensor.from_numpy(np.array([-1.0, 1.0]))
    out = leaky_relu(x)
    assert out.data[0] == -0.01


def test_sum():
    x = Tensor.ones((2, 3))
    out = sum(x)
    assert np.allclose(out.data, 6.0)


def test_mean():
    x = Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
    out = mean(x, dim=1)
    assert np.allclose(out.data, [1.5, 3.5])


def test_max():
    x = Tensor.from_numpy(np.array([[1.0, 5.0], [3.0, 2.0]]))
    values, indices = max(x, dim=1)
    assert np.allclose(values.data, [5.0, 3.0])


def test_norm():
    x = Tensor.from_numpy(np.array([3.0, 4.0]))
    out = norm(x, p=2)
    assert np.allclose(out.data, 5.0)


def test_pad():
    x = Tensor.ones((2, 2))
    out = pad(x, (1, 1, 1, 1))
    assert out.shape == (4, 4)


def test_permute():
    x = Tensor.ones((2, 3, 4))
    out = permute(x, (2, 0, 1))
    assert out.shape == (4, 2, 3)


def test_transpose():
    x = Tensor.ones((2, 3))
    out = transpose(x, 0, 1)
    assert out.shape == (3, 2)


if __name__ == "__main__":
    import sys
    locals_ = locals().copy()
    passed = 0
    failed = 0
    for name, fn in sorted(locals_.items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  PASS {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    sys.exit(failed)
