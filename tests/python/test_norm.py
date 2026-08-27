import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings.nn import BatchNorm1d, BatchNorm2d, GroupNorm

EPS = 1e-6


def fd(inp, loss_fn):
    base = inp.data.copy()
    g = np.zeros_like(base)
    it = np.nditer(base, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        o = base[idx]
        base[idx] = o + EPS
        lp = loss_fn(Tensor(base.copy(), dtype=inp.dtype)).data.flat[0]
        base[idx] = o - EPS
        lm = loss_fn(Tensor(base.copy(), dtype=inp.dtype)).data.flat[0]
        base[idx] = o
        g[idx] = (lp - lm) / (2 * EPS)
        it.iternext()
    return g


def check(name, inp, loss_fn, atol=1e-5):
    inp2 = Tensor(inp.data.copy())
    inp2.requires_grad_(True)
    loss_fn(inp2).backward()
    numeric = fd(inp, loss_fn)
    assert np.allclose(inp2.grad.data, numeric, atol=atol), \
        f"{name}: max|d|={np.abs(inp2.grad.data - numeric).max()}"
    print(f"{name}: OK")


def test_batchnorm1d():
    x = Tensor(np.random.randn(4, 3), dtype="float64")

    def bn1_loss(i):
        m = BatchNorm1d(3)
        m.train()
        return (m(i) * m(i)).mean()
    check("BatchNorm1d x grad", x, bn1_loss)
    w = Tensor(np.random.randn(3), dtype="float64")
    w.requires_grad_(True)

    def bn1_w(i):
        m = BatchNorm1d(3)
        m.weight = i
        m.bias = Tensor(np.zeros((3,)))
        m.train()
        return (m(x) * m(x)).mean()
    check("BatchNorm1d weight grad", w, bn1_w)


def test_batchnorm2d():
    x = Tensor(np.random.randn(2, 3, 4, 4), dtype="float64")

    def bn2_loss(i):
        m = BatchNorm2d(3)
        m.train()
        return (m(i) * m(i)).mean()
    check("BatchNorm2d x grad", x, bn2_loss)


def test_groupnorm():
    x = Tensor(np.random.randn(2, 6, 4, 4), dtype="float64")

    def gn_loss(i):
        m = GroupNorm(3, 6)
        m.train()
        return (m(i) * m(i)).mean()
    check("GroupNorm x grad", x, gn_loss)
    wg = Tensor(np.random.randn(6), dtype="float64")
    wg.requires_grad_(True)

    def gn_w(i):
        m = GroupNorm(3, 6)
        m.weight = i
        m.bias = Tensor(np.zeros((6,)))
        m.train()
        return (m(x) * m(x)).mean()
    check("GroupNorm weight grad", wg, gn_w)


def test_batchnorm_eval_grad():
    bn = BatchNorm1d(3)
    bn.train()
    for _ in range(3):
        bn(Tensor(np.random.randn(4, 3), dtype="float64"))
    bn.eval()
    xe = Tensor(np.random.randn(4, 3), dtype="float64")
    xe.requires_grad_(True)
    out = bn(xe)
    (out * out).mean().backward()
    assert xe.grad is not None
    print("BatchNorm eval grad: OK")


if __name__ == "__main__":
    test_batchnorm1d()
    test_batchnorm2d()
    test_groupnorm()
    test_batchnorm_eval_grad()
    print("ALL NORM OK")
