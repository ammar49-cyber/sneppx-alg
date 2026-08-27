"""Tests for additional loss functions (gradients verified by finite differences)."""

import numpy as np
import pytest

from SneppX_ALG.interface_bindings.tensor import Tensor

np.random.seed(7)
EPS = 1e-6


def T(arr):
    return Tensor(np.asarray(arr, dtype=np.float64), dtype="float64")


def fd(inp, loss_fn):
    base = inp.data.copy()
    g = np.zeros_like(base)
    it = np.nditer(base, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = base[idx]
        base[idx] = orig + EPS
        lp = loss_fn(T(base.copy())).data.flat[0]
        base[idx] = orig - EPS
        lm = loss_fn(T(base.copy())).data.flat[0]
        base[idx] = orig
        g[idx] = (lp - lm) / (2 * EPS)
        it.iternext()
    return g


def check_grad(inp, loss_fn, atol=1e-4):
    inp2 = T(inp.data.copy())
    inp2.requires_grad_(True)
    loss_fn(inp2).backward()
    numeric = fd(inp, loss_fn)
    assert np.allclose(inp2.grad.data, numeric, atol=atol)


def test_smooth_l1():
    x = T(np.random.randn(5, 4))
    t = T(np.random.randn(5, 4))
    check_grad(x, lambda i: i.smooth_l1_loss(t, beta=1.0))


def test_huber():
    x = T(np.random.randn(5, 4))
    t = T(np.random.randn(5, 4))
    check_grad(x, lambda i: i.huber_loss(t, delta=1.0))


def test_bce():
    x = T(np.random.rand(6, 3))
    t = T(np.random.rand(6, 3))
    check_grad(x, lambda i: i.bce_loss(t))


def test_bce_with_logits():
    x = T(np.random.rand(6, 3))
    t = T(np.random.rand(6, 3))
    check_grad(x, lambda i: i.bce_with_logits_loss(t))


def test_focal():
    logits = T(np.random.randn(4, 5))
    targets = Tensor(np.array([0, 2, 1, 3], dtype=np.int64))
    check_grad(logits, lambda i: i.focal_loss(targets, gamma=2.0, alpha=1.0))


def test_cosine_embedding():
    x1 = T(np.random.randn(3, 4))
    x2 = T(np.random.randn(3, 4))
    y = Tensor(np.array([1, -1, -1], dtype=np.int64))
    check_grad(x1, lambda i: i.cosine_embedding_loss(x2, y, margin=0.0))


def test_triplet():
    a = T(np.random.randn(2, 4))
    p = T(np.random.randn(2, 4))
    n = T(np.random.randn(2, 4))
    check_grad(a, lambda i: i.triplet_margin_loss(p, n, margin=1.0))


def test_contrastive():
    x1 = T(np.random.randn(3, 4))
    x2 = T(np.random.randn(3, 4))
    y = Tensor(np.array([1, 0, 0], dtype=np.int64))
    check_grad(x1, lambda i: i.contrastive_loss(x2, y, margin=1.0))


def test_margin_ranking():
    x1 = T(np.random.randn(5))
    x2 = T(np.random.randn(5))
    y = Tensor(np.array([1, -1, 1, -1, 1], dtype=np.int64))
    check_grad(x1, lambda i: i.margin_ranking_loss(x2, y, margin=0.0))


if __name__ == "__main__":
    import sys

    locals_ = locals().copy()
    passed = failed = 0
    for name, fn in sorted(locals_.items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                failed += 1
    print(f"\n  {passed} passed, {failed} failed")
    sys.exit(failed)
