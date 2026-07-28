"""Tests for Gradient Checkpointing."""

import numpy as np
from SneppX_ALG.interface_bindings.grad_checkpoint import CheckpointSegment, checkpoint, GradientCheckpointer
from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings.nn import Linear, ReLU, Sequential


def double(x: Tensor) -> Tensor:
    return Tensor.from_numpy(x.data * 2.0)


def square(x: Tensor) -> Tensor:
    return Tensor.from_numpy(x.data ** 2.0)


def test_checkpoint_segment_forward():
    x = Tensor.ones((2, 3))
    seg = CheckpointSegment(double, (x,))
    out = seg.forward()
    assert np.allclose(out.data, 2.0)
    assert seg.output is not None


def test_checkpoint_segment_recompute():
    x = Tensor.ones((2, 3))
    seg = CheckpointSegment(double, (x,))
    seg.forward()
    out = seg.recompute()
    assert np.allclose(out.data, 2.0)


def test_checkpoint_function():
    x = Tensor.ones((2, 3)) * 5.0
    out = checkpoint(square, x)
    assert np.allclose(out.data, 25.0)
    assert hasattr(out, "_checkpoint_segments")
    assert len(out._checkpoint_segments) == 1


def test_checkpoint_multiple_inputs():
    a = Tensor.ones((2,)) * 3.0
    b = Tensor.ones((2,)) * 4.0

    def add(x, y):
        return Tensor.from_numpy(x.data + y.data)

    out = checkpoint(add, a, b)
    assert np.allclose(out.data, 7.0)
    assert len(out._checkpoint_segments) == 1


def test_checkpoint_preserves_computation():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    x = Tensor.randn((2, 4))
    out_direct = model(x)
    out_ckpt = checkpoint(lambda x: model(x), x)
    assert out_direct.shape == out_ckpt.shape
    assert np.allclose(out_direct.data, out_ckpt.data, atol=1e-5)


def test_gradient_checkpointer_context():
    gc = GradientCheckpointer()
    with gc.context():
        out = gc.checkpoint(double, Tensor.ones((3,)))
        out = gc.checkpoint(square, out)
    assert len(gc.segments) == 2
    assert np.allclose(out.data, 4.0)


def test_gradient_checkpointer_recompute_all():
    gc = GradientCheckpointer()
    with gc.context():
        x = Tensor.ones((3,)) * 2.0
        y = gc.checkpoint(double, x)
        z = gc.checkpoint(square, y)
    gc.recompute_all()
    assert len(gc.segments) == 2
    for seg in gc.segments:
        assert seg.output is not None
    assert np.allclose(z.data, 16.0)


def test_checkpoint_sequential():
    from SneppX_ALG.interface_bindings.grad_checkpoint import checkpoint_sequential
    layers = [Linear(4, 8), ReLU(), Linear(8, 4), ReLU(), Linear(4, 2)]
    x = Tensor.randn((2, 4))
    out = checkpoint_sequential(layers, x, segments=2)
    assert out.shape == (2, 2)


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
