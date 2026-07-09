import numpy as np
from SneppX_ALG import Tensor


def test_create():
    t = Tensor([2, 3])
    assert t.shape == (2,), f"Expected (2,), got {t.shape}"
    print("  test_create PASS")


def test_numpy_roundtrip():
    arr = np.random.randn(3, 4).astype(np.float32)
    t = Tensor.from_numpy(arr)
    out = t.numpy()
    assert np.allclose(arr, out), "Roundtrip mismatch"
    print("  test_numpy_roundtrip PASS")


if __name__ == '__main__':
    test_create()
    test_numpy_roundtrip()
    print("All tensor tests passed.")
