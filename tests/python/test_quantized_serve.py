"""Tests for quantized model serving."""

import numpy as np
from SneppX_ALG.interface_bindings.quantized_serve import (
    QuantizedModelConfig,
    quantize_model_weights,
    dequantize_weights,
    estimate_model_size_mb,
)
from SneppX_ALG.interface_bindings.quantization import QuantMode, QuantGranularity, QuantizedLinear


def _dummy_params():
    rng = np.random.RandomState(42)
    return {
        "layers.0.weights": rng.randn(16, 32).astype(np.float32),
        "layers.1.weights": rng.randn(32, 16).astype(np.float32),
        "lm_head.weight": rng.randn(10, 32).astype(np.float32),
    }


def test_quantize_default():
    params = _dummy_params()
    quantized = quantize_model_weights(params)
    assert len(quantized) == 3
    assert isinstance(quantized["layers.0.weights"], QuantizedLinear)
    print("  test_quantize_default PASS")


def test_skip_layers():
    params = _dummy_params()
    cfg = QuantizedModelConfig(skip_layers=["lm_head"])
    quantized = quantize_model_weights(params, cfg)
    assert isinstance(quantized["lm_head.weight"], np.ndarray)
    assert isinstance(quantized["layers.0.weights"], QuantizedLinear)
    print("  test_skip_layers PASS")


def test_dequantize():
    params = _dummy_params()
    quantized = quantize_model_weights(params)
    dequant = dequantize_weights(quantized)
    for name, item in dequant.items():
        assert isinstance(item, np.ndarray), f"{name} is {type(item)}"
    print("  test_dequantize PASS")


def test_estimate_size():
    params = _dummy_params()
    quantized = quantize_model_weights(params)
    size_mb = estimate_model_size_mb(quantized)
    assert size_mb > 0
    assert isinstance(size_mb, float)
    print("  test_estimate_size PASS")


def test_quantize_fp8():
    params = _dummy_params()
    cfg = QuantizedModelConfig(quant_mode=QuantMode.FP8_E4M3)
    quantized = quantize_model_weights(params, cfg)
    assert isinstance(quantized["layers.0.weights"], QuantizedLinear)
    print("  test_quantize_fp8 PASS")


def test_quantize_per_tensor():
    params = _dummy_params()
    cfg = QuantizedModelConfig(quant_granularity=QuantGranularity.PER_TENSOR)
    quantized = quantize_model_weights(params, cfg)
    assert isinstance(quantized["layers.0.weights"], QuantizedLinear)
    print("  test_quantize_per_tensor PASS")


def test_empty_params():
    quantized = quantize_model_weights({})
    assert quantized == {}
    print("  test_empty_params PASS")


if __name__ == "__main__":
    test_quantize_default()
    test_skip_layers()
    test_dequantize()
    test_estimate_size()
    test_quantize_fp8()
    test_quantize_per_tensor()
    test_empty_params()
    print("ALL quantized_serve TESTS PASS")
