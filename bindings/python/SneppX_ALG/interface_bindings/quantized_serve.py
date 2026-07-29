"""FP8/INT4 quantized model serving — wraps QuantizedLinear into serving-ready format."""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

import numpy as np

from .tensor import Tensor
from .quantization import QuantMode, QuantGranularity, QuantizedLinear
from .nn import Module, Linear


@dataclass
class QuantizedModelConfig:
    quant_mode: QuantMode = QuantMode.INT4_ASYMMETRIC
    quant_granularity: QuantGranularity = QuantGranularity.PER_CHANNEL
    skip_layers: List[str] = field(default_factory=lambda: ["lm_head", "embed_tokens"])
    use_fp8_for_embed: bool = False


def quantize_linear_layer(
    weight: np.ndarray,
    mode: QuantMode = QuantMode.INT4_ASYMMETRIC,
    granularity: QuantGranularity = QuantGranularity.PER_CHANNEL,
) -> QuantizedLinear:
    q = QuantizedLinear(
        weight=weight,
        quant_mode=mode,
        granularity=granularity,
    )
    return q


def quantize_model_weights(
    model_params: Dict[str, np.ndarray],
    config: Optional[QuantizedModelConfig] = None,
) -> Dict[str, Any]:
    cfg = config or QuantizedModelConfig()
    quantized = {}
    for name, weight in model_params.items():
        skip = any(s in name for s in cfg.skip_layers)
        if skip:
            quantized[name] = weight
            continue
        if cfg.use_fp8_for_embed and "embed" in name:
            mode = QuantMode.FP8_E4M3
        else:
            mode = cfg.quant_mode
        q = quantize_linear_layer(weight, mode=mode, granularity=cfg.quant_granularity)
        quantized[name] = q
    return quantized


def dequantize_weights(
    quantized: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    result = {}
    for name, item in quantized.items():
        if isinstance(item, QuantizedLinear):
            result[name] = item.dequantize()
        elif isinstance(item, np.ndarray):
            result[name] = item
    return result


def quantized_forward(
    quantized_params: Dict[str, Any],
    layer_name: str,
    x: np.ndarray,
) -> np.ndarray:
    item = quantized_params.get(layer_name)
    if isinstance(item, QuantizedLinear):
        q_weight = item.quantize_weight()
        scale = item.scale
        if item.quant_mode in (QuantMode.INT4_SYMMETRIC, QuantMode.INT4_ASYMMETRIC):
            dequant = item.dequantize()
            return x @ dequant.T
        return x @ item.dequantize().T
    elif isinstance(item, np.ndarray):
        return x @ item.T
    return x


def estimate_model_size_mb(quantized: Dict[str, Any]) -> float:
    total_bytes = 0
    for name, item in quantized.items():
        if isinstance(item, QuantizedLinear):
            total_bytes += item.weight.nbytes
        elif isinstance(item, np.ndarray):
            total_bytes += item.nbytes
    return total_bytes / (1024 * 1024)
