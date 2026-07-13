import json
import os
import struct
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from enum import IntEnum

from .tensor import Tensor, _HAS_C_BACKEND


class ModelFamily(IntEnum):
    LLAMA_2 = 0
    LLAMA_3 = 1
    MISTRAL = 2
    QWEN_2 = 3
    DEEPSEEK_V2 = 4
    UNKNOWN = 255


@dataclass
class LlamaConfig:
    family: ModelFamily = ModelFamily.LLAMA_2
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    use_scaled_rope: bool = False
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    head_dim: int = 128
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    num_experts: int = 0
    num_experts_per_tok: int = 0

    @property
    def num_head_dim(self) -> int:
        return self.head_dim

    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads


@dataclass
class MistralConfig:
    family: ModelFamily = ModelFamily.MISTRAL
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    vocab_size: int = 32000
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    sliding_window: int = 4096
    head_dim: int = 128
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0


@dataclass
class Qwen2Config:
    family: ModelFamily = ModelFamily.QWEN_2
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    vocab_size: int = 152064
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling_factor: float = 1.0
    use_rope_scaling: bool = False
    head_dim: int = 128
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0


@dataclass
class DeepSeekV2Config:
    family: ModelFamily = ModelFamily.DEEPSEEK_V2
    hidden_size: int = 2048
    intermediate_size: int = 10944
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    vocab_size: int = 102400
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    head_dim: int = 128
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0


# Registry of known model architectures
_MODEL_REGISTRY: Dict[str, Dict[str, dict]] = {
    "llama2": {
        "7B": {
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
        },
        "13B": {
            "hidden_size": 5120,
            "intermediate_size": 13824,
            "num_hidden_layers": 40,
            "num_attention_heads": 40,
            "num_key_value_heads": 40,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
        },
        "70B": {
            "hidden_size": 8192,
            "intermediate_size": 28672,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
        },
    },
    "llama3": {
        "8B": {
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 8192,
            "use_scaled_rope": True,
            "rope_theta": 500000.0,
        },
        "70B": {
            "hidden_size": 8192,
            "intermediate_size": 28672,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 8192,
            "use_scaled_rope": True,
            "rope_theta": 500000.0,
        },
    },
    "mistral": {
        "7B": {
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "max_position_embeddings": 32768,
            "sliding_window": 4096,
        },
    },
    "qwen2": {
        "7B": {
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "vocab_size": 152064,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
        },
        "72B": {
            "hidden_size": 8192,
            "intermediate_size": 29568,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "vocab_size": 152064,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
        },
    },
    "deepseek_v2": {
        "lite": {
            "hidden_size": 2048,
            "intermediate_size": 10944,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "vocab_size": 102400,
            "max_position_embeddings": 4096,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
        },
        "full": {
            "hidden_size": 5120,
            "intermediate_size": 12288,
            "num_hidden_layers": 60,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "vocab_size": 102400,
            "max_position_embeddings": 4096,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
        },
    },
}


def get_model_config(family: str, size: str) -> dict:
    """Get a model config dict by family and size name."""
    family = family.lower().replace("-", "_")
    if family == "llama":
        family = "llama2"
    if family not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model family '{family}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    if size not in _MODEL_REGISTRY[family]:
        raise ValueError(
            f"Unknown size '{size}' for {family}. "
            f"Available: {list(_MODEL_REGISTRY[family].keys())}"
        )
    return dict(_MODEL_REGISTRY[family][size])


def config_from_json(path: str) -> dict:
    """Load model config from a JSON file."""
    with open(path) as f:
        return json.load(f)


def list_available_models() -> List[str]:
    """List all known model family:size combinations."""
    models = []
    for family, sizes in _MODEL_REGISTRY.items():
        for size in sizes:
            models.append(f"{family}:{size}")
    return models


# =========================================================================
# HF Weight Name Mapping
# =========================================================================

HF_WEIGHT_MAP: Dict[str, Dict[str, str]] = {
    "llama2": {
        "model.embed_tokens.weight": "embedding.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
        "self_attn.q_proj.weight": "attn.q_proj.weight",
        "self_attn.k_proj.weight": "attn.k_proj.weight",
        "self_attn.v_proj.weight": "attn.v_proj.weight",
        "self_attn.o_proj.weight": "attn.o_proj.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "mlp_norm.weight",
    },
    "mistral": {
        "model.embed_tokens.weight": "embedding.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
        "self_attn.q_proj.weight": "attn.q_proj.weight",
        "self_attn.k_proj.weight": "attn.k_proj.weight",
        "self_attn.v_proj.weight": "attn.v_proj.weight",
        "self_attn.o_proj.weight": "attn.o_proj.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "mlp_norm.weight",
    },
    "qwen2": {
        "model.embed_tokens.weight": "embedding.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
        "self_attn.q_proj.weight": "attn.q_proj.weight",
        "self_attn.k_proj.weight": "attn.k_proj.weight",
        "self_attn.v_proj.weight": "attn.v_proj.weight",
        "self_attn.o_proj.weight": "attn.o_proj.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "mlp_norm.weight",
    },
    "deepseek_v2": {
        "model.embed_tokens.weight": "embedding.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
        "self_attn.q_proj.weight": "attn.q_proj.weight",
        "self_attn.kv_b_proj.weight": "attn.kv_b_proj.weight",
        "self_attn.o_proj.weight": "attn.o_proj.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "mlp_norm.weight",
    },
}

# LLaMA 3 renamed q_proj/k_proj/v_proj -> qkv_proj in some configs
HF_WEIGHT_MAP["llama3"] = dict(HF_WEIGHT_MAP["llama2"])


def _remap_hf_weight_name(hf_name: str, family: str) -> Optional[str]:
    """Map an HF weight name to SneppX internal format using the layer pattern.

    HF format: model.layers.{i}.{module}.weight
    SneppX format: layers.{i}.{module}.weight
    """
    if family not in HF_WEIGHT_MAP:
        return None

    mapping = HF_WEIGHT_MAP[family]

    # Non-layer weights
    if hf_name in mapping:
        return mapping[hf_name]

    # Layer weights: model.layers.{i}.{submodule}.{param}
    parts = hf_name.split(".")
    if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
        try:
            layer_idx = int(parts[2])
        except ValueError:
            return None
        suffix = ".".join(parts[3:])
        if suffix in mapping:
            remapped = f"layers.{layer_idx}.{mapping[suffix]}"
            return remapped
    return None


def _generate_sneppx_weight_names(family: str, num_layers: int) -> List[str]:
    """Generate all expected SneppX weight names for a given model."""
    names = ["embedding.weight", "norm.weight", "lm_head.weight"]
    for i in range(num_layers):
        names.append(f"layers.{i}.attn_norm.weight")
        names.append(f"layers.{i}.attn.q_proj.weight")
        names.append(f"layers.{i}.attn.k_proj.weight")
        names.append(f"layers.{i}.attn.v_proj.weight")
        names.append(f"layers.{i}.attn.o_proj.weight")
        names.append(f"layers.{i}.mlp_norm.weight")
        names.append(f"layers.{i}.mlp.gate_proj.weight")
        names.append(f"layers.{i}.mlp.up_proj.weight")
        names.append(f"layers.{i}.mlp.down_proj.weight")
    return names


# =========================================================================
# Safetensors Reader
# =========================================================================


def read_safetensors(path: str) -> Tuple[dict, Dict[str, bytes]]:
    """Read a safetensors file, returns (metadata, {name: tensor_bytes})."""
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes)
        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            f.seek(8 + header_len + info["data_offsets"][0])
            tensors[name] = f.read(info["data_offsets"][1] - info["data_offsets"][0])
        return header.get("__metadata__", {}), tensors


# =========================================================================
# Weight Converter
# =========================================================================


def convert_hf_to_sneppx(
    hf_dir: str,
    family: str,
    output_path: str,
    num_layers: Optional[int] = None,
    verbose: bool = True,
) -> int:
    """Convert HuggingFace safetensors weights to SneppX checkpoint format.

    Args:
        hf_dir: Directory containing .safetensors files and config.json
        family: Model family ('llama2', 'llama3', 'mistral', 'qwen2', 'deepseek_v2')
        output_path: Where to write the .sneppx checkpoint
        num_layers: Override number of layers (auto-detected if None)

    Returns:
        Number of weights converted
    """
    from .checkpoint import CheckpointWriter

    # Find all safetensors files
    import glob

    st_files = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {hf_dir}")

    # Auto-detect config
    config_path = os.path.join(hf_dir, "config.json")
    hf_config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            hf_config = json.load(f)

    if num_layers is None:
        num_layers = hf_config.get("num_hidden_layers", 0)
        if num_layers == 0:
            raise ValueError("num_layers not specified and not found in config.json")

    # Read all tensors
    if verbose:
        print(
            f"[SNEPPX Convert] Reading {len(st_files)} safetensors files from {hf_dir}"
        )
    all_tensors = {}
    for st_path in st_files:
        meta, tensors = read_safetensors(st_path)
        all_tensors.update(tensors)

    if verbose:
        print(f"[SNEPPX Convert] Found {len(all_tensors)} weight tensors")

    # Map to SneppX names
    converted = {}
    mapped_count = 0
    for hf_name, data in all_tensors.items():
        sneppx_name = _remap_hf_weight_name(hf_name, family)
        if sneppx_name:
            converted[sneppx_name] = data
            mapped_count += 1
            if verbose:
                print(f"  {hf_name} -> {sneppx_name} ({len(data)} bytes)")
        else:
            if verbose:
                print(f"  {hf_name} -> (skipped, no mapping)")

    # Write checkpoint
    w = CheckpointWriter(output_path)
    for name in _generate_sneppx_weight_names(family, num_layers):
        data = converted.get(name)
        if data is None:
            if verbose:
                print(f"  WARNING: missing weight '{name}', writing zeros")
            data = b"\x00" * 4  # placeholder
        w.write_tensor(data, shape=(len(data),), dtype=0)
    meta = {
        "family": family,
        "num_layers": num_layers,
        "num_converted": mapped_count,
        "model_config": json.dumps(hf_config),
    }
    w.write_metadata(meta)
    w.close()

    if verbose:
        print(f"[SNEPPX Convert] Saved {mapped_count} weights to {output_path}")
    return mapped_count


# =========================================================================
# Model Builder
# =========================================================================


def build_model_from_config(
    config: dict,
    batch_size: int = 1,
    seq_len: int = 128,
) -> "Tensor":
    """Build a model from a config dict and return example output shape info.

    In a real implementation this would construct Transformer layers, but the
    current NN module (nn.py) already has TransformerBlock and Transformer
    classes. This function serves as the config-to-architecture mapper.

    Returns the estimated parameter count.
    """
    hidden_size = config.get("hidden_size", 4096)
    intermediate_size = config.get("intermediate_size", 11008)
    num_layers = config.get("num_hidden_layers", 32)
    num_heads = config.get("num_attention_heads", 32)
    kv_heads = config.get("num_key_value_heads", num_heads)
    vocab_size = config.get("vocab_size", 32000)
    max_seq_len = config.get("max_position_embeddings", 4096)
    head_dim = hidden_size // num_heads

    # Parameter count estimation
    # Embedding
    params = vocab_size * hidden_size
    # Per layer
    attn_q = hidden_size * (num_heads * head_dim)
    attn_k = hidden_size * (kv_heads * head_dim)
    attn_v = hidden_size * (kv_heads * head_dim)
    attn_o = (num_heads * head_dim) * hidden_size
    mlp_gate = hidden_size * intermediate_size
    mlp_up = hidden_size * intermediate_size
    mlp_down = intermediate_size * hidden_size
    norms = hidden_size * 2  # attn_norm + mlp_norm
    per_layer = attn_q + attn_k + attn_v + attn_o + mlp_gate + mlp_up + mlp_down + norms
    params += num_layers * per_layer
    # Final norm + lm_head
    params += hidden_size + vocab_size * hidden_size

    result = {
        "config": config,
        "total_params": params,
        "param_str": f"{params/1e9:.2f}B",
        "estimated_memory_gb": params * 4 / (1024**3),  # FP32
        "layers": num_layers,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "max_seq_len": max_seq_len,
        "has_gqa": kv_heads != num_heads,
    }

    # Check for DeepSeek MLA
    if "kv_lora_rank" in config:
        result["has_mla"] = True
        result["kv_lora_rank"] = config["kv_lora_rank"]
        result["q_lora_rank"] = config.get("q_lora_rank", 0)

    return result


def build_transformer_from_config(
    config: dict,
) -> "Module":
    """Build a Transformer nn.Module from a config dict."""
    from .nn import Transformer, TransformerBlock, MultiheadAttention

    hidden_size = config.get("hidden_size", 4096)
    intermediate_size = config.get("intermediate_size", 11008)
    num_layers = config.get("num_hidden_layers", 32)
    num_heads = config.get("num_attention_heads", 32)
    kv_heads = config.get("num_key_value_heads", num_heads)
    vocab_size = config.get("vocab_size", 32000)
    max_seq_len = config.get("max_position_embeddings", 4096)
    dropout = config.get("hidden_dropout", 0.0)

    blocks = []
    for i in range(num_layers):
        block = TransformerBlock(
            dim=hidden_size,
            num_heads=num_heads,
            ffn_dim=intermediate_size,
            dropout=dropout,
        )
        blocks.append(block)

    model = Transformer(
        vocab_size=vocab_size,
        dim=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=intermediate_size,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
    return model


# =========================================================================
# from_pretrained() API
# =========================================================================


def from_pretrained(
    model_id: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    verbose: bool = True,
) -> dict:
    """Load a pretrained model configuration.

    This is a simplified version that returns a config dict. In production,
    this would download weights from HuggingFace and build the full model.

    Args:
        model_id: HF-style model ID (e.g., 'meta-llama/Llama-2-7b-hf')
        cache_dir: Cache directory for downloaded files
        force_download: Re-download even if cached
        verbose: Print progress

    Returns:
        Model config dict with architecture info
    """
    # Map model_id to (family, size)
    model_id_lower = model_id.lower()
    mapping = [
        ("llama-2-7b", "llama2", "7B"),
        ("llama-2-13b", "llama2", "13B"),
        ("llama-2-70b", "llama2", "70B"),
        ("llama-3-8b", "llama3", "8B"),
        ("llama-3-70b", "llama3", "70B"),
        ("mistral-7b", "mistral", "7B"),
        ("qwen2-7b", "qwen2", "7B"),
        ("qwen2-72b", "qwen2", "72B"),
        ("deepseek-v2-lite", "deepseek_v2", "lite"),
        ("deepseek-v2", "deepseek_v2", "full"),
    ]
    family = None
    size = None
    for pattern, f, s in mapping:
        if pattern in model_id_lower:
            family, size = f, s
            break

    if family is None:
        raise ValueError(
            f"Unknown model_id '{model_id}'. " f"Supported: {[p[0] for p in mapping]}"
        )

    config = get_model_config(family, size)
    if verbose:
        print(f"[SNEPPX from_pretrained] {model_id} -> family={family}, size={size}")
        print(
            f"  hidden_size={config['hidden_size']}, "
            f"layers={config['num_hidden_layers']}, "
            f"heads={config['num_attention_heads']}, "
            f"kv_heads={config.get('num_key_value_heads', 'N/A')}"
        )

    result = build_model_from_config(config)
    result["model_id"] = model_id
    result["family"] = family
    result["size"] = size
    return result
