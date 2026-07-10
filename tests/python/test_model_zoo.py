import sys, os, json, tempfile
if '__file__' in globals():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../bindings/python'))
else:
    sys.path.insert(0, os.path.join(os.getcwd(), 'tests/python/../../bindings/python'))

from SneppX_ALG.interface_bindings.model_zoo import (
    ModelFamily, LlamaConfig, MistralConfig, Qwen2Config, DeepSeekV2Config,
    get_model_config, config_from_json, list_available_models,
    _remap_hf_weight_name, _generate_sneppx_weight_names,
    read_safetensors, convert_hf_to_sneppx,
    build_model_from_config, build_transformer_from_config,
    from_pretrained, _MODEL_REGISTRY, HF_WEIGHT_MAP,
)

failed = []

def check(name, cond):
    if not cond:
        print(f"  FAIL {name}")
        failed.append(name)
    else:
        print(f"  PASS {name}")

def test_get_model_config():
    c = get_model_config("llama2", "7B")
    check("llama2 7B hidden", c["hidden_size"] == 4096)
    check("llama2 7B layers", c["num_hidden_layers"] == 32)
    c = get_model_config("llama3", "8B")
    check("llama3 8B kv_heads", c["num_key_value_heads"] == 8)
    check("llama3 8B scaled rope", c["use_scaled_rope"] == True)
    c = get_model_config("mistral", "7B")
    check("mistral 7B sliding", c["sliding_window"] == 4096)
    c = get_model_config("qwen2", "72B")
    check("qwen2 72B layers", c["num_hidden_layers"] == 80)
    c = get_model_config("deepseek_v2", "lite")
    check("deepseek lite kv_lora", c["kv_lora_rank"] == 512)
    check("deepseek lite q_lora", c["q_lora_rank"] == 1536)

def test_get_model_config_errors():
    try:
        get_model_config("unknown", "7B")
        check("unknown family raises", False)
    except ValueError:
        check("unknown family raises", True)
    try:
        get_model_config("llama2", "999B")
        check("unknown size raises", False)
    except ValueError:
        check("unknown size raises", True)

def test_list_available():
    models = list_available_models()
    check("has llama2:7B", "llama2:7B" in models)
    check("has llama3:8B", "llama3:8B" in models)
    check("has mistral:7B", "mistral:7B" in models)
    check("has qwen2:72B", "qwen2:72B" in models)
    check("has deepseek_v2:lite", "deepseek_v2:lite" in models)
    check("has deepseek_v2:full", "deepseek_v2:full" in models)

def test_config_from_json():
    with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
        json.dump({"family": "llama3", "hidden_size": 4096, "num_hidden_layers": 32}, f)
        path = f.name
    try:
        c = config_from_json(path)
        check("json load family", c["family"] == "llama3")
    finally:
        os.unlink(path)

def test_remap_hf_weight_name():
    # Non-layer
    result = _remap_hf_weight_name("model.embed_tokens.weight", "llama2")
    check("embed remap", result == "embedding.weight")
    # Layer
    result = _remap_hf_weight_name("model.layers.0.self_attn.q_proj.weight", "llama2")
    check("q_proj remap", result == "layers.0.attn.q_proj.weight")
    result = _remap_hf_weight_name("model.layers.15.mlp.gate_proj.weight", "llama2")
    check("gate_proj remap", result == "layers.15.mlp.gate_proj.weight")
    result = _remap_hf_weight_name("model.layers.5.input_layernorm.weight", "llama2")
    check("layernorm remap", result == "layers.5.attn_norm.weight")

def test_remap_unknown_family():
    result = _remap_hf_weight_name("model.embed_tokens.weight", "unknown")
    check("unknown family", result is None)

def test_remap_unmappable():
    result = _remap_hf_weight_name("model.layers.0.foo.bar.weight", "llama2")
    check("unmapped layer", result is None)

def test_remap_deepseek():
    result = _remap_hf_weight_name("model.layers.0.self_attn.kv_b_proj.weight", "deepseek_v2")
    check("deepseek kv_b_proj", result == "layers.0.attn.kv_b_proj.weight")

def test_generate_weight_names():
    names = _generate_sneppx_weight_names("llama2", 2)
    # 3 non-layer + 9 per layer * 2 = 21
    check("3 non-layer", len(names) == 3 + 9 * 2)
    check("embedding first", names[0] == "embedding.weight")
    check("layer 0 attn_norm", "layers.0.attn_norm.weight" in names)
    check("layer 1 down_proj", "layers.1.mlp.down_proj.weight" in names)

def test_build_model_from_config():
    config = get_model_config("llama3", "8B")
    info = build_model_from_config(config)
    check("param count present", "total_params" in info)
    check("param string", "B" in info["param_str"])
    check("has_gqa", info["has_gqa"] == True)
    check("layers", info["layers"] == 32)
    check("hidden_size", info["hidden_size"] == 4096)

def test_build_model_deepseek():
    config = get_model_config("deepseek_v2", "lite")
    info = build_model_from_config(config)
    check("deepseek has_mla", info.get("has_mla") == True)
    check("deepseek kv_lora_rank", info.get("kv_lora_rank") == 512)

def test_from_pretrained():
    info = from_pretrained("meta-llama/Llama-3-8B-hf", verbose=False)
    check("from_pretrained family", info["family"] == "llama3")
    check("from_pretrained size", info["size"] == "8B")
    check("from_pretrained params", "B" in info.get("param_str", ""))

def test_from_pretrained_all():
    models = [
        ("llama2", "7B"),
        ("llama3", "8B"),
        ("mistral", "7B"),
        ("qwen2", "7B"),
        ("deepseek_v2", "lite"),
    ]
    for family, size in models:
        # Map family name to expected pattern
        name_map = {
            "llama2": "llama-2",
            "llama3": "llama-3",
            "mistral": "mistral",
            "qwen2": "qwen2",
            "deepseek_v2": "deepseek-v2",
        }
        model_id = f"{name_map[family]}-{size}".lower()
        info = from_pretrained(model_id, verbose=False)
        check(f"{family} {size} from_pretrained", info["family"] == family)
        check(f"{family} {size}", info["family"] == family)

def test_build_transformer():
    from SneppX_ALG.interface_bindings.nn import Module
    # Use a tiny config to keep test fast
    config = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 100,
        "max_position_embeddings": 32,
        "hidden_dropout": 0.0,
        "rms_norm_eps": 1e-5,
    }
    model = build_transformer_from_config(config)
    check("transformer built", isinstance(model, Module))
    params = model.parameters()
    check("has parameters", len(params) > 0)

def test_safetensors_reader_invalid():
    try:
        read_safetensors("nonexistent_file.safetensors")
        check("invalid file raises", False)
    except FileNotFoundError:
        check("invalid file raises", True)
    except Exception:
        check("invalid file raises", True)

def test_dataclass_defaults():
    lc = LlamaConfig()
    check("LlamaConfig default hidden", lc.hidden_size == 4096)
    mc = MistralConfig()
    check("MistralConfig default sliding", mc.sliding_window == 4096)
    qc = Qwen2Config()
    check("Qwen2Config default rope_theta", abs(qc.rope_theta - 1000000.0) < 0.1)
    dc = DeepSeekV2Config()
    check("DeepSeekV2 KV lora", dc.kv_lora_rank == 512)


if __name__ == '__main__':
    test_get_model_config()
    test_get_model_config_errors()
    test_list_available()
    test_config_from_json()
    test_remap_hf_weight_name()
    test_remap_unknown_family()
    test_remap_unmappable()
    test_remap_deepseek()
    test_generate_weight_names()
    test_build_model_from_config()
    test_build_model_deepseek()
    test_from_pretrained()
    test_from_pretrained_all()
    test_build_transformer()
    test_safetensors_reader_invalid()
    test_dataclass_defaults()

    print(f"\n{'All model zoo tests passed!' if not failed else f'{len(failed)} failures: {failed}'}")
