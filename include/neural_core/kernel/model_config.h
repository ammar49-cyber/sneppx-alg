#ifndef NEURAL_CORE_KERNEL_MODEL_CONFIG_H
#define NEURAL_CORE_KERNEL_MODEL_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct ModelConfig ModelConfig;
typedef struct ModelConfigList ModelConfigList;

// Model architecture types
typedef enum {
    MODEL_ARCH_TRANSFORMER = 0,
    MODEL_ARCH_VIT = 1,
    MODEL_ARCH_DIFFUSION = 2,
    MODEL_ARCH_RNN = 3,
    MODEL_ARCH_GAN = 4,
    MODEL_ARCH_GCN = 5,
    MODEL_ARCH_HSS = 6,
    MODEL_ARCH_NPE = 7,
    MODEL_ARCH_SER = 8,
    MODEL_ARCH_ARC = 9,
    MODEL_ARCH_CUSTOM = 255
} ModelArchitecture;

// Activation types
typedef enum {
    ACTIVATION_GELU = 0,
    ACTIVATION_RELU = 1,
    ACTIVATION_SILU = 2,
    ACTIVATION_GEGLU = 3,
    ACTIVATION_SWIGLU = 4
} ActivationType;

// Position encoding types
typedef enum {
    POS_ENCODING_LEARNED = 0,
    POS_ENCODING_ROPE = 1,
    POS_ENCODING_ALIBI = 2,
    POS_ENCODING_SINUSOIDAL = 3
} PositionEncodingType;

// Attention types
typedef enum {
    ATTENTION_CAUSAL = 0,
    ATTENTION_FLASH = 1,
    ATTENTION_SLIDING_WINDOW = 2,
    ATTENTION_BLOCK_SPARSE = 3
} AttentionType;

// Model configuration structure
struct ModelConfig {
    // Identity
    char *name;
    char *version;
    char *description;
    char *author;
    char *license;

    // Architecture
    ModelArchitecture architecture;
    int vocab_size;
    int hidden_size;
    int num_layers;
    int num_heads;
    int num_kv_heads;          // For GQA
    int intermediate_size;     // FFN dimension
    int max_position_embeddings;
    int max_seq_len;

    // Normalization
    float layer_norm_eps;
    float rms_norm_eps;
    int use_rms_norm;          // 1 for RMSNorm, 0 for LayerNorm

    // Activations
    ActivationType hidden_act;
    ActivationType ffn_act;
    int gated_ffn;             // 1 for SwiGLU/GEGLU

    // Attention
    AttentionType attention_type;
    float attention_dropout;
    float hidden_dropout;
    int use_flash_attention;
    int sliding_window;        // For sliding window attention

    // Position encoding
    PositionEncodingType pos_encoding;
    float rope_theta;
    int rope_scaling;          // 0=none, 1=linear, 2=dynamic (NTK-aware/YaRN)
    float rope_scaling_factor; // scaling factor for extended context

    // Initialization
    float initializer_range;
    int tie_word_embeddings;

    // Quantization
    int quantize;              // 1 if quantized
    int quant_bits;            // 4, 8, etc.
    int quant_group_size;      // For grouped quantization

    // Distributed
    int tensor_parallel_size;
    int pipeline_parallel_size;
    int sequence_parallel;     // 1 for sequence parallelism

    // Training
    float learning_rate;
    float weight_decay;
    float max_grad_norm;
    int warmup_steps;
    int max_steps;
    int gradient_accumulation_steps;
    int mixed_precision;       // 1 for FP16/BF16

    // MoE (Mixture of Experts)
    int num_experts;
    int num_experts_per_token;
    float router_aux_loss_coef;

    // Custom fields (key-value pairs)
    char **custom_keys;
    char **custom_values;
    int custom_count;
};

// List of model configs
struct ModelConfigList {
    ModelConfig **configs;
    int count;
    int capacity;
};

// Create/destroy
ModelConfig *model_config_create(void);
void model_config_destroy(ModelConfig *config);
ModelConfigList *model_config_list_create(void);
void model_config_list_destroy(ModelConfigList *list);

// JSON serialization
char *model_config_to_json(const ModelConfig *config, int pretty);
ModelConfig *model_config_from_json(const char *json);

// File I/O
int model_config_save(const ModelConfig *config, const char *path);
ModelConfig *model_config_load(const char *path);

// Validation
int model_config_validate(const ModelConfig *config, char **error_out);

// Preset configs
ModelConfig *model_config_llama2_7b(void);
ModelConfig *model_config_llama2_13b(void);
ModelConfig *model_config_llama3_8b(void);
ModelConfig *model_config_mistral_7b(void);
ModelConfig *model_config_qwen2_7b(void);
ModelConfig *model_config_bert_base(void);
ModelConfig *model_config_vit_base(void);
ModelConfig *model_config_sdxl(void);

// Copy
ModelConfig *model_config_copy(const ModelConfig *config);

// Merge (overlay second onto first)
void model_config_merge(ModelConfig *base, const ModelConfig *override);

#ifdef __cplusplus
}
#endif

#endif // NEURAL_CORE_KERNEL_MODEL_CONFIG_H