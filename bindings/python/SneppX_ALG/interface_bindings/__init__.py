from .. import _neural_engine_bridge

from .tensor import (
    Tensor,
    Dtype,
    Device,
    Layout,
    Tensorable,
    _HAS_C_BACKEND,
)
from .nn import (
    Module,
    Linear,
    Embedding,
    Dropout,
    LayerNorm,
    RMSNorm,
    GELU,
    SiLU,
    ReLU,
    Sigmoid,
    Tanh,
    Sequential,
    MultiheadAttention,
    TransformerBlock,
    Transformer,
)
from .optim import (
    Optimizer,
    SGD,
    AdamW,
    Lion,
    LAMB,
    CosineAnnealingLR,
)
from .data import (
    Dataset,
    TensorDataset,
    TextDataset,
    SimpleTokenizer,
    BatchCollator,
    Preprocessor,
)
from .distributed import (
    DistributedContext,
    DistributedSampler,
    DistributedDataParallel,
    init_process_group,
    destroy_process_group,
    get_world_size,
    get_rank,
    barrier,
    all_reduce,
    launch,
)
from .hf_integration import (
    load_hf_model,
    save_hf_model,
    load_config,
)
from .model import (
    Model,
    ModelConfig,
    HSSModel,
    HSSConfig,
    SERModel,
    SERConfig,
    ARCModel,
    ARCConfig,
    NPEModel,
    NPEConfig,
    FMModel,
    FMConfig,
)
from .quantization import (
    QuantMode,
    QuantGranularity,
    quantize_int8_sym,
    dequantize_int8_sym,
    quantize_int8_asym,
    dequantize_int8_asym,
    quantize_int8_channel,
    dequantize_int8_channel,
    quantize_int4_sym,
    dequantize_int4_sym,
    quantize_fp8_e4m3,
    dequantize_fp8_e4m3,
    quantize_fp8_e5m2,
    dequantize_fp8_e5m2,
    awq_scale_weights,
    awq_quantize,
    gptq_compute_hessian,
    gptq_quantize,
    QuantizedLinear,
    quantize_error,
)
from .amp import autocast, GradScaler
from .schedulers import (
    LRScheduler, StepLR, ExponentialLR, CosineAnnealingLR,
    CosineAnnealingWarmRestarts, ConstantLRWithWarmup,
    LinearWarmupCosineDecay, PolynomialLR, OneCycleLR,
    ReduceLROnPlateau, CosineAnnealingWithWarmup, SequentialLR,
    ChainedScheduler, TriStageLR, get_scheduler,
)
from .optim_extra import (
    Lion, LAMB, LARS, AdaFactor, RAdam, Sophia, Adan,
    ScheduleFreeAdamW, get_optimizer as get_extra_optimizer,
)
from .optim_advanced import (
    SM3, Demeter, CaProp, SOAP, DistributedAdam, OrthoAdam,
    get_optimizer as get_advanced_optimizer,
)
from .grad_checkpoint import (
    GradientCheckpointer, checkpoint, CheckpointSegment,
    checkpoint_sequential,
)
from .data_loader import (
    Dataset, TensorDataset, MemoryMappedTextDataset,
    DistributedSampler, DataLoader, default_collate,
    StreamingTokenDataset, StreamingTokenDataset,
)
from .profiler import Profiler, ProfileEntry, Timer, MemoryTracker, TrainProfiler, timeit, get_profiler
from .model_zoo import (
    ModelFamily, LlamaConfig, MistralConfig, Qwen2Config, DeepSeekV2Config,
    get_model_config, config_from_json, list_available_models,
    read_safetensors, convert_hf_to_sneppx,
    build_model_from_config, build_transformer_from_config,
    from_pretrained, HF_WEIGHT_MAP,
)
from .model_implementations import (
    BertModel, BertForMaskedLM, BertConfig,
    GPTModel, GPTLMHeadModel, GPTConfig,
    T5Model, T5ForConditionalGeneration, T5Config,
    create_bert_model, create_gpt_model, create_t5_model,
    get_model_config,
    count_parameters, get_model_size_mb,
)
from .advanced_ops import (
    conv2d, conv1d, max_pool2d, avg_pool2d,
    adaptive_avg_pool2d, adaptive_max_pool2d,
    rnn_cell, lstm_cell, gru_cell,
    multi_head_attention, transformer_block,
    softmax, layernorm, rmsnorm,
)
from .augmentation import (
    Compose, RandomApply, RandomChoice, RandomOrder,
    random_resized_crop, random_horizontal_flip, random_vertical_flip,
    random_rotation, color_jitter, random_grayscale, gaussian_blur,
    solarize, posterize, auto_contrast, equalize, invert, sharpness,
    mixup, cutmix, augmix, cutout, random_erasing,
    random_token_mask, random_word_dropout, random_word_shuffle,
    token_cutout, time_stretch, pitch_shift, time_mask, freq_mask,
    center_crop, random_crop, five_crop, ten_crop,
    IMAGENET_TRAIN_TRANSFORMS, IMAGENET_EVAL_TRANSFORMS,
    CIFAR10_TRAIN_TRANSFORMS, CIFAR10_EVAL_TRANSFORMS,
)
from .onnx_export import (
    OnnxExporter, OnnxImporter, OnnxModel, OnnxGraph,
    OnnxNode, OnnxTensor, OnnxInitializer,
    export_onnx, import_onnx, SNEPPX_TO_ONNX_OP,
    ONNX_OP_REGISTRY, TensorRTExporter,
    onnx_validate, onnx_version_compatible, upgrade_opset,
)
from .amp import autocast, GradScaler
from .grad_checkpoint import checkpoint, GradientCheckpointer
from .schedulers import (
    LRScheduler,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ConstantLRWithWarmup,
    LinearWarmupCosineDecay,
    PolynomialLR,
    OneCycleLR,
    ReduceLROnPlateau,
    CosineAnnealingWithWarmupRestarts,
    SequentialLR,
    ChainedScheduler,
    TriStageLR,
    get_scheduler,
)
from .optim_extra import (
    Lion,
    LAMB,
    LARS,
    AdaFactor,
    RAdam,
    Sophia,
    Adan,
    ScheduleFreeAdamW,
    get_optimizer as get_extra_optimizer,
)
from .trainer_v2 import UltraTrainConfig, UltraTrainer
from .checkpoint import (
    CheckpointWriter,
    CheckpointReader,
    CheckpointCoordinator,
    HeartbeatMonitor,
    ElasticTrainer,
    FaultToleranceManager,
    validate_checkpoint,
    CheckpointHeader,
    TensorRecord,
)
from .quantization import (
    QuantMode,
    QuantGranularity,
    quantize_int8_sym,
    dequantize_int8_sym,
    quantize_int8_asym,
    dequantize_int8_asym,
    quantize_int8_channel,
    dequantize_int8_channel,
    quantize_int4_sym,
    dequantize_int4_sym,
    quantize_fp8_e4m3,
    dequantize_fp8_e4m3,
    quantize_fp8_e5m2,
    dequantize_fp8_e5m2,
    awq_scale_weights,
    awq_quantize,
    gptq_compute_hessian,
    gptq_quantize,
    QuantizedLinear,
    quantize_error,
)
from .profiler import (
    Profiler,
    ProfileEntry,
    Timer,
    MemoryTracker,
    TrainProfiler,
    timeit, get_profiler, _GLOBAL_PROFILER,
)
from .data_loader import (
    Dataset,
    TensorDataset,
    MemoryMappedTextDataset,
    DistributedSampler,
    DataLoader,
    default_collate,
)
from .pruning import (
    magnitude_prune, l1_channel_prune, taylor_pruning, global_magnitude_prune,
    movement_pruning, soft_pruning,
    compute_sparsity, count_parameters, print_pruning_summary,
    apply_pruning_mask, recover_pruned_weights,
    distillation_loss, prune_and_distill,
    find_winning_ticket, rewrite_weights,
)
from .distillation import (
    kd_loss, attention_transfer_loss, feature_matching_loss,
    correlation_congruence_loss, hint_loss, crd_loss,
    multi_teacher_distillation_loss, ensemble_teacher_distillation,
    OnlineDistillation, DistillationPruner,
    distill_bert, distill_gpt,
)
from .train import (
    Trainer,
    TrainConfig,
    Optimizer as CppOptimizer,
    SGD as CppSGD,
    Adam as CppAdam,
    AdamW as CppAdamW,
    StepLR,
    ExponentialLR,
    CosineLR,
    ReduceLROnPlateau,
    MSELoss,
    CrossEntropyLoss,
    MAELoss,
    NLLLoss,
    KLDivLoss,
    BCELoss,
    Dataset as CppDataset,
    TensorDataset as CppTensorDataset,
    DataLoader,
)

__all__ = [
    # tensor
    'Tensor', 'Dtype', 'Device', 'Layout', 'Tensorable', '_HAS_C_BACKEND',
    # nn
    'Module', 'Linear', 'Embedding', 'Dropout',
    'LayerNorm', 'RMSNorm',
    'GELU', 'SiLU', 'ReLU', 'Sigmoid', 'Tanh',
    'Sequential', 'MultiheadAttention', 'TransformerBlock', 'Transformer',
    # optim
    'Optimizer', 'SGD', 'AdamW', 'Lion', 'LAMB', 'LARS', 'AdaFactor',
    'RAdam', 'Sophia', 'Adan', 'ScheduleFreeAdamW',
    'CosineAnnealingLR',
    # data
    'Dataset', 'TensorDataset', 'TextDataset',
    'SimpleTokenizer', 'BatchCollator', 'Preprocessor',
    # distributed
    'DistributedContext', 'DistributedSampler', 'DistributedDataParallel',
    'init_process_group', 'destroy_process_group',
    'get_world_size', 'get_rank', 'barrier', 'all_reduce', 'launch',
    # hf
    'load_hf_model', 'save_hf_model', 'load_config',
    # model
    'Model', 'ModelConfig',
    'HSSModel', 'HSSConfig',
    'SERModel', 'SERConfig',
    'ARCModel', 'ARCConfig',
    'NPEModel', 'NPEConfig',
    'FMModel', 'FMConfig',
    # quantization
    'QuantMode', 'QuantGranularity',
    'quantize_int8_sym', 'dequantize_int8_sym',
    'quantize_int8_asym', 'dequantize_int8_asym',
    'quantize_int8_channel', 'dequantize_int8_channel',
    'quantize_int4_sym', 'dequantize_int4_sym',
    'quantize_fp8_e4m3', 'dequantize_fp8_e4m3',
    'quantize_fp8_e5m2', 'dequantize_fp8_e5m2',
    'awq_scale_weights', 'awq_quantize',
    'gptq_compute_hessian', 'gptq_quantize',
    'QuantizedLinear', 'quantize_error',
    # checkpoint
    'CheckpointWriter', 'CheckpointReader', 'CheckpointCoordinator',
    'HeartbeatMonitor', 'ElasticTrainer', 'FaultToleranceManager',
    'validate_checkpoint', 'CheckpointHeader', 'TensorRecord',
    # profiler
    'Profiler', 'ProfileEntry', 'Timer', 'MemoryTracker', 'TrainProfiler',
    'timeit', 'get_profiler', '_GLOBAL_PROFILER',
# model zoo
    'ModelFamily', 'LlamaConfig', 'MistralConfig', 'Qwen2Config', 'DeepSeekV2Config',
    'get_model_config', 'config_from_json', 'list_available_models',
    'read_safetensors', 'convert_hf_to_sneppx',
    'build_model_from_config', 'build_transformer_from_config',
    'from_pretrained', 'HF_WEIGHT_MAP',
    # model implementations
    'BertModel', 'BertForMaskedLM', 'BertConfig',
    'GPTModel', 'GPTLMHeadModel', 'GPTConfig',
    'T5Model', 'T5ForConditionalGeneration', 'T5Config',
    'create_bert_model', 'create_gpt_model', 'create_t5_model',
    'get_model_config',
    'count_parameters', 'get_model_size_mb',
    # augmentation
    'Compose', 'RandomApply', 'RandomChoice', 'RandomOrder',
    'random_resized_crop', 'random_horizontal_flip', 'random_vertical_flip',
    'random_rotation', 'color_jitter', 'random_grayscale', 'gaussian_blur',
    'solarize', 'posterize', 'auto_contrast', 'equalize', 'invert', 'sharpness',
    'mixup', 'cutmix', 'augmix', 'cutout', 'random_erasing',
    'random_token_mask', 'random_word_dropout', 'random_word_shuffle',
    'token_cutout', 'time_stretch', 'pitch_shift', 'time_mask', 'freq_mask',
    'center_crop', 'random_crop', 'five_crop', 'ten_crop',
    'IMAGENET_TRAIN_TRANSFORMS', 'IMAGENET_EVAL_TRANSFORMS',
    'CIFAR10_TRAIN_TRANSFORMS', 'CIFAR10_EVAL_TRANSFORMS',
    # pruning
    'magnitude_prune', 'l1_channel_prune', 'taylor_pruning', 'global_magnitude_prune',
    'movement_pruning', 'soft_pruning',
    'compute_sparsity', 'count_parameters', 'print_pruning_summary',
    'apply_pruning_mask', 'recover_pruned_weights',
    'distillation_loss', 'prune_and_distill',
    'find_winning_ticket', 'rewrite_weights',
    # distillation
    'kd_loss', 'attention_transfer_loss', 'feature_matching_loss',
    'correlation_congruence_loss', 'hint_loss', 'crd_loss',
    'multi_teacher_distillation_loss', 'ensemble_teacher_distillation',
    'OnlineDistillation', 'DistillationPruner',
    'distillation_loss', 'distill_bert', 'distill_gpt',
    # train (C++ backend)
    'Trainer', 'TrainConfig',
    'CppOptimizer', 'CppSGD', 'CppAdam', 'CppAdamW',
    'StepLR', 'ExponentialLR', 'CosineLR', 'ReduceLROnPlateau',
    'MSELoss', 'CrossEntropyLoss', 'MAELoss', 'NLLLoss', 'KLDivLoss', 'BCELoss',
    'CppDataset', 'CppTensorDataset', 'DataLoader',
    '_neural_engine_bridge',
]
