# API Reference

## C API

### Tensor (`arix_tensor.h`)

```c
ArixTensor* arix_tensor_create(const size_t* shape, size_t ndim, ArixDtype dtype);
void arix_tensor_destroy(ArixTensor* t);
ArixTensor* arix_tensor_clone(const ArixTensor* t);
int arix_tensor_reshape(ArixTensor* t, const size_t* new_shape, size_t new_ndim);
int arix_tensor_copy(ArixTensor* dst, const ArixTensor* src);
void arix_tensor_fill(ArixTensor* t, float value);
```

### Memory (`arix_memory.h`)

```c
void* arix_malloc(size_t size, size_t alignment);
void arix_free(void* ptr, size_t size);
```

### Autodiff (`arix_autodiff.h`) — v0.1 stub

```c
ArixVariable* arix_variable_create(ArixTensor* value);
void arix_variable_destroy(ArixVariable* v);
void arix_tape_reset(void);
void arix_backward(ArixVariable* loss);
```

### Optimizer (`arix_optimizer.h`)

```c
ArixOptimizer* arix_optimizer_create(ArixOptimizerType type, float learning_rate, float momentum, float weight_decay, float grad_clip);
void arix_optimizer_destroy(ArixOptimizer* opt);
void arix_optimizer_zero_grad(ArixOptimizer* opt);
int arix_optimizer_step(ArixOptimizer* opt, ArixTensor* param, ArixTensor* grad);
```

### HSS (`arix_hss.h`)

```c
ArixHSSLayer* arix_hss_layer_create(const ArixHSSConfig* config, unsigned int seed);
void arix_hss_layer_destroy(ArixHSSLayer* layer);
ArixHSSModel* arix_hss_model_create(const ArixHSSConfig* config, unsigned int seed);
void arix_hss_model_destroy(ArixHSSModel* model);
int arix_hss_forward(ArixHSSModel* model, const ArixTensor* input, ArixTensor** output);
```

### SER (`arix_ser.h`)

```c
ArixExpert* arix_expert_create(const ArixSERConfig* config, unsigned int seed, ArixActivation activation);
void arix_expert_destroy(ArixExpert* expert);
ArixSERLayer* arix_ser_layer_create(const ArixSERConfig* config, unsigned int seed);
void arix_ser_layer_destroy(ArixSERLayer* layer);
int arix_ser_forward(ArixSERLayer* layer, const ArixTensor* input, ArixTensor** output);
float arix_ser_load_balance_loss(const ArixSERLayer* layer, const ArixTensor* routing_weights);
ArixSERModel* arix_ser_model_create(const ArixSERConfig* config, unsigned int seed, size_t num_layers);
void arix_ser_model_destroy(ArixSERModel* model);
```

### ARC (`arix_arc.h`)

```c
ArixARCLayer* arix_arc_layer_create(const ArixARCConfig* config, size_t input_dim, size_t output_dim, unsigned int seed);
void arix_arc_layer_destroy(ArixARCLayer* layer);
int arix_arc_forward(ArixARCLayer* layer, const ArixTensor* input, ArixTensor** output, float metrics[4]);
float arix_arc_guard_score(const ArixARCLayer* layer, const ArixTensor* input);
float arix_arc_verify_score(const ArixARCLayer* layer, const ArixTensor* output);
int arix_arc_simulate_attack(ArixARCLayer* layer, const ArixTensor* input, int attack_type, float eps, ArixTensor** adversarial);
```

### NPE (`arix_npe.h`)

```c
ArixNPEProgram* arix_npe_program_create(size_t max_instructions);
void arix_npe_program_destroy(ArixNPEProgram* prog);
void arix_npe_program_append(ArixNPEProgram* prog, ArixNPEInstruction inst);
ArixNPEVM* arix_npe_vm_create(const ArixNPEConfig* config);
void arix_npe_vm_destroy(ArixNPEVM* vm);
void arix_npe_vm_load(ArixNPEVM* vm, ArixNPEProgram* program);
int arix_npe_vm_run(ArixNPEVM* vm, const ArixTensor* input, ArixTensor** output);
ArixNPEProgram* arix_npe_compile_mlp(size_t input_dim, size_t hidden_dim);
ArixNPEProgram* arix_npe_compile_attention(size_t dim);
int arix_npe_verify_program(const ArixNPEProgram* prog);
```

### FM (`arix_fm.h`)

```c
ArixFMConfig arix_fm_config_default(void);
ArixFMMemoryBank* arix_fm_memory_bank_create(size_t capacity, size_t dim);
void arix_fm_memory_bank_destroy(ArixFMMemoryBank* bank);
int arix_fm_memory_bank_query(ArixFMMemoryBank* bank, const ArixTensor* key, ArixTensor** value, float* similarity);
int arix_fm_memory_bank_write(ArixFMMemoryBank* bank, const ArixTensor* key, const ArixTensor* value);
ArixFMNode* arix_fm_node_create(size_t node_id, size_t mem_capacity, size_t mem_dim);
void arix_fm_node_destroy(ArixFMNode* node);
ArixFMController* arix_fm_controller_create(const ArixFMConfig* config);
void arix_fm_controller_destroy(ArixFMController* ctrl);
int arix_fm_sync_all_reduce(ArixFMController* ctrl, ArixFMNode** nodes, size_t num_nodes, float privacy_epsilon);
int arix_fm_sync_gossip(ArixFMController* ctrl, ArixFMNode* node, ArixFMNode* peer, size_t dim);
int arix_fm_sync_ring(ArixFMController* ctrl, ArixFMNode** nodes, size_t num_nodes, size_t dim);
ArixTensor* arix_fm_compress_gradients(const ArixTensor* grad, float compression_ratio);
int arix_fm_forward(ArixFMController* ctrl, size_t node_id, const ArixTensor* input, ArixTensor** output);
```

### Unified Model (`arix_arch.h`)

```c
typedef struct {
    ArixHSSConfig hss_config;
    ArixSERConfig ser_config;
    ArixARCConfig arc_config;
    ArixNPEConfig npe_config;
    ArixFMConfig fm_config;
    size_t input_dim;
    size_t output_dim;
    unsigned int seed;
} ArixArchConfig;

ArixArchConfig arix_arch_config_default(void);
ArixModel* arix_model_create(const ArixArchConfig* config);
void arix_model_destroy(ArixModel* model);
int arix_model_forward(ArixModel* model, const ArixTensor* input, ArixTensor** output);
```

### Training (`arix_train.h`)

```c
typedef struct {
    float learning_rate;
    size_t num_epochs;
    size_t batch_size;
    size_t log_interval;
    size_t save_interval;
} ArixTrainConfig;

ArixTrainConfig arix_train_config_default(void);
ArixTrainer* arix_trainer_create(ArixModel* model, const ArixTrainConfig* config);
void arix_trainer_destroy(ArixTrainer* trainer);
float arix_trainer_train_step(ArixTrainer* trainer, const ArixTensor* input, const ArixTensor* target);
float arix_trainer_evaluate(ArixTrainer* trainer, const ArixTensor* input, const ArixTensor* target);
int arix_trainer_save_checkpoint(ArixTrainer* trainer, const char* path);
int arix_trainer_load_checkpoint(ArixTrainer* trainer, const char* path);
```

## Python API

### Tensor

```python
import numpy as np
from arix_algo import Tensor

t = Tensor(np.random.randn(4, 8).astype(np.float32))
arr = t.numpy()  # → np.ndarray
```

### Model

```python
from arix_algo import Model

model = Model({'input_dim': 8, 'output_dim': 8})
out = model.forward(np.random.randn(1, 4, 8).astype(np.float32))
# out → Tensor with shape (4, 8)
```

### Trainer

```python
from arix_algo import Model, Trainer

model = Model({'input_dim': 8, 'output_dim': 8})
trainer = Trainer(model, {'learning_rate': 0.01, 'num_epochs': 10})

# Single train step
loss = trainer.train_step(input_data, target_data)

# Evaluate
avg_loss = trainer.evaluate(input_data, target_data)

# Checkpoint
trainer.save("checkpoint.bin")
trainer.load("checkpoint.bin")
```
