# API Reference

## C API

### Tensor (`SNEPPX_tensor.h`)

```c
SNEPPXTensor* SNEPPX_tensor_create(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
void SNEPPX_tensor_destroy(SNEPPXTensor* t);
SNEPPXTensor* SNEPPX_tensor_clone(const SNEPPXTensor* t);
int SNEPPX_tensor_reshape(SNEPPXTensor* t, const size_t* new_shape, size_t new_ndim);
int SNEPPX_tensor_copy(SNEPPXTensor* dst, const SNEPPXTensor* src);
void SNEPPX_tensor_fill(SNEPPXTensor* t, float value);
```

### Memory (`SNEPPX_memory.h`)

```c
void* SNEPPX_malloc(size_t size, size_t alignment);
void SNEPPX_free(void* ptr, size_t size);
```

### Autodiff (`SNEPPX_autodiff.h`)

```c
SNEPPXVariable* SNEPPX_variable_create(SNEPPXTensor* data, int requires_grad);
void SNEPPX_variable_destroy(SNEPPXVariable* var);
void SNEPPX_variable_set_requires_grad(SNEPPXVariable* var, int requires_grad);
SNEPPXVariable* SNEPPX_variable_detach(SNEPPXVariable* var);
SNEPPXVariable* SNEPPX_variable_copy(SNEPPXVariable* var);
void SNEPPX_variable_zero_grad(SNEPPXVariable* var);
float SNEPPX_variable_item(SNEPPXVariable* var);
size_t SNEPPX_variable_numel(SNEPPXVariable* var);

SNEPPXTape* SNEPPX_tape_create(void);
void SNEPPX_tape_destroy(SNEPPXTape* tape);
void SNEPPX_tape_record(SNEPPXTape* tape, SNEPPXVariable* var);
void SNEPPX_tape_backward(SNEPPXTape* tape, SNEPPXVariable* loss);
void SNEPPX_tape_zero_grad(SNEPPXTape* tape);
float SNEPPX_tape_global_norm(SNEPPXTape* tape);
void SNEPPX_tape_clip_grad_norm(SNEPPXTape* tape, float max_norm);

void SNEPPX_no_grad_enter(void);
void SNEPPX_no_grad_exit(void);
int SNEPPX_no_grad_is_active(void);

SNEPPXVariable* SNEPPX_add(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_sub(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_mul(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_div(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_pow(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_neg(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_matmul(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_mse_loss(SNEPPXTape* tape, SNEPPXVariable* pred, SNEPPXVariable* target);
SNEPPXVariable* SNEPPX_relu(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_gelu(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_silu(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_sigmoid(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_tanh(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_softmax(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
SNEPPXVariable* SNEPPX_exp(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_log(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_sum(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
SNEPPXVariable* SNEPPX_mean(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
SNEPPXVariable* SNEPPX_transpose(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim1, size_t dim2);
SNEPPXVariable* SNEPPX_dropout(SNEPPXTape* tape, SNEPPXVariable* a, float rate, unsigned int seed);
SNEPPXVariable* SNEPPX_layer_norm(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* gamma, SNEPPXVariable* beta, float eps);
SNEPPXVariable* SNEPPX_conv2d(SNEPPXTape* tape, SNEPPXVariable* input, SNEPPXVariable* kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w);
SNEPPXVariable* SNEPPX_concat(SNEPPXTape* tape, SNEPPXVariable** vars, size_t num_vars, size_t dim);
```

### Optimizer (`SNEPPX_optimizer.h`)

```c
SNEPPXOptimizerConfig SNEPPX_optimizer_config_default(void);
SNEPPXOptimizer* SNEPPX_optimizer_create(const SNEPPXOptimizerConfig* config);
void SNEPPX_optimizer_destroy(SNEPPXOptimizer* opt);
void SNEPPX_optimizer_step(SNEPPXOptimizer* opt, SNEPPXTensor** params, SNEPPXTensor** grads, size_t num_params);

// Convenience factory functions:
SNEPPXOptimizer* SNEPPX_sgd_create(float lr, float momentum, float weight_decay);
SNEPPXOptimizer* SNEPPX_adam_create(float lr, float beta1, float beta2, float eps, float weight_decay);
SNEPPXOptimizer* SNEPPX_adamw_create(float lr, float beta1, float beta2, float eps, float weight_decay);
SNEPPXOptimizer* SNEPPX_rmsprop_create(float lr, float alpha, float eps, float momentum, float weight_decay);
SNEPPXOptimizer* SNEPPX_adagrad_create(float lr, float eps, float weight_decay);

// LR Schedulers:
SNEPPXLRScheduler* SNEPPX_lr_scheduler_step_lr(float* lr_ptr, float gamma, size_t step_size);
SNEPPXLRScheduler* SNEPPX_lr_scheduler_exponential(float* lr_ptr, float gamma);
SNEPPXLRScheduler* SNEPPX_lr_scheduler_cosine(float* lr_ptr, float min_lr, float max_lr, size_t total_steps);
SNEPPXLRScheduler* SNEPPX_lr_scheduler_reduce_on_plateau(float* lr_ptr, float factor, size_t patience, int mode_min);
void SNEPPX_lr_scheduler_destroy(SNEPPXLRScheduler* sched);
void SNEPPX_lr_scheduler_step(SNEPPXLRScheduler* sched, float current_loss);
```

### HSS (`SNEPPX_hss.h`)

```c
SNEPPXHSSLayer* SNEPPX_hss_layer_create(const SNEPPXHSSConfig* config, unsigned int seed);
void SNEPPX_hss_layer_destroy(SNEPPXHSSLayer* layer);
SNEPPXHSSModel* SNEPPX_hss_model_create(const SNEPPXHSSConfig* config, unsigned int seed);
void SNEPPX_hss_model_destroy(SNEPPXHSSModel* model);
int SNEPPX_hss_forward(SNEPPXHSSModel* model, const SNEPPXTensor* input, SNEPPXTensor** output);
```

### SER (`SNEPPX_ser.h`)

```c
SNEPPXExpert* SNEPPX_expert_create(const SNEPPXSERConfig* config, unsigned int seed, SNEPPXActivation activation);
void SNEPPX_expert_destroy(SNEPPXExpert* expert);
SNEPPXSERLayer* SNEPPX_ser_layer_create(const SNEPPXSERConfig* config, unsigned int seed);
void SNEPPX_ser_layer_destroy(SNEPPXSERLayer* layer);
int SNEPPX_ser_forward(SNEPPXSERLayer* layer, const SNEPPXTensor* input, SNEPPXTensor** output);
float SNEPPX_ser_load_balance_loss(const SNEPPXSERLayer* layer, const SNEPPXTensor* routing_weights);
SNEPPXSERModel* SNEPPX_ser_model_create(const SNEPPXSERConfig* config, unsigned int seed, size_t num_layers);
void SNEPPX_ser_model_destroy(SNEPPXSERModel* model);
```

### ARC (`SNEPPX_arc.h`)

```c
SNEPPXARCLayer* SNEPPX_arc_layer_create(const SNEPPXARCConfig* config, size_t input_dim, size_t output_dim, unsigned int seed);
void SNEPPX_arc_layer_destroy(SNEPPXARCLayer* layer);
int SNEPPX_arc_forward(SNEPPXARCLayer* layer, const SNEPPXTensor* input, SNEPPXTensor** output, float metrics[4]);
float SNEPPX_arc_guard_score(const SNEPPXARCLayer* layer, const SNEPPXTensor* input);
float SNEPPX_arc_verify_score(const SNEPPXARCLayer* layer, const SNEPPXTensor* output);
int SNEPPX_arc_simulate_attack(SNEPPXARCLayer* layer, const SNEPPXTensor* input, int attack_type, float eps, SNEPPXTensor** adversarial);
```

### NPE (`SNEPPX_npe.h`)

```c
SNEPPXNPEProgram* SNEPPX_npe_program_create(size_t max_instructions);
void SNEPPX_npe_program_destroy(SNEPPXNPEProgram* prog);
void SNEPPX_npe_program_append(SNEPPXNPEProgram* prog, SNEPPXNPEInstruction inst);
SNEPPXNPEVM* SNEPPX_npe_vm_create(const SNEPPXNPEConfig* config);
void SNEPPX_npe_vm_destroy(SNEPPXNPEVM* vm);
void SNEPPX_npe_vm_load(SNEPPXNPEVM* vm, SNEPPXNPEProgram* program);
int SNEPPX_npe_vm_run(SNEPPXNPEVM* vm, const SNEPPXTensor* input, SNEPPXTensor** output);
SNEPPXNPEProgram* SNEPPX_npe_compile_mlp(size_t input_dim, size_t hidden_dim);
SNEPPXNPEProgram* SNEPPX_npe_compile_attention(size_t dim);
int SNEPPX_npe_verify_program(const SNEPPXNPEProgram* prog);
```

### FM (`SNEPPX_fm.h`)

```c
SNEPPXFMConfig SNEPPX_fm_config_default(void);
SNEPPXFMMemoryBank* SNEPPX_fm_memory_bank_create(size_t capacity, size_t dim);
void SNEPPX_fm_memory_bank_destroy(SNEPPXFMMemoryBank* bank);
int SNEPPX_fm_memory_bank_query(SNEPPXFMMemoryBank* bank, const SNEPPXTensor* key, SNEPPXTensor** value, float* similarity);
int SNEPPX_fm_memory_bank_write(SNEPPXFMMemoryBank* bank, const SNEPPXTensor* key, const SNEPPXTensor* value);
SNEPPXFMNode* SNEPPX_fm_node_create(size_t node_id, size_t mem_capacity, size_t mem_dim);
void SNEPPX_fm_node_destroy(SNEPPXFMNode* node);
SNEPPXFMController* SNEPPX_fm_controller_create(const SNEPPXFMConfig* config);
void SNEPPX_fm_controller_destroy(SNEPPXFMController* ctrl);
int SNEPPX_fm_sync_all_reduce(SNEPPXFMController* ctrl, SNEPPXFMNode** nodes, size_t num_nodes, float privacy_epsilon);
int SNEPPX_fm_sync_gossip(SNEPPXFMController* ctrl, SNEPPXFMNode* node, SNEPPXFMNode* peer, size_t dim);
int SNEPPX_fm_sync_ring(SNEPPXFMController* ctrl, SNEPPXFMNode** nodes, size_t num_nodes, size_t dim);
SNEPPXTensor* SNEPPX_fm_compress_gradients(const SNEPPXTensor* grad, float compression_ratio);
int SNEPPX_fm_forward(SNEPPXFMController* ctrl, size_t node_id, const SNEPPXTensor* input, SNEPPXTensor** output);
```

### Unified Model (`SNEPPX_arch.h`)

```c
typedef struct {
    SNEPPXHSSConfig hss_config;
    SNEPPXSERConfig ser_config;
    SNEPPXARCConfig arc_config;
    SNEPPXNPEConfig npe_config;
    SNEPPXFMConfig fm_config;
    size_t input_dim;
    size_t output_dim;
    unsigned int seed;
} SNEPPXArchConfig;

SNEPPXArchConfig SNEPPX_arch_config_default(void);
SNEPPXModel* SNEPPX_model_create(const SNEPPXArchConfig* config);
void SNEPPX_model_destroy(SNEPPXModel* model);
int SNEPPX_model_forward(SNEPPXModel* model, const SNEPPXTensor* input, SNEPPXTensor** output);
```

### Training (`SNEPPX_train.h`)

```c
typedef struct {
    float learning_rate;
    size_t num_epochs;
    size_t batch_size;
    size_t log_interval;
    size_t save_interval;
} SNEPPXTrainConfig;

SNEPPXTrainConfig SNEPPX_train_config_default(void);
SNEPPXTrainer* SNEPPX_trainer_create(SNEPPXModel* model, const SNEPPXTrainConfig* config);
void SNEPPX_trainer_destroy(SNEPPXTrainer* trainer);
float SNEPPX_trainer_train_step(SNEPPXTrainer* trainer, const SNEPPXTensor* input, const SNEPPXTensor* target);
float SNEPPX_trainer_evaluate(SNEPPXTrainer* trainer, const SNEPPXTensor* input, const SNEPPXTensor* target);
int SNEPPX_trainer_save_checkpoint(SNEPPXTrainer* trainer, const char* path);
int SNEPPX_trainer_load_checkpoint(SNEPPXTrainer* trainer, const char* path);
```

## Python API

### Tensor

```python
from SneppX_ALG import Tensor

t = Tensor(np.random.randn(4, 8).astype(np.float32))
arr = t.numpy()          # → np.ndarray
val = t.item()            # → float (scalar)
t2 = t.to('cuda')         # device transfer
t3 = Tensor.zeros(4, 8)   # static factory
t4 = Tensor.ones(4, 8)
t5 = Tensor.randn(4, 8)

# Operator overloads:
c = a + b          # __add__
c = a - b          # __sub__
c = a * b          # __mul__
c = a / b          # __truediv__
c = a @ b          # __matmul__
c = -a             # __neg__
c = a ** 2         # __pow__
c = a[0:2]         # __getitem__
```

### Model

```python
from SneppX_ALG import Model

model = Model({'input_dim': 8, 'output_dim': 8})
out = model.forward(np.random.randn(1, 4, 8).astype(np.float32))
# out → Tensor with shape (4, 8)
model.train()  # training mode
model.eval()   # evaluation mode
```

### Linear Layer

```python
from SneppX_ALG import Linear

layer = Linear(8, 16)  # in_features, out_features
out = layer(torch.randn(4, 8))
params = layer.parameters()  # [weight, bias]
```

### Sequential

```python
from SneppX_ALG import Sequential, Linear

net = Sequential(
    Linear(8, 32),
    Linear(32, 16),
    Linear(16, 4),
)
out = net(torch.randn(4, 8))
```

### Optimizer

```python
from SneppX_ALG import Optimizer

opt = Optimizer(params, lr=0.001, optimizer_type='adam')
opt.step()
opt.zero_grad()
```

### Trainer

```python
from SneppX_ALG import Model, Trainer

model = Model({'input_dim': 8, 'output_dim': 8})
trainer = Trainer(model, {'learning_rate': 0.01, 'num_epochs': 10})

loss = trainer.train_step(input_data, target_data)
avg_loss = trainer.evaluate(input_data, target_data)
trainer.save("checkpoint.bin")
trainer.load("checkpoint.bin")
```

### ARC (Adversarial Robustness)

```python
from SneppX_ALG import ARCLayer

layer = ARCLayer(input_dim=16, output_dim=16)
output = layer.forward(input_array)
adversarial = layer.simulate_attack(input_array, attack_type=1, epsilon=0.1)
```

### NPE (Neural Processing Engine)

```python
from SneppX_ALG import NPEVM

vm = NPEVM()
vm.load_program(program_bytes)
output = vm.run(input_array)
```

### FM (Federated Memory)

```python
from SneppX_ALG import FMController

ctrl = FMController(num_nodes=4, memory_dim=64, memory_capacity=100)
output = ctrl.forward(node_id=0, input_array)
ctrl.sync()
```

### SER (Sparse Expert Routing)

```python
from SneppX_ALG import SERModel

ser = SERModel(num_experts=8, num_active=2, input_dim=32, expert_dim=64, output_dim=32)
output = ser.forward(input_array)
params = ser.parameters()
```
