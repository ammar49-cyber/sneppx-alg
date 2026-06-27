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

### Autodiff (`arix_autodiff.h`)

```c
ArixVariable* arix_variable_create(ArixTensor* data, int requires_grad);
void arix_variable_destroy(ArixVariable* var);
void arix_variable_set_requires_grad(ArixVariable* var, int requires_grad);
ArixVariable* arix_variable_detach(ArixVariable* var);
ArixVariable* arix_variable_copy(ArixVariable* var);
void arix_variable_zero_grad(ArixVariable* var);
float arix_variable_item(ArixVariable* var);
size_t arix_variable_numel(ArixVariable* var);

ArixTape* arix_tape_create(void);
void arix_tape_destroy(ArixTape* tape);
void arix_tape_record(ArixTape* tape, ArixVariable* var);
void arix_tape_backward(ArixTape* tape, ArixVariable* loss);
void arix_tape_zero_grad(ArixTape* tape);
float arix_tape_global_norm(ArixTape* tape);
void arix_tape_clip_grad_norm(ArixTape* tape, float max_norm);

void arix_no_grad_enter(void);
void arix_no_grad_exit(void);
int arix_no_grad_is_active(void);

ArixVariable* arix_add(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_sub(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_mul(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_div(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_pow(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_neg(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_matmul(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_mse_loss(ArixTape* tape, ArixVariable* pred, ArixVariable* target);
ArixVariable* arix_relu(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_gelu(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_silu(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_sigmoid(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_tanh(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_softmax(ArixTape* tape, ArixVariable* a, size_t dim);
ArixVariable* arix_exp(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_log(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_sum(ArixTape* tape, ArixVariable* a, size_t dim);
ArixVariable* arix_mean(ArixTape* tape, ArixVariable* a, size_t dim);
ArixVariable* arix_transpose(ArixTape* tape, ArixVariable* a, size_t dim1, size_t dim2);
ArixVariable* arix_dropout(ArixTape* tape, ArixVariable* a, float rate, unsigned int seed);
ArixVariable* arix_layer_norm(ArixTape* tape, ArixVariable* a, ArixVariable* gamma, ArixVariable* beta, float eps);
ArixVariable* arix_conv2d(ArixTape* tape, ArixVariable* input, ArixVariable* kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w);
ArixVariable* arix_concat(ArixTape* tape, ArixVariable** vars, size_t num_vars, size_t dim);
```

### Optimizer (`arix_optimizer.h`)

```c
ArixOptimizerConfig arix_optimizer_config_default(void);
ArixOptimizer* arix_optimizer_create(const ArixOptimizerConfig* config);
void arix_optimizer_destroy(ArixOptimizer* opt);
void arix_optimizer_step(ArixOptimizer* opt, ArixTensor** params, ArixTensor** grads, size_t num_params);

// Convenience factory functions:
ArixOptimizer* arix_sgd_create(float lr, float momentum, float weight_decay);
ArixOptimizer* arix_adam_create(float lr, float beta1, float beta2, float eps, float weight_decay);
ArixOptimizer* arix_adamw_create(float lr, float beta1, float beta2, float eps, float weight_decay);
ArixOptimizer* arix_rmsprop_create(float lr, float alpha, float eps, float momentum, float weight_decay);
ArixOptimizer* arix_adagrad_create(float lr, float eps, float weight_decay);

// LR Schedulers:
ArixLRScheduler* arix_lr_scheduler_step_lr(float* lr_ptr, float gamma, size_t step_size);
ArixLRScheduler* arix_lr_scheduler_exponential(float* lr_ptr, float gamma);
ArixLRScheduler* arix_lr_scheduler_cosine(float* lr_ptr, float min_lr, float max_lr, size_t total_steps);
ArixLRScheduler* arix_lr_scheduler_reduce_on_plateau(float* lr_ptr, float factor, size_t patience, int mode_min);
void arix_lr_scheduler_destroy(ArixLRScheduler* sched);
void arix_lr_scheduler_step(ArixLRScheduler* sched, float current_loss);
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
from arix_algo import Tensor

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
from arix_algo import Model

model = Model({'input_dim': 8, 'output_dim': 8})
out = model.forward(np.random.randn(1, 4, 8).astype(np.float32))
# out → Tensor with shape (4, 8)
model.train()  # training mode
model.eval()   # evaluation mode
```

### Linear Layer

```python
from arix_algo import Linear

layer = Linear(8, 16)  # in_features, out_features
out = layer(torch.randn(4, 8))
params = layer.parameters()  # [weight, bias]
```

### Sequential

```python
from arix_algo import Sequential, Linear

net = Sequential(
    Linear(8, 32),
    Linear(32, 16),
    Linear(16, 4),
)
out = net(torch.randn(4, 8))
```

### Optimizer

```python
from arix_algo import Optimizer

opt = Optimizer(params, lr=0.001, optimizer_type='adam')
opt.step()
opt.zero_grad()
```

### Trainer

```python
from arix_algo import Model, Trainer

model = Model({'input_dim': 8, 'output_dim': 8})
trainer = Trainer(model, {'learning_rate': 0.01, 'num_epochs': 10})

loss = trainer.train_step(input_data, target_data)
avg_loss = trainer.evaluate(input_data, target_data)
trainer.save("checkpoint.bin")
trainer.load("checkpoint.bin")
```

### ARC (Adversarial Robustness)

```python
from arix_algo import ARCLayer

layer = ARCLayer(input_dim=16, output_dim=16)
output = layer.forward(input_array)
adversarial = layer.simulate_attack(input_array, attack_type=1, epsilon=0.1)
```

### NPE (Neural Processing Engine)

```python
from arix_algo import NPEVM

vm = NPEVM()
vm.load_program(program_bytes)
output = vm.run(input_array)
```

### FM (Federated Memory)

```python
from arix_algo import FMController

ctrl = FMController(num_nodes=4, memory_dim=64, memory_capacity=100)
output = ctrl.forward(node_id=0, input_array)
ctrl.sync()
```

### SER (Sparse Expert Routing)

```python
from arix_algo import SERModel

ser = SERModel(num_experts=8, num_active=2, input_dim=32, expert_dim=64, output_dim=32)
output = ser.forward(input_array)
params = ser.parameters()
```
