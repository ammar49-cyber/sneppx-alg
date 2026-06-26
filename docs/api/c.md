# C API Reference

## Tensor API

**Header**: `#include "arix_tensor.h"`

### Types

```c
typedef struct ArixTensor ArixTensor;
typedef enum {
    ARIX_BOOL,      // int8_t (0/1)
    ARIX_INT32,     // int32_t
    ARIX_INT64,     // int64_t
    ARIX_FLOAT32,   // float
    ARIX_FLOAT64    // double
} ArixDType;
```

### Creation

```c
ArixTensor* arix_tensor_create(const size_t* shape, size_t ndim, ArixDType dtype);
ArixTensor* arix_tensor_zeros(const size_t* shape, size_t ndim, ArixDType dtype);
ArixTensor* arix_tensor_ones(const size_t* shape, size_t ndim, ArixDType dtype);
ArixTensor* arix_tensor_full(const size_t* shape, size_t ndim, ArixDType dtype, double value);
ArixTensor* arix_tensor_randn(const size_t* shape, size_t ndim, ArixDType dtype);
ArixTensor* arix_tensor_from_buffer(const void* data, const size_t* shape, size_t ndim, ArixDType dtype);
ArixTensor* arix_tensor_arange(double start, double step, size_t n, ArixDType dtype);
ArixTensor* arix_tensor_identity(size_t n, ArixDType dtype);
ArixTensor* arix_tensor_copy(const ArixTensor* src);
```

### Shape Operations

```c
ArixTensor* arix_tensor_reshape(const ArixTensor* a, const size_t* shape, size_t ndim);
ArixTensor* arix_tensor_transpose(const ArixTensor* a);
ArixTensor* arix_tensor_permute(const ArixTensor* a, const int32_t* axes, size_t ndim);
ArixTensor* arix_tensor_expand(const ArixTensor* a, const size_t* shape, size_t ndim);
ArixTensor* arix_tensor_squeeze(const ArixTensor* a);
ArixTensor* arix_tensor_unsqueeze(const ArixTensor* a, int32_t axis);
ArixTensor* arix_tensor_slice(const ArixTensor* a, int32_t dim, int32_t start, int32_t end, int32_t step);
ArixTensor* arix_tensor_pad(const ArixTensor* a, const int32_t* pad_before, const int32_t* pad_after, size_t ndim);
```

### Arithmetic Operations

```c
ArixTensor* arix_tensor_add(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_sub(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_mul(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_div(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_pow(const ArixTensor* a, double exponent);
ArixTensor* arix_tensor_sqrt(const ArixTensor* a);
ArixTensor* arix_tensor_neg(const ArixTensor* a);
ArixTensor* arix_tensor_abs(const ArixTensor* a);
```

### Linear Algebra

```c
ArixTensor* arix_tensor_matmul(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_batch_matmul(const ArixTensor* a, const ArixTensor* b);
```

### Reduction Operations

```c
ArixTensor* arix_tensor_sum(const ArixTensor* a, int32_t axis);
ArixTensor* arix_tensor_mean(const ArixTensor* a, int32_t axis);
ArixTensor* arix_tensor_max(const ArixTensor* a, int32_t axis);
ArixTensor* arix_tensor_min(const ArixTensor* a, int32_t axis);
ArixTensor* arix_tensor_var(const ArixTensor* a, int32_t axis);
int32_t arix_tensor_argmax(const ArixTensor* a);
int32_t arix_tensor_argmin(const ArixTensor* a);
```

### Comparison Operations

```c
ArixTensor* arix_tensor_eq(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_neq(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_lt(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_gt(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_le(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_ge(const ArixTensor* a, const ArixTensor* b);
```

### Unary Operations

```c
ArixTensor* arix_tensor_exp(const ArixTensor* a);
ArixTensor* arix_tensor_log(const ArixTensor* a);
ArixTensor* arix_tensor_sin(const ArixTensor* a);
ArixTensor* arix_tensor_cos(const ArixTensor* a);
ArixTensor* arix_tensor_tan(const ArixTensor* a);
ArixTensor* arix_tensor_sigmoid(const ArixTensor* a);
ArixTensor* arix_tensor_relu(const ArixTensor* a);
ArixTensor* arix_tensor_softmax(const ArixTensor* a);
ArixTensor* arix_tensor_gelu(const ArixTensor* a);
ArixTensor* arix_tensor_layer_norm(const ArixTensor* a, double eps);
```

### Utility

```c
void   arix_tensor_print(const ArixTensor* a);
void   arix_tensor_save(const ArixTensor* a, const char* filename);
ArixTensor* arix_tensor_load(const char* filename);
void   arix_tensor_destroy(ArixTensor* t);
void   arix_tensor_to_host(const ArixTensor* a, void* host_buf);
void   arix_tensor_fill(ArixTensor* a, double value);
size_t arix_tensor_num_elements(const ArixTensor* a);
size_t arix_tensor_size_in_bytes(const ArixTensor* a);

int32_t    arix_tensor_ndim(const ArixTensor* a);
const size_t* arix_tensor_shape(const ArixTensor* a);
ArixDType  arix_tensor_dtype(const ArixTensor* a);
```

## Autodiff API

**Header**: `#include "arix_autodiff.h"`

```c
// Variables
ArixVariable* arix_variable_create(ArixTensor* data, bool requires_grad);
void arix_variable_destroy(ArixVariable* var);
ArixTensor* arix_variable_data(ArixVariable* var);
ArixTensor* arix_variable_grad(ArixVariable* var);
void arix_variable_zero_grad(ArixVariable* var);

// Tape
ArixTape* arix_tape_create(void);
void arix_tape_record(ArixTape* tape, const char* op_name,
                       ArixVariable** inputs, int32_t num_inputs,
                       ArixVariable** outputs, int32_t num_outputs,
                       ArixGradFn grad_fn);
void arix_tape_backward(ArixTape* tape, ArixVariable* loss);
void arix_tape_destroy(ArixTape* tape);

// No-grad context
void arix_no_grad_begin(void);
void arix_no_grad_end(void);
bool arix_no_grad_is_active(void);

// Gradient functions (for tape recording)
typedef void (*ArixGradFn)(ArixVariable** inputs, int32_t num_inputs,
                            ArixVariable** outputs, int32_t num_outputs,
                            ArixTensor* grad_output);
```

## Optimizer API

**Header**: `#include "arix_optimizer.h"`

```c
ArixOptimizer* arix_optimizer_create(double lr);
void arix_optimizer_add_parameter(ArixOptimizer* opt, ArixVariable* var);
void arix_optimizer_step(ArixOptimizer* opt);
void arix_optimizer_zero_grad(ArixOptimizer* opt);
void arix_optimizer_destroy(ArixOptimizer* opt);
```

## Memory API

**Header**: `#include "arix_memory.h"`

```c
void* arix_malloc_aligned(size_t size, size_t alignment);
void* arix_malloc_aligned_guarded(size_t size, size_t alignment);
void  arix_free_aligned(void* ptr);
void  arix_secure_zero(void* ptr, size_t size);
```

## Thread Pool API

**Header**: `#include "arix_thread_pool.h"`

```c
typedef void (*ArixTaskFn)(void* arg);

ArixThreadPool* arix_thread_pool_create(int32_t num_threads);
void arix_thread_pool_enqueue(ArixThreadPool* pool, ArixTaskFn fn, void* arg);
void arix_thread_pool_wait(ArixThreadPool* pool);
void arix_thread_pool_destroy(ArixThreadPool* pool);
```

## HSS API

**Header**: `#include "arix_hss.h"`

```c
ArixHSSConfig arix_hss_config_default(void);
ArixHSSModel* arix_hss_model_create(ArixHSSConfig* config, int32_t seed);
void arix_hss_forward(ArixHSSModel* model, ArixTensor* input, ArixTensor** output);
void arix_hss_model_destroy(ArixHSSModel* model);
```

## SER API

**Header**: `#include "arix_ser.h"`

```c
ArixSERConfig arix_ser_config_default(void);
ArixSERModel* arix_ser_model_create(ArixSERConfig* config, int32_t seed);
void arix_ser_forward(ArixSERModel* model, ArixTensor* input, ArixTensor** output);
void arix_ser_compute_load_balance_loss(ArixSERModel* model, float* loss);
void arix_ser_model_destroy(ArixSERModel* model);
```

## ARC API

**Header**: `#include "arix_arc.h"`

```c
ArixARCConfig arix_arc_config_default(void);
ArixARCModel* arix_arc_model_create(ArixARCConfig* config);
void arix_arc_forward(ArixARCModel* model, ArixTensor* input,
                      ArixTensor** output, ArixTensor** anomaly_scores);
void arix_arc_attack_fgsm(ArixARCModel* model, ArixTensor* input,
                          ArixTensor* gradient, float epsilon);
void arix_arc_attack_pgd(ArixARCModel* model, ArixTensor* input,
                         ArixTensor* gradient, float epsilon, int32_t steps);
void arix_arc_model_destroy(ArixARCModel* model);
```

## NPE API

**Header**: `#include "arix_npe.h"`

```c
ArixNPEConfig arix_npe_config_default(void);
ArixNPEModel* arix_npe_model_create(ArixNPEConfig* config);
int32_t arix_npe_load_program(ArixNPEModel* model, uint8_t* program, int32_t size);
void arix_npe_execute(ArixNPEModel* model, ArixTensor** outputs, int32_t num_outputs);
int32_t arix_npe_verify(ArixNPEModel* model);
void arix_npe_model_destroy(ArixNPEModel* model);
```

## FM API

**Header**: `#include "arix_fm.h"`

```c
ArixFMConfig arix_fm_config_default(void);
ArixFMModel* arix_fm_model_create(ArixFMConfig* config, int32_t node_id, int32_t seed);
void arix_fm_write(ArixFMModel* model, ArixTensor* key, ArixTensor* value);
void arix_fm_read(ArixFMModel* model, ArixTensor* query, ArixTensor** output);
void arix_fm_sync(ArixFMModel* model, ArixTensor* global_bank,
                  int32_t* trust_scores, int32_t num_nodes);
void arix_fm_model_destroy(ArixFMModel* model);
```

## Security API

### S0 — Crypto

**Header**: `#include "arix_s0_crypto.h"`

```c
// Ed25519
void arix_ed25519_keypair(uint8_t pk[32], uint8_t sk[64]);
void arix_ed25519_sign(uint8_t sig[64], const uint8_t msg[], size_t msglen,
                       const uint8_t sk[64], const uint8_t pk[32]);
int32_t arix_ed25519_verify(const uint8_t sig[64], const uint8_t msg[], size_t msglen,
                            const uint8_t pk[32]);
void arix_ed25519_pk_from_sk(uint8_t pk[32], const uint8_t sk[64]);

// X25519
void arix_x25519_keypair(uint8_t pk[32], uint8_t sk[32]);
void arix_x25519_shared_secret(uint8_t shared[32], const uint8_t sk[32], const uint8_t pk[32]);

// ChaCha20-Poly1305
void arix_chacha20_poly1305_encrypt(uint8_t* ct, size_t* ctlen,
                                     const uint8_t* pt, size_t ptlen,
                                     const uint8_t key[32], const uint8_t nonce[12],
                                     const uint8_t* aad, size_t aadlen);
int32_t arix_chacha20_poly1305_decrypt(uint8_t* pt, size_t* ptlen,
                                        const uint8_t* ct, size_t ctlen,
                                        const uint8_t key[32], const uint8_t nonce[12],
                                        const uint8_t* aad, size_t aadlen);
```

### S1 — Secure Memory

**Header**: `#include "arix_s1_secure_memory.h"`

```c
void* arix_s1_alloc_locked(size_t size);
void  arix_s1_free_locked(void* ptr, size_t size);
int32_t arix_s1_memcmp_consttime(const void* a, const void* b, size_t n);
```

See also `docs/security.md` for full S0-S3 documentation.
