# C API Reference

## Tensor API

**Header**: `#include "SNEPPX_tensor.h"`

### Types

```c
typedef struct SNEPPXTensor SNEPPXTensor;
typedef enum {
    SNEPPX_FLOAT32,
    SNEPPX_FLOAT64,
    SNEPPX_FLOAT16,
    SNEPPX_BFLOAT16,
    SNEPPX_FLOAT8,
    SNEPPX_INT32,
    SNEPPX_INT64,
    SNEPPX_INT16,
    SNEPPX_INT8,
    SNEPPX_UINT8,
    SNEPPX_BOOL,
    SNEPPX_COMPLEX64,
    SNEPPX_COMPLEX128,
} SNEPPXDtype;
```

### Creation

```c
SNEPPXTensor* SNEPPX_tensor_create(const size_t* shape, size_t ndim, SNEPPXDType dtype);
SNEPPXTensor* SNEPPX_tensor_zeros(const size_t* shape, size_t ndim, SNEPPXDType dtype);
SNEPPXTensor* SNEPPX_tensor_ones(const size_t* shape, size_t ndim, SNEPPXDType dtype);
SNEPPXTensor* SNEPPX_tensor_full(const size_t* shape, size_t ndim, SNEPPXDType dtype, double value);
SNEPPXTensor* SNEPPX_tensor_randn(const size_t* shape, size_t ndim, SNEPPXDType dtype);
SNEPPXTensor* SNEPPX_tensor_from_buffer(const void* data, const size_t* shape, size_t ndim, SNEPPXDType dtype);
SNEPPXTensor* SNEPPX_tensor_arange(double start, double step, size_t n, SNEPPXDType dtype);
SNEPPXTensor* SNEPPX_tensor_identity(size_t n, SNEPPXDType dtype);
SNEPPXTensor* SNEPPX_tensor_copy(const SNEPPXTensor* src);
```

### Shape Operations

```c
SNEPPXTensor* SNEPPX_tensor_reshape(const SNEPPXTensor* a, const size_t* shape, size_t ndim);
SNEPPXTensor* SNEPPX_tensor_transpose(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_permute(const SNEPPXTensor* a, const int32_t* axes, size_t ndim);
SNEPPXTensor* SNEPPX_tensor_expand(const SNEPPXTensor* a, const size_t* shape, size_t ndim);
SNEPPXTensor* SNEPPX_tensor_squeeze(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_unsqueeze(const SNEPPXTensor* a, int32_t axis);
SNEPPXTensor* SNEPPX_tensor_slice(const SNEPPXTensor* a, int32_t dim, int32_t start, int32_t end, int32_t step);
SNEPPXTensor* SNEPPX_tensor_pad(const SNEPPXTensor* a, const int32_t* pad_before, const int32_t* pad_after, size_t ndim);
```

### Arithmetic Operations

```c
SNEPPXTensor* SNEPPX_tensor_add(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_sub(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_mul(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_div(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_pow(const SNEPPXTensor* a, double exponent);
SNEPPXTensor* SNEPPX_tensor_sqrt(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_neg(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_abs(const SNEPPXTensor* a);
```

### Linear Algebra

```c
SNEPPXTensor* SNEPPX_tensor_matmul(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_batch_matmul(const SNEPPXTensor* a, const SNEPPXTensor* b);
```

### Reduction Operations

```c
SNEPPXTensor* SNEPPX_tensor_sum(const SNEPPXTensor* a, int32_t axis);
SNEPPXTensor* SNEPPX_tensor_mean(const SNEPPXTensor* a, int32_t axis);
SNEPPXTensor* SNEPPX_tensor_max(const SNEPPXTensor* a, int32_t axis);
SNEPPXTensor* SNEPPX_tensor_min(const SNEPPXTensor* a, int32_t axis);
SNEPPXTensor* SNEPPX_tensor_var(const SNEPPXTensor* a, int32_t axis);
int32_t SNEPPX_tensor_argmax(const SNEPPXTensor* a);
int32_t SNEPPX_tensor_argmin(const SNEPPXTensor* a);
```

### Comparison Operations

```c
SNEPPXTensor* SNEPPX_tensor_eq(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_neq(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_lt(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_gt(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_le(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_ge(const SNEPPXTensor* a, const SNEPPXTensor* b);
```

### Unary Operations

```c
SNEPPXTensor* SNEPPX_tensor_exp(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_log(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_sin(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_cos(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_tan(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_sigmoid(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_relu(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_softmax(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_gelu(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_layer_norm(const SNEPPXTensor* a, double eps);
```

### Utility

```c
void   SNEPPX_tensor_print(const SNEPPXTensor* a);
void   SNEPPX_tensor_save(const SNEPPXTensor* a, const char* filename);
SNEPPXTensor* SNEPPX_tensor_load(const char* filename);
void   SNEPPX_tensor_destroy(SNEPPXTensor* t);
void   SNEPPX_tensor_to_host(const SNEPPXTensor* a, void* host_buf);
void   SNEPPX_tensor_fill(SNEPPXTensor* a, double value);
size_t SNEPPX_tensor_num_elements(const SNEPPXTensor* a);
size_t SNEPPX_tensor_size_in_bytes(const SNEPPXTensor* a);

int32_t    SNEPPX_tensor_ndim(const SNEPPXTensor* a);
const size_t* SNEPPX_tensor_shape(const SNEPPXTensor* a);
SNEPPXDType  SNEPPX_tensor_dtype(const SNEPPXTensor* a);
```

## Autodiff API

**Header**: `#include "SNEPPX_autodiff.h"`

```c
// Variables
SNEPPXVariable* SNEPPX_variable_create(SNEPPXTensor* data, bool requires_grad);
void SNEPPX_variable_destroy(SNEPPXVariable* var);
SNEPPXTensor* SNEPPX_variable_data(SNEPPXVariable* var);
SNEPPXTensor* SNEPPX_variable_grad(SNEPPXVariable* var);
void SNEPPX_variable_zero_grad(SNEPPXVariable* var);

// Tape
SNEPPXTape* SNEPPX_tape_create(void);
void SNEPPX_tape_record(SNEPPXTape* tape, const char* op_name,
                       SNEPPXVariable** inputs, int32_t num_inputs,
                       SNEPPXVariable** outputs, int32_t num_outputs,
                       SNEPPXGradFn grad_fn);
void SNEPPX_tape_backward(SNEPPXTape* tape, SNEPPXVariable* loss);
void SNEPPX_tape_destroy(SNEPPXTape* tape);

// No-grad context
void SNEPPX_no_grad_begin(void);
void SNEPPX_no_grad_end(void);
bool SNEPPX_no_grad_is_active(void);

// Gradient functions (for tape recording)
typedef void (*SNEPPXGradFn)(SNEPPXVariable** inputs, int32_t num_inputs,
                            SNEPPXVariable** outputs, int32_t num_outputs,
                            SNEPPXTensor* grad_output);
```

## Optimizer API

**Header**: `#include "SNEPPX_optimizer.h"`

```c
SNEPPXOptimizer* SNEPPX_optimizer_create(double lr);
void SNEPPX_optimizer_add_parameter(SNEPPXOptimizer* opt, SNEPPXVariable* var);
void SNEPPX_optimizer_step(SNEPPXOptimizer* opt);
void SNEPPX_optimizer_zero_grad(SNEPPXOptimizer* opt);
void SNEPPX_optimizer_destroy(SNEPPXOptimizer* opt);
```

## Memory API

**Header**: `#include "SNEPPX_memory.h"`

```c
void* SNEPPX_malloc_aligned(size_t size, size_t alignment);
void* SNEPPX_malloc_aligned_guarded(size_t size, size_t alignment);
void  SNEPPX_free_aligned(void* ptr);
void  SNEPPX_secure_zero(void* ptr, size_t size);
```

## Thread Pool API

**Header**: `#include "SNEPPX_thread_pool.h"`

```c
typedef void (*SNEPPXTaskFn)(void* arg);

SNEPPXThreadPool* SNEPPX_thread_pool_create(int32_t num_threads);
void SNEPPX_thread_pool_enqueue(SNEPPXThreadPool* pool, SNEPPXTaskFn fn, void* arg);
void SNEPPX_thread_pool_wait(SNEPPXThreadPool* pool);
void SNEPPX_thread_pool_destroy(SNEPPXThreadPool* pool);
```

## HSS API

**Header**: `#include "SNEPPX_hss.h"`

```c
SNEPPXHSSConfig SNEPPX_hss_config_default(void);
SNEPPXHSSModel* SNEPPX_hss_model_create(SNEPPXHSSConfig* config, int32_t seed);
void SNEPPX_hss_forward(SNEPPXHSSModel* model, SNEPPXTensor* input, SNEPPXTensor** output);
void SNEPPX_hss_model_destroy(SNEPPXHSSModel* model);
```

## SER API

**Header**: `#include "SNEPPX_ser.h"`

```c
SNEPPXSERConfig SNEPPX_ser_config_default(void);
SNEPPXSERModel* SNEPPX_ser_model_create(SNEPPXSERConfig* config, int32_t seed);
void SNEPPX_ser_forward(SNEPPXSERModel* model, SNEPPXTensor* input, SNEPPXTensor** output);
void SNEPPX_ser_compute_load_balance_loss(SNEPPXSERModel* model, float* loss);
void SNEPPX_ser_model_destroy(SNEPPXSERModel* model);
```

## ARC API

**Header**: `#include "SNEPPX_arc.h"`

```c
SNEPPXARCConfig SNEPPX_arc_config_default(void);
SNEPPXARCModel* SNEPPX_arc_model_create(SNEPPXARCConfig* config);
void SNEPPX_arc_forward(SNEPPXARCModel* model, SNEPPXTensor* input,
                      SNEPPXTensor** output, SNEPPXTensor** anomaly_scores);
void SNEPPX_arc_attack_fgsm(SNEPPXARCModel* model, SNEPPXTensor* input,
                          SNEPPXTensor* gradient, float epsilon);
void SNEPPX_arc_attack_pgd(SNEPPXARCModel* model, SNEPPXTensor* input,
                         SNEPPXTensor* gradient, float epsilon, int32_t steps);
void SNEPPX_arc_model_destroy(SNEPPXARCModel* model);
```

## NPE API

**Header**: `#include "SNEPPX_npe.h"`

```c
SNEPPXNPEConfig SNEPPX_npe_config_default(void);
SNEPPXNPEModel* SNEPPX_npe_model_create(SNEPPXNPEConfig* config);
int32_t SNEPPX_npe_load_program(SNEPPXNPEModel* model, uint8_t* program, int32_t size);
void SNEPPX_npe_execute(SNEPPXNPEModel* model, SNEPPXTensor** outputs, int32_t num_outputs);
int32_t SNEPPX_npe_verify(SNEPPXNPEModel* model);
void SNEPPX_npe_model_destroy(SNEPPXNPEModel* model);
```

## FM API

**Header**: `#include "SNEPPX_fm.h"`

```c
SNEPPXFMConfig SNEPPX_fm_config_default(void);
SNEPPXFMModel* SNEPPX_fm_model_create(SNEPPXFMConfig* config, int32_t node_id, int32_t seed);
void SNEPPX_fm_write(SNEPPXFMModel* model, SNEPPXTensor* key, SNEPPXTensor* value);
void SNEPPX_fm_read(SNEPPXFMModel* model, SNEPPXTensor* query, SNEPPXTensor** output);
void SNEPPX_fm_sync(SNEPPXFMModel* model, SNEPPXTensor* global_bank,
                  int32_t* trust_scores, int32_t num_nodes);
void SNEPPX_fm_model_destroy(SNEPPXFMModel* model);
```

## Security API

### S0 — Crypto

**Header**: `#include "SNEPPX_crypto.h"` (includes all sub-headers)

```c
// Ed25519
int SNEPPX_ed25519_keypair_generate(SNEPPXEd25519Keypair* kp);
int SNEPPX_ed25519_secret_key_expand(uint8_t* expanded_sk, const uint8_t* seed);
int SNEPPX_ed25519_sign(const SNEPPXEd25519Keypair* kp, const uint8_t* message, size_t msg_len, SNEPPXEd25519Signature* sig);
int SNEPPX_ed25519_verify(const uint8_t* public_key, const uint8_t* message, size_t msg_len, const SNEPPXEd25519Signature* sig);
int SNEPPX_ed25519_scalar_multiply(uint8_t* result, const uint8_t* scalar, const uint8_t* point);

// ChaCha20-Poly1305 (AEAD)
int SNEPPX_aead_encrypt(uint8_t* ct, size_t* ctlen, const uint8_t* pt, size_t ptlen,
                       const uint8_t key[32], const uint8_t nonce[12],
                       const uint8_t* aad, size_t aadlen);
int SNEPPX_aead_decrypt(uint8_t* pt, size_t* ptlen, const uint8_t* ct, size_t ctlen,
                       const uint8_t key[32], const uint8_t nonce[12],
                       const uint8_t* aad, size_t aadlen);

// Hashing
void SNEPPX_sha512(const uint8_t* input, size_t len, uint8_t* output);
void SNEPPX_sha3_256(const uint8_t* input, size_t len, uint8_t* output);
void SNEPPX_blake3(const uint8_t* input, size_t len, uint8_t* output, size_t output_len);

// Key derivation
int SNEPPX_argon2_hash(uint8_t* hash, size_t hash_len, const uint8_t* pwd, size_t pwd_len,
                      const uint8_t* salt, size_t salt_len, uint32_t t_cost, uint32_t m_cost);

// Random
int SNEPPX_random_bytes(uint8_t* buffer, size_t len);

// Constant-time comparison
int SNEPPX_ct_is_zero(const uint8_t* b, size_t n);
int SNEPPX_ct_equal(const uint8_t* a, const uint8_t* b, size_t n);
```

### S1 — Secure Memory

**Header**: `#include "SNEPPX_secure_mem.h"`

```c
void SNEPPX_secure_zero(void* ptr, size_t size);
void* SNEPPX_secure_malloc(size_t size);
void SNEPPX_secure_free(void* ptr, size_t size);
```

See `docs/security.md` for full S0-S3 documentation.

See also `docs/security.md` for full S0-S3 documentation.
