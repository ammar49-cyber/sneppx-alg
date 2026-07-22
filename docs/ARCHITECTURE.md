# SNEPPX-Algo Architecture

## Overview

SNEPPX-Algo implements a composite AI algorithm with cryptographic integrity. The system processes inputs through five sequential algorithm components, each wrapped in four security layers.

```
                     ┌──────────────────────────────────────┐
                     │           Security Layer              │
                      │  S0 Crypto · S1 Secure Mem · S2 Obf  │
                      │  S3 Behavioral Monitor · S4-S9       │
                     └──────────────────────────────────────┘
                                     │
                     ┌───────────────▼──────────────────────┐
                     │         Algorithm Pipeline            │
                     │  HSS → SER → ARC → NPE → FM          │
                     │  (SSM) (MoE) (Guard) (VM)  (Fed Mem) │
                     └──────────────────────────────────────┘
                                     │
                      ┌───────────────▼──────────────────────┐
                      │         Integrity Layer               │
                      │  ZK Proofs (opt-in, SNEPPX_BUILD_ZK) · │
                      │  Formal Safety · On-Device Attestation │
                      └──────────────────────────────────────┘
```

## Algorithm Pipeline

### HSS — Hierarchical State Space

**Layer**: `src/arch/src/hss/`
**Files**: `hss.h`, `hss.c`

HSS is a multi-layer state space model that processes sequences by maintaining a hidden state updated per timestep.

#### Mathematical Formulation

For each layer l at timestep k:

```
x_{k}  ∈ R^{d_model}     (input at timestep k)
h^{l}_k                  (hidden state for layer l, timestep k)
y^{l}_k                  (output for layer l, timestep k)

Transition matrices:
A^{l} ∈ R^{d_state × d_state}
B^{l} ∈ R^{d_state × d_model}
C^{l} ∈ R^{d_model × d_state}
Δ^{l} ∈ R^{d_model × d_model}     (step size)

Discretization (zero-order hold):
Ā = exp(Δ · A)
B̄ = (Δ · A)^{-1} · (exp(Δ · A) - I) · Δ · B

State update:
h^{l}_k = Ā · h^{l}_{k-1} + B̄ · x^{l}_k

Output:
y^{l}_k = C^{l} · h^{l}_k

Merge multi-layer outputs:
y_k = Σ_{l} W^{l} · y^{l}_k + x_k     (residual connection)
```

The sequential scan processes timesteps sequentially. The Blelloch tree-based parallel scan over time is enabled by default on CPU (set `use_parallel_scan=0` in config to disable). A CUDA parallel scan over time is pending.

#### Code Structure

```c
typedef struct {
    int32_t d_model;      // Input/output dimension
    int32_t d_state;      // Hidden state dimension (default: d_model/4)
    int32_t num_layers;   // Number of HSS layers
    float dt_min;         // Minimum step size (default: 0.001)
    float dt_max;         // Maximum step size (default: 0.1)
    int32_t seed;         // Random seed for init
} SNEPPXHSSConfig;

SNEPPXHSSConfig SNEPPX_hss_config_default(void);
SNEPPXHSSModel* SNEPPX_hss_model_create(SNEPPXHSSConfig* config, int32_t seed);
void SNEPPX_hss_forward(SNEPPXHSSModel* model, SNEPPXTensor* input, SNEPPXTensor** output);
void SNEPPX_hss_model_destroy(SNEPPXHSSModel* model);
```

### SER — Sparse Expert Routing

**Layer**: `src/arch/src/ser/`
**Files**: `ser.h`, `ser.c`

SER implements a Mixture-of-Experts layer with top-k routing.

#### Mathematical Formulation

```
Input:  x ∈ R^{batch × seq × d_model}
Experts: E_1, ..., E_n where E_i : R^{d_model} → R^{d_model}
Router:  r(x) = Softmax(W_r · x + b_r)     W_r ∈ R^{n × d_model}

Top-k selection (k << n):
indices = topk(r(x), k)
weights = Softmax(r(x)[indices])

Output:
y = Σ_{i=1}^{k} weights_i · E_{indices_i}(x)
```

Load-balancing loss:

```
ℓ_balance = α · n · Σ_{i=1}^{n} f_i · P_i

where:
f_i = fraction of tokens routed to expert i
P_i = average router probability for expert i
α = balancing coefficient (default: 0.01)
```

#### Code Structure

```c
typedef struct {
    int32_t d_model;         // Input/output dimension
    int32_t d_expert;        // Expert hidden dimension (default: 4*d_model)
    int32_t num_experts;     // Total number of experts (default: 8)
    int32_t top_k;           // Number of active experts per token (default: 2)
    int32_t seed;            // Random seed for expert init
} SNEPPXSERConfig;

SNEPPXSERConfig SNEPPX_ser_config_default(void);
SNEPPXSERModel* SNEPPX_ser_model_create(SNEPPXSERConfig* config, int32_t seed);
void SNEPPX_ser_forward(SNEPPXSERModel* model, SNEPPXTensor* input, SNEPPXTensor** output);
void SNEPPX_ser_compute_load_balance_loss(SNEPPXSERModel* model, float* loss);
void SNEPPX_ser_model_destroy(SNEPPXSERModel* model);
```

### ARC — Adversarial Robustness Core

**Layer**: `src/arch/src/arc/`
**Files**: `arc.h`, `arc.c`

ARC provides a three-layer defense-with-verification pipeline.

#### Defense Layers

1. **Input Guard**: z-score anomaly detection across the input.

```c
// For each feature dimension j:
z_j = (x_j - μ_j) / σ_j
// Flag if |z_j| > threshold (default: 3.0)
```

2. **Gradient Obfuscation**: Noise injection + gradient clipping.

```c
// During training:
g' = g + N(0, σ² · |g|)     // noise proportional to gradient magnitude
g'' = clamp(g', -γ, γ)       // gradient clipping threshold γ
```

3. **Output Verifier**: Consistency check with cosine similarity + temporal smoothing.

```c
// Cosine similarity:
sim = ŷ · y_prev / (|ŷ| · |y_prev|)
// Detect if sim drops below threshold τ_adv (default: 0.5)
// Temporal smoothing:
y_smooth = β · y_prev + (1 - β) · ŷ     // with β ∈ [0.7, 0.99]
```

#### Attack Simulation

ARC includes simulation for:

- **FGSM** (Fast Gradient Sign Method): `x_adv = x + ε · sign(∇_x L(x, y))`
- **PGD** (Projected Gradient Descent): Multi-step FGSM with projection
- **C&W** (Carlini-Wagner): Optimization-based attack: `min ||δ|| + c · f(x + δ)`

#### Code Structure

```c
typedef struct {
    float z_threshold;        // Z-score anomaly threshold (default: 3.0)
    float noise_std;          // Gradient noise std deviation (default: 0.01)
    float grad_clip;          // Gradient clipping threshold (default: 1.0)
    float sim_threshold;      // Cosine similarity threshold (default: 0.5)
    float temporal_beta;      // Temporal smoothing coefficient (default: 0.9)
    int32_t seed;             // Random seed
} SNEPPXARCConfig;

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

### NPE — Neural Program Executor

**Layer**: `src/arch/src/npe/`
**Files**: `npe.h`, `npe.c`

NPE is a 16-register virtual machine with 14 opcodes for executing neural network operations.

#### Virtual Machine

```
Registers: R0-R15, each holds an SNEPPXTensor*
Program Counter: PC
Halt Flag: H

Opcodes:
  0x00  NOP          No operation
  0x01  LOAD rd, imm Load immediate into register
  0x02  MOV rd, rs   Move register
  0x03  MATMUL rd,rs1,rs2    Matrix multiply
  0x04  ADD  rd, rs1, rs2    Element-wise add
  0x05  RELU rd, rs          ReLU activation
  0x06  SOFTMAX rd, rs       Softmax over last dim
  0x07  LAYERNORM rd, rs     Layer normalization
  0x08  SIGMOID rd, rs       Sigmoid activation
  0x13  TANH  rd, rs         Tanh activation
  0x09  SUM  rd, rs          Sum all elements
  0x0A  MEAN rd, rs          Mean of elements
  0x0D  ATTENTION rd, q, k, v  Scaled dot-product attention
  0x10  COMPUTE rd, rs       Compute opcode (extended)
  0x11  HALT                 Stop execution
```

#### Compilers

NPE includes compilers that generate programs from network configurations:

- **Attention Compiler**: Generates QKV projection → scaled dot-product → output projection
- **MLP Compiler**: Generates linear → activation → linear

#### Static Verifier

The verifier checks:
- No register is used before being written
- No out-of-bound register access
- The program terminates (no infinite loops)

#### Code Structure

```c
typedef struct {
    int32_t num_registers;    // Number of registers (default: 16)
    int32_t max_program_size; // Maximum program length (default: 1024)
} SNEPPXNPEConfig;

SNEPPXNPEConfig SNEPPX_npe_config_default(void);
SNEPPXNPEModel* SNEPPX_npe_model_create(SNEPPXNPEConfig* config);
int32_t SNEPPX_npe_load_program(SNEPPXNPEModel* model, uint8_t* program, int32_t size);
void SNEPPX_npe_execute(SNEPPXNPEModel* model, SNEPPXTensor** outputs, int32_t num_outputs);
int32_t SNEPPX_npe_verify(SNEPPXNPEModel* model);
void SNEPPX_npe_model_destroy(SNEPPXNPEModel* model);
```

### FM — Federated Memory

**Layer**: `src/arch/src/fm/`
**Files**: `fm.h`, `fm.c`

FM provides distributed memory with per-node memory banks and trust-weighted synchronization.

#### Memory Bank

Each node maintains a fixed-size memory bank with:
- **Euclidean similarity** for nearest-neighbor retrieval
- **LRU eviction** when the bank is full
- **Learnable read/write** operations

#### Synchronization Protocol

```
Trust-weighted all-reduce:
θ_global = Σ_i trust_score_i · θ_i / Σ_j trust_score_j

Differential privacy noise:
θ_shared = θ_global + Lap(0, ε)     // Laplace noise
ε = noise_scale / sensitivity

Gradient compression:
Select top-k gradients by magnitude (random sampling fallback)
```

#### Code Structure

```c
typedef struct {
    int32_t memory_size;       // Number of memory slots per node (default: 1024)
    int32_t d_model;           // Memory vector dimension
    float lr;                  // Learning rate for memory writes
    float diff_privacy_noise;  // Differential privacy noise scale
    float compression_ratio;   // Gradient compression ratio (default: 0.01)
} SNEPPXFMConfig;

SNEPPXFMConfig SNEPPX_fm_config_default(void);
SNEPPXFMModel* SNEPPX_fm_model_create(SNEPPXFMConfig* config, int32_t node_id, int32_t seed);
void SNEPPX_fm_write(SNEPPXFMModel* model, SNEPPXTensor* key, SNEPPXTensor* value);
void SNEPPX_fm_read(SNEPPXFMModel* model, SNEPPXTensor* query, SNEPPXTensor** output);
void SNEPPX_fm_sync(SNEPPXFMModel* model, SNEPPXTensor* global_bank, int32_t* trust_scores, int32_t num_nodes);
void SNEPPX_fm_model_destroy(SNEPPXFMModel* model);
```

## Security Layers

### S0 — Cryptographic Core

**Status: ✅ Complete**
**Files**: `src/security/c/SNEPPX_s0_*.h/.c`

Production-grade cryptographic primitives:

| Primitive | Implementation | Status |
|-----------|---------------|--------|
| Ed25519 | RFC 8032 | Verified, 304/306 vectors pass |
| X25519 | RFC 7748 | Full DH exchange |
| ChaCha20-Poly1305 | RFC 8439 | AEAD encrypt/decrypt |
| SHA-3 | FIPS 202 | 224/256/384/512 |
| BLAKE3 | Reference impl | Fast hashing |
| Argon2id | RFC 9106 | KDF with timing defense |
| Secure Random | OS CPRG | Entropy source abstraction |

### S1 — Secure Memory

**Status: ✅ Complete**
**Files**: `src/security/c/SNEPPX_s1_*.h/.c`

| Feature | Description |
|---------|-------------|
| Guard Pages | RW pages with PROT_NONE adjacent to detect overflow |
| Canaries | Stack-based overflow detection |
| ASLR | Heap entropy via Windows VirtualAlloc / Linux mmap randomization |
| Locked Memory | mlock/VirtualLock to prevent swapping to disk (CAP_IPC_LOCK) |
| Secure Wipe | memset with compiler barrier to prevent removal |
| Constant-Time Ops | memcmp variant for secret comparison |

### S2 — Obfuscation Engine

**Status: ✅ Complete**
**Files**: `src/security/cpp/SNEPPX_s2_*.h/.cpp`

| Feature | Status |
|---------|--------|
| Control Flow Flattening | Complete |
| String Encryption | Complete |
| Instruction Substitution | Complete |
| Opaque Predicates | Complete |
| VM Obfuscation | Complete |

### S3 — Behavioral Monitor

**Status: ✅ Complete**
**Files**: `src/security/cpp/SNEPPX_s3_*.h/.cpp`

| Feature | Status |
|---------|--------|
| Frequency Analysis | Complete |
| Timing Analysis | Complete |
| Anomaly Detection (ML-based) | Complete |
| Code/Heap Integrity | Complete |

## Foundation Components

### Tensor Core

**Files**: `src/core/tensor.h`, `src/core/tensor.c`

Multi-dimensional array with row-major layout.

```c
typedef enum {
    SNEPPX_BOOL,      // sizeof(int8_t)
    SNEPPX_INT32,     // sizeof(int32_t)
    SNEPPX_INT64,     // sizeof(int64_t)
    SNEPPX_FLOAT32,   // sizeof(float) — default
    SNEPPX_FLOAT64    // sizeof(double)
} SNEPPXDType;
```

**Supported ops (50+):**

- Creation: zeros, ones, full, randn, arange, identity, from_buffer, copy
- Shape: reshape, transpose, permute, expand, squeeze, unsqueeze, slice, pad
- Arithmetic: add, sub, mul, div, pow, sqrt, neg, abs
- Matmul: matmul, batch_matmul
- Reduction: sum, mean, var, max, min, argmax, argmin
- Comparison: eq, ne, lt, gt, le, ge
- Unary: exp, log, sin, cos, tan, sigmoid, relu, softmax, gelu, layer_norm
- Utility: print, save, load, to_host, from_host

### Memory

**Files**: `src/core/memory.h`, `src/core/memory.c`

Aligned allocation on aligned boundaries. Free list with best-fit for sizes > 4KB. Optional guard pages on request.

### Thread Pool

**Files**: `src/core/thread_pool.h`, `src/core/thread_pool.c`

Single-threaded fallback. Interface supports work queue and tasks. Real parallelism deferred to v0.5.

## Data Flow

```
Input (SNEPPXTensor)
    │
    ▼
┌─────────────────────┐
│  HSS Forward         │
│  (sequential scan)   │
│  Output: (B, S, D)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  SER Forward         │
│  (top-k routing)     │
│  Output: (B, S, D)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  ARC Forward         │
│  (guard + filter)    │
│  Output: (B, S, D)   │
│  + anomaly_scores    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  NPE Execute         │
│  (VM program)        │
│  Output: (B, S, D)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  FM Read             │
│  (memory lookup)     │
│  Output: (B, S, D)   │
└─────────┬───────────┘
          │
          ▼
      Output (SNEPPXTensor)
```

## Training Flow

```
Forward pass (HSS → SER → ARC → NPE → FM)
    │
    ▼
Compute loss (MSE + load balance + ARC regularization)
    │
    ▼
Backward pass (autodiff) — real (tape-based reverse-topological gradient
  computation; layer-norm gamma/beta gradients fixed in v0.9.7.890e)
    │
    ▼
Optimizer step (Adam / SGD) — real (in-place parameter update)
    │
    ▼
Secure memory wipe of intermediate tensors
```

## Future Architecture

### Integrity Layer (available via opt-in backends)

```
┌──────────────────────────────────────┐
│         Integrity Layer               │
│  ZK Proofs · Formal Safety ·         │
│  On-Device Attestation               │
└──────────────────────────────────────┘
                    │
    ┌───────────────▼──────────────────────┐
    │         Algorithm Pipeline            │
    │  HSS → SER → ARC → NPE → FM          │
    └──────────────────────────────────────┘
                    │
    ┌───────────────▼──────────────────────┐
    │         Security Layer                │
    │  S0-S9 Full Implementation            │
    └──────────────────────────────────────┘
```

The integrity layer provides:
- **Zero-knowledge proofs** of correct inference — `SNEPPX_BUILD_ZK` enables a real
  Schnorr proof over Curve25519 (p = 2^255 - 19) reference backend.
- **Formal safety verification** of constraint compliance
- **On-device attestation** for distributed verification
