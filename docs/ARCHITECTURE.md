# Architecture

ARIX-Algo is a composable 5-component neural pipeline with training loop and Python bindings.

## Pipeline

```
                    ┌─────────────┐
                    │   Input     │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  HSS Layer  │  Hierarchical State Space
                    │  (SSM)      │  Multi-layer, ZOH discretization
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  SER Layer  │  Sparse Expert Routing
                    │  (MoE)      │  Softmax + top-k, load balance
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  ARC Layer  │  Adversarial Robustness
                    │  (Guard)    │  I/O guard, gradient obfuscation
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  NPE VM     │  Neural Program Executor
                    │  (Program)  │  14-opcode register VM
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  FM Node    │  Federated Memory
                    │  (Sync)     │  Trust-weighted all-reduce
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │   Output    │
                    └─────────────┘
```

## Directory Structure

```
ARIX_Algo/
├── CMakeLists.txt              # Top-level build
├── src/
│   ├── core/                   # Foundation layer
│   │   ├── include/            # Core headers (tensor, memory, autodiff, optimizer)
│   │   └── src/                # Core implementations
│   ├── arch/                   # Architecture layer
│   │   ├── include/            # Component headers (HSS, SER, ARC, NPE, FM, train)
│   │   ├── src/
│   │   │   ├── hss/            # State-space model
│   │   │   ├── ser/            # Sparse expert routing
│   │   │   ├── arc/            # Adversarial robustness
│   │   │   ├── npe/            # Neural program VM + compiler
│   │   │   ├── fm/             # Federated memory + sync
│   │   │   ├── arch/           # ArixModel pipeline composition
│   │   │   └── train/          # Training loop (trainer)
│   │   └── include/            # Public arch headers
│   └── python/                 # Python bindings
│       ├── bindings.cpp        # pybind11 C++ extension
│       ├── setup.py            # pip build config
│       └── arix_algo/          # Python package
│           ├── __init__.py
│           ├── tensor.py
│           ├── model.py
│           └── train.py
├── tests/
│   ├── unit/                   # Per-component unit tests (25 tests)
│   │   ├── core/               # test_tensor
│   │   ├── hss/                # test_hss_layer, test_hss_model
│   │   ├── ser/                # test_ser_expert, test_ser_layer, etc.
│   │   ├── arc/                # test_arc_attack, test_arc_layer, etc.
│   │   ├── npe/                # test_npe_program, test_npe_vm, etc.
│   │   ├── fm/                 # test_fm_memory_bank, test_fm_sync, etc.
│   │   └── train/              # test_trainer
│   ├── integration/            # Stack integration tests (4 tests)
│   └── python/                 # Python tests (5 tests)
├── examples/                   # Component demos (HSS, SER, ARC, NPE, FM)
└── docs/                       # Documentation
```

## Component Details

### Core (`src/core/`)

| Module | Files | Description |
|--------|-------|-------------|
| Tensor | `arix_tensor.h`, `tensor.c` | N-dimensional array, row-major, strides, dtype |
| Memory | `arix_memory.h`, `memory.c` | Aligned allocation, secure-zero on free |
| Thread | `arix_thread.h`, `thread.c` | Thread pool stub (v0.1) |
| Autodiff | `arix_autodiff.h`, `variable.c`, `tape.c`, `ops.c` | Autograd stubs (v0.1, backward is no-op) |
| Optimizer | `arix_optimizer.h`, `optimizer.c` | SGD with momentum, weight decay, gradient clipping |

### HSS — Hierarchical State Space (`src/arch/src/hss/`)

- Converts continuous SSM parameters (A, B, C, D, dt) to discrete via ZOH
- Sequential scan for state propagation
- Hierarchical scan stub for multi-resolution
- Layer forward: input projection → layernorm → discretize → scan
- Multi-layer model: stacks multiple HSS layers with residual

### SER — Sparse Expert Routing (`src/arch/src/ser/`)

- Expert types: ReLU, GELU, Swish (configurable per layer)
- Routing: softmax → top-k selection (greedy or noisy)
- Layer forward: route input → gather experts → weighted combine with dropout
- Load balance: auxiliary loss penalizing unbalanced routing

### ARC — Adversarial Robustness Core (`src/arch/src/arc/`)

- Input guard: L2 norm vs running stats, flags input as anomalous
- Gradient obfuscator: configurable noise + magnitude clamping
- Output verifier: affine+ReLU projection, cosine consistency check, history smoothing
- Security scoring: 4 metrics (guard, verify, statistical, combined)
- Attack simulation: FGSM, PGD, C&W (white-box)

### NPE — Neural Program Executor (`src/arch/src/npe/`)

- 16-register file, 64K memory pool
- 14 opcodes: NOP, LOAD, STORE, ADD, MUL, MATMUL, RELU, SOFTMAX, LAYERNORM, ATTENTION, BRANCH, HALT
- Compiler: `npe_compile_mlp` (2-layer MLP with ReLU), `npe_compile_attention` (QK^T → softmax → @V)
- Static verifier: halting, register bounds, memory bounds, opcode validity

### FM — Federated Memory (`src/arch/src/fm/`)

- Memory bank: key-value store with euclidean similarity search, LRU eviction, retention-based forgetting
- Node: per-node memory bank + gradient accumulator + trust score
- Controller: federation manager with parameterized config
- Sync: all-reduce (trust-weighted avg + Laplace DP noise + conflict detection), gossip, ring topology, gradient compression via sign+threshold

### Training (`src/arch/src/train/`)

- Trainer: wraps model with optimizer, provides `train_step`, `evaluate`, `save_checkpoint`, `load_checkpoint`
- SGD optimizer with momentum, weight_decay, gradient clipping
- Save/load: binary checkpoint format (config + weights)

## Python Bindings

The Python layer wraps the C API via pybind11:

```
Python           C++
──────────────────────────
Tensor   ───→  (numpy array)
Model    ───→  PyModel ← arix_model_create/create/forward
Trainer  ───→  PyTrainer ← arix_trainer_create
ArchConfig ──→ ArixArchConfig
TrainConfig ──→ ArixTrainConfig
```
