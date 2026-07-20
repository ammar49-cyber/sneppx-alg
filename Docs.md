# SNEPPX-Alg Project Documentation

This file contains practical documentation and quick start guides for working with the SNEPPX-Alg project.

## Quick Start Guide

### 5-Minute Setup

```bash
# Clone the project
git clone https://github.com/ammar49-cyber/sneppx-alg.git
cd sneppx-alg

# Configure build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build everything
cmake --build build --config Release -j$(nproc)

# Test everything
cd build && ctest -C Release --output-on-failure

# Install Python bindings
pip install -e bindings/python

# You're ready to go!
```

### Python Usage

```python
# Import the high-level API
from SneppX_ALG import Tensor, Linear, AdamW, TrainConfig, Trainer

# Create a simple neural network
model = Linear(1024, 512)
optimizer = AdamW(model.parameters(), lr=0.001)

# Training loop
x = Tensor.randn(32, 1024)
y = Tensor.randn(32, 512)
config = TrainConfig(learning_rate=0.001, batch_size=32, max_steps=100)
trainer = Trainer(model, config)
trainer.fit(x, y)

# Inference
output = model(x)
```

### C/C++ Integration

```c
#include "include/neural_core/tensor.h"
#include <stdio.h>

int main() {
    // Create a tensor
    SNEPPX_Tensor* t = sneppx_tensor_create(2, 3);
    
    // Fill with zeros
    sneppx_tensor_fill(t, 0.0f);
    
    // Operations
    sneppx_tensor_print(t);
    
    // Clean up
    sneppX_tensor_free(t);
    return 0;
}
```

### Rust Bindings

```rust
use neural_core_algo::Tensor;

fn main() {
    // Create a tensor
    let mut t = Tensor::new(vec![2, 3]);
    
    // Use it
    let ones = Tensor::ones(vec![2, 3]);
    
    println!("Shape: {:?}", ones.shape());
    
    // Clean up happens automatically
}
```

## Build Configuration Options

### Debug vs Release

**Debug Build (for development):**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
```

**Release Build (for production):**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### Special Build Options

```bash
# Build with Python bindings
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_PYTHON=ON

# Build with CUDA support
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_CUDA=ON

# Build with all security layers
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_SECURITY=ON

# Build opt-in reference backends (real computation; OFF by default)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_VULKAN=ON
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_TPU=ON
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_HTTP=ON
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_ZK=ON

# Build with tests
-DSNEPPX_BUILD_TESTS=ON

# Build with benchmarks
-DSNEPPX_BUILD_BENCHMARKS=ON
```

## Command Line Tools

### Core CLI

The project comes with several CLI tools pre-installed:

#### sneppx-train

```bash
# Training command
sneppx-train --help
```

#### sneppx-serve

```bash
# Inference server
sneppx-serve --help
```

#### sneppx-experiment

```bash
# Run experiments
sneppx-experiment --help
```

### Usage Examples

#### Train a Model

```bash
# Train on MNIST dataset
sneppx-train \
  --model resnet \
  --dataset mnist \
  --epochs 10 \
  --output-dir ./experiments/resnet-mnist
```

#### Serve a Model

```bash
# Start inference server with a trained model
sneppx-serve \
  --model-path ./experiments/resnet-mnist/best.pth \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

#### Run Experiment

```bash
# Run a hyperparameter experiment
sneppx-experiment \
  --config ./experiments/lr-tuning.yaml \
  --output ./experiments/lr-tuning \
  --resume \
  --continue
```

## Python Module Structure

### Core Modules (37 modules across 7 phases)

#### Phase 1: Tensor Engine & Autodiff

| Module | Purpose |
|--------|---------|
| `tensor.py` | Tensor creation, manipulation, operations |
| `autograd.py` | Automatic differentiation |
| `autograd_ops.py` | Differentiable operations |
| `advdat` | Advanced tensor operations |

#### Phase 2: Neural Network Building Blocks

| Module | Purpose |
|--------|---------|
| `nn.py` | Core neural network layers (Linear, Conv2d, etc.) |
| `attention.py` | Attention mechanisms |
| `activations.py` | Activation functions |

#### Phase 3: Optimization & Training

| Module | Purpose |
|--------|---------|
| `optim.py` | Optimizers (Adam, SGD, etc.) |
| `optim_extra.py` | Additional optimizers |
| `optim_advanced.py` | Advanced optimization |

#### Phase 4: Data Management

| Module | Purpose |
|--------|---------|
| `data.py` | Data loading utilities |
| `data_loader.py` | Dataset loaders |
| `tokenizer.py` | Tokenization for NLP |

#### Phase 5: Checkpoints & State Management

| Module | Purpose |
|--------|---------|
| `checkpoint.py` | Model saving/loading |
| `checkpoint_manager.py` | Checkpoint management |
| `experiment_tracker.py` | Experiment tracking |

#### Phase 6: Quantization & Compression

| Module | Purpose |
|--------|---------|
| `quantization.py` | Model quantization |
| `pruning.py` | Model pruning |
| `distillation.py` | Knowledge distillation |

#### Phase 7: Profiling & Deployment

| Module | Purpose |
|--------|---------|
| `profiler.py` | Performance profiling |
| `benchmark.py` | Benchmarking |
| `serve_cli.py` | CLI for serving models |
| `train_cli.py` | CLI for training |

### Advanced Interfaces

#### Low-level C Bindings

```python
from SneppX_ALG import _neural_engine_bridge as ax

# Access low-level C APIs
t = ax.SNEPPX_Tensor_create(2, 3)
ax.SNEPPX_Tensor_fill(t, 1.0f)
```

#### Security Interfaces

```python
from SneppX_ALG import (
    crypto_hash,
    crypto_encrypt,
    crypto_decrypt,
    generate_keypair,
    sign_data,
    verify_signature,
)

# Cryptographic operations
keypair = generate_keypair()
encrypted = crypto_encrypt(b"data", keypair.public_key)
decrypted = crypto_decrypt(encrypted, keypair.private_key)
```

#### Distributed Training

```python
from SneppX_ALG import (
    distributed,
    zero,
    nccl,
    checkpoint_distributed,
)

# Distributed training setup
distributed.init({"world_size": 4, "rank": 0})
zero_optimizer = zero.ZeroOptimizer(model.parameters())
```

## Architecture Overview

### Core Components

1. **Kernel Layer** (`kernel/`)
   - Tensor operations (SIMD, CUDA)
   - Autodiff engine
   - Optimizers
   - Training loop

2. **Algorithm Layer** (`algorithms/`)
   - HSS (Hierarchical State Spaces)
   - SER (Sparse Expert Routing)
   - ARC (Adversarial Robustness Certification)
   - NPE (Neural Program Engine)
   - FM (Factorized Manifolds)

3. **Security Layer** (`security/`)
   - S0-S9 security layers
   - Post-quantum crypto
   - Code obfuscation

4. **Network Layer** (`net/`)
   - Distributed training (NCCL)
   - Communication protocols
   - Checkpointing

### Supported Architectures

#### Large Language Models (LLMs)

- LLaMA 2/3 (7B, 13B, 70B)
- Mistral 7B
- Qwen2 (7B, 72B)
- DeepSeek V2 (Lite, Full)

#### Custom Architectures

- HSS/Mamba: Hierarchical state spaces
- SER/MoE: Sparse expert routing
- ARC: Adversarial robustness
- NPE: Neural program extraction
- FM: Fractal memory

### Hardware Support

#### Current
- x86-64 (AVX2, AVX-512)
- CUDA (NVIDIA GPUs)
- ROCm (AMD GPUs)
- Vulkan, TPU — opt-in reference backends (`SNEPPX_BUILD_VULKAN` / `SNEPPX_BUILD_TPU`)
- HTTP, ZK — opt-in reference backends (`SNEPPX_BUILD_HTTP` / `SNEPPX_BUILD_ZK`)
- Metal, oneAPI — reference backends (`SNEPPX_BUILD_METAL` / `SNEPPX_BUILD_ONEAPI`)

#### Planned
- ARMv8-A (NEON)
- NPU (Qualcomm, etc.)

## API Reference

### Python API

#### Core Types

| Class | Description |
|-------|-------------|
| `Tensor` | Multi-dimensional array |
| `Linear` | Fully connected layer |
| `Model` | Base neural network class |
| `Optimizer` | Base optimizer |
| `Trainer` | Training loop wrapper |

#### Key Functions

| Function | Purpose |
|----------|---------|
| `Tensor.create()` | Create a tensor |
| `Tensor.zeros()` | Create zero tensor |
| `Tensor.ones()` | Create one tensor |
| `Tensor.randn()` | Create random tensor |
| `optim.SGD()` | SGD optimizer |
| `optim.Adam()` | Adam optimizer |
| `optim.AdamW()` | AdamW optimizer |

### C API

Key C functions (from `include/neural_core/`):

- `sneppx_tensor_create()` - Create tensor
- `sneppx_tensor_free()` - Free tensor
- `sneppx_tensor_fill()` - Fill with values
- `sneppx_tensor_print()` - Print tensor
- `sneppx_tensor_add()` - Add tensors
- `sneppx_tensor_multiply()` - Multiply tensors

### Rust API

Rust bindings provide safe, idiomatic access:

```rust
use neural_core_algo::{Tensor, Linear, TrainingLoop};

fn main() {
    // Tensor operations
    let mut t = Tensor::zeros(vec![2, 3]);
    t.fill(1.0);
    
    // Neural network
    let model = Linear::new(1024, 512);
    
    // Training loop
    let mut trainer = TrainingLoop::new(model, 0.001);
    trainer.train(&t);
}
```

## Development Workflow

### Setting Up Development Environment

```bash
# Clone the repo
git clone https://github.com/ammar49-cyber/sneppx-alg.git
cd sneppx-alg

# Install Python dependencies
pip install -r requirements.txt
pip install -e bindings/python

# Install Rust toolchain (if needed)
rustup toolchain install stable
rustup component add rust-src

# Setup subprojects
rustup toolchain install nightly
```

### Making Changes

1. **Always create a feature branch**
2. **Follow the project's commit message conventions**
3. **Test locally before creating PRs**
4. **All tests must pass**

### Testing

#### Python Tests

```bash
# Run Python tests
$env:PYTHONPATH = "bindings/python"
pytest tests/python/ -v

# Quick test
pytest tests/python/test_tensor.py -v
```

#### C/C++ Tests

```bash
# Run C/C++ tests
cd build && ctest -C Release --output-on-failure

# Individual test
ctest -C Release -R test_tensor -v
```

#### Rust Tests

```bash
# Run Rust tests
cd net/distributed && cargo test
cd lib/rust && cargo test
```

#### Performance Benchmarks

```bash
# Run benchmarks
cd build && ./benchmarks/benchmark_suite --mode=full
```

## Examples

### Python Examples

See `examples/python/` for detailed examples:

1. **Simple Neural Network**
2. **Distributed Training**
3. **Quantization**
4. **Model Persistence**

### C/C++ Examples

See `examples/c/` for:

1. **Tensor Basics**
2. **Autodiff**
3. **Custom Operations**

### Rust Examples

See `examples/rust/` for:

1. **Tensor Operations**
2. **Neural Network Building**
3. **Integration Patterns**

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# Error: Could not import 'SnepX_ALG'
$ pip install SneppX_ALG==0.9.7.890e
```

#### Build Failures

```bash
# Error: C++11 not found
# Fix: Install g++ >= 7
apt-get update && apt-get install g++
```

#### CUDA Errors

```bash
# Error: No CUDA device
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_CUDA=OFF
```

### Getting Help

If you run into issues:

1. **Check the docs**: https://github.com/ammar49-cyber/sneppx-alg
2. **Search issues**: Look for similar problems
3. **Create a new issue**: Include:
   - Reproduction steps
   - Error messages
   - Environment info
   - Code snippets

## Future Roadmap

### Short-term (1-3 months)

- [ ] ARMv8-A support
- [ ] More LLM model support
- [ ] Improved quantization pipelines
- [ ] Enhanced error messages

### Medium-term (3-9 months)

- [x] TPU integration — opt-in reference backend (`SNEPPX_BUILD_TPU`); NPU planned
- [ ] Federated learning
- [ ] Formal verification for security
- [ ] Production-ready tooling

### Long-term (9-24 months)

- [ ] Edge deployment
- [ ] Quantum-resistant networking
- [ ] Trusted execution environments
- [ ] Full hardware acceleration

## References

### Academic Papers

- "Hierarchical State Spaces for Sequence Modeling" - https://arxiv.org/abs/xxxx.xxxxx
- "Mixture of Experts with Dynamic Routing" - https://arxiv.org/abs/yyyy.yyyyy
- "Post-Quantum Cryptography for ML" - https://arxiv.org/abs/zzzz.zzzzz

### Tools & Libraries

- **CMake** - Build system
- **pybind11** - Python bindings
- **tokio** - Asynchronous runtime
- **serde** - Serialization
- **tokio-retry** - Retry logic
- **anyhow** - Error handling
- **tracing** - Logging
- **clap** - CLI argument parsing

## License

MIT License

## Copyright

© 2024 Ammar [SNEPPX] - algoSNEPPX@gmail.com

---

_Generated with SNEPPX-Alg v0.9.7.890e_