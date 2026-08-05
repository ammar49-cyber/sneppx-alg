# FAQ

## General

**Q: What is SNEPPX-Algo?**  
A: A composable 5-component AI algorithm pipeline (HSS -> SER -> ARC -> NPE -> FM) with 10 embedded security layers (S0-S9). v1.0.0 delivers a stable release with all 8 phases complete, including a full model zoo with `from_pretrained()`, distributed training, quantization, and profiling.

**Q: Is this production-ready?**  
A: v1.0.0 is the first stable release. The core infrastructure (tensor engine, autodiff, optimizers, distributed training, model zoo, security layers) is fully implemented and tested. Serving integration (vLLM/TensorRT-LLM) and fine-tuning (LoRA/QLoRA) are planned for v1.1.

**Q: Can I use it as a PyTorch replacement?**  
A: Not the goal. SNEPPX-Algo explores a different design axis: security in every instruction plus a hybrid neuro-symbolic pipeline.

## Build

**Q: What compilers are supported?**  
A: MSVC 2022 (Windows), GCC 11+ (Linux), Clang 14+ (Linux/macOS). C11 required.

**Q: Do I need Python to build?**  
A: No. The C core has zero dependencies. Python wrappers are pure Python (stdlib only).

**Q: How do I enable CUDA?**  
A: `cmake -B build -G Ninja -DSNEPPX_BUILD_CUDA=ON`. Requires CUDA Toolkit 12.x. CUDA optimizer is an opt-in feature for SGD/AdamW acceleration.

## Architecture

**Q: How does HSS training work?**  
A: The HSS forward builds a computation graph via the autodiff tape. Backward computes real gradients for all 50+ tensor ops. The optimizer (SGD/AdamW) updates parameters in place. See `docs/hss_training.md`.

**Q: What is the JIT pipeline?**  
A: The NPE JIT pipeline applies DCE, constant folding, and fusion passes (MATMUL+ADD+RELU -> single fused op) to optimize VM programs. Auto-JIT mode triggers optimization when a program reaches a hot threshold.

**Q: Can I use individual components without the full pipeline?**  
A: Yes. Every component (HSS, SER, ARC, NPE, FM) has a standalone API and can be used independently.

## Security

**Q: Which security layers are implemented?**  
A: All ten phases (S0-S9) are complete: cryptographic core (Ed25519, ChaCha20-Poly1305, SHA-3, BLAKE3, Argon2id, Kyber, Dilithium, SPHINCS+), secure memory, obfuscation engine, behavioral monitoring, network security, AI sanitization, key vault, secure updates, formal verification, penetration testing.

**Q: Is the cryptography audited?**  
A: Not yet. S0 passes standard test vectors (NIST, RFC 8439, RFC 8032, FIPS 203/204/205) but has not undergone third-party audit.

## Python

**Q: How do I use the Python wrappers?**  
A: Set `$env:PYTHONPATH = "bindings/python"` and import from `SneppX_ALG`. No C compilation needed -- pure Python fallbacks are provided.

**Q: Do I need pybind11?**  
A: No. The Python wrappers are pure Python with optional C backend via ctypes. Pybind11 is only needed for `SNEPPX_BUILD_PYTHON=ON`.

## Roadmap

**Q: When will GPU support arrive?**  
A: CUDA backend is already implemented (opt-in, `SNEPPX_BUILD_CUDA=ON`). Covers GEMM, elementwise, layernorm, softmax, AdamW, memory pool, RNG, and the training optimizer.

**Q: When will the model be trainable?**  
A: CPU training works today -- `test_train_integration` builds a full HSS train graph, runs tape backward, and steps Adam deterministically (loss drops >90% over 10 steps).

**Q: Will you release pretrained weights?**  
A: A 7B parameter model is planned for v1.0 (2028).
