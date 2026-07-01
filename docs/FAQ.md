# FAQ

## General

**Q: What is ARIX-Algo?**  
A: A composable, cryptographically-secure AI algorithm pipeline built with security as a first principle.

**Q: Is this a production-ready framework?**  
A: No — v0.1.x is a research prototype. The architecture is real, the math is sound, but GPU support, distributed training, and formal verification are future work.

**Q: Can I use it as a PyTorch replacement?**  
A: That is not the goal. ARIX-Algo explores a different design axis: security in every instruction. Interop with PyTorch via ONNX or custom export may come in v1.0+.

## Build

**Q: What compilers are supported?**  
A: MSVC 2022 (Windows), GCC 11+ (Linux), Clang 14+ (Linux/macOS). C11 required, C++20 optional for S2 obfuscation.

**Q: Why does the build fail on MSVC with C4819?**  
A: Add `/utf-8` to your CMake flags, or ensure all source files are saved as UTF-8 with BOM.

**Q: Do I need Python to build?**  
A: No. The C core has zero dependencies. Python is only needed for the optional pybind11 bindings.

## Architecture

**Q: How does attention work without softmax?**  
A: Standard softmax attention is implemented. Flash attention and linear attention variants are planned for v1.0.

**Q: What is RoPE and why is it used?**  
A: Rotary Position Embedding encodes relative position through rotation matrices applied to query and key vectors. It enables better length generalization than absolute position encoding.

**Q: Can I use individual components without the full pipeline?**  
A: Yes. Every component (HSS, SER, ARC, NPE, FM, Attention) has a standalone API and can be used independently.

## Security

**Q: Which security layers are implemented?**  
A: S0 (cryptographic core — Ed25519, ChaCha20-Poly1305, SHA-3, BLAKE3, Argon2id), S1 (secure memory — guard pages, canaries, ASLR), S2 (obfuscation engine — control flow, string encryption, virtualization), S3 (behavioral monitor — structural).

**Q: Is the cryptography audited?**  
A: Not yet. S0 passes standard test vectors (NIST, RFC 8439, RFC 8032) but has not undergone third-party audit.

## Contributing

**Q: How do I submit a patch?**  
A: Email `git format-patch` output to algoarix@gmail.com. No pull requests. See CONTRIBUTING.md.

**Q: Can I become a maintainer?**  
A: Not at this time. The project uses a BDFL governance model.

## Roadmap

**Q: When will GPU support arrive?**  
A: CUDA kernels are planned for v1.0 (target 2027-2028).

**Q: When will the model be trainable?**  
A: v0.5.0 (target 2026 Q4) will have a working CPU training loop.

**Q: Will you release pretrained weights?**  
A: Yes — a 7B parameter model is planned for v1.0 (2028).
