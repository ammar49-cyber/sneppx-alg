# Roadmap

## v0.1.1 — Current (2026-06-30)

- Multi-head attention with RoPE, causal mask, KV-cache, batched matmul 3D
- Inference engine: autoregressive generation, top-k/top-p/temperature, token streaming
- Data pipeline: TextDataset, BPE tokenization, sequence batching
- Model architecture: modular pipeline with per-module enable/disable flags
- Infrastructure: linting configs, style guide, maintainers docs

## v0.5.0 — Trainable on CPU (6 months)

| Area | Deliverable | Metric |
|------|-------------|--------|
| Autodiff | Real C gradients for all 50+ tensor ops | Backward matches numerical gradient |
| Optimizer | SGD + AdamW with weight decay & LR scheduling | Converges on toy problems |
| HSS | Parallel scan over state dimension | 2x speedup vs sequential |
| SER | Learned gating (tiny MLP per expert) | 5% perplexity improvement |
| ARC | Real attack simulation during training | Robust to epsilon=0.1 FGSM |
| NPE | JIT compilation of attention/MLP programs | 10x VM speedup |
| FM | Single-node on-device sync (simulated multi-node) | Correct gradient aggregation |
| Trainer | CPU training loop with checkpointing | 10k steps WikiText-2 in 24h |
| Python API | Tensor, Variable, Tape, Model, Trainer classes | API complete for demo |
| Testing | 200+ tests, all pass | 100% coverage of exposed API |

## v1.0.0 — Competitive with GPT-2 (18 months)

| Area | Deliverable | Metric |
|------|-------------|--------|
| GPU | CUDA kernels for all tensor ops, HSS, SER, ARC, NPE, FM | 50x speedup vs CPU |
| Distributed | Multi-GPU training with NCCL | Linear scaling up to 8 GPUs |
| HSS | Hierarchical scan with multi-resolution states | 5x longer context than GPT-2 |
| SER | Top-2 out of 64 experts, load-balanced | 2x parameter efficiency |
| ARC | Provable robustness guarantees | Certified robustness at epsilon=0.1 |
| NPE | On-device execution with CUDA backend | 1000 programs/s throughput |
| FM | Federated training across 4 nodes | 95% of centralized performance |
| Python | Full API, pip install, documentation | 90% API coverage documented |
| Security | S3 behavioral monitor complete, S4 ZK proofs start | Real-time anomaly detection |
| **Size** | **~200,000 LOC** | **7B parameters** |

## v2.0.0 — Competitive with LLaMA-3 (4 years)

- 70B parameters, 1M LOC
- Multi-node training, 64+ GPUs
- Adaptive state size per HSS layer
- Hierarchical MoE with 256 experts
- Self-modifying NPE programs with meta-optimization
- Cross-datacenter federated learning
- S4 ZK proofs complete, S5 on-device runtime
- Formal verification of critical paths

## v3.0.0 — Competitive with GPT-4 (7 years)

- 1T parameters, 5M LOC
- Fully learned routing, 1000+ GPU cluster
- S6 federated contribution protocol
- On-device attestation

## v4.0.0 — Beyond Existing Systems (10 years)

- 10T parameters, 15M LOC
- Self-evolving architecture
- S7 self-evolving NPE paths
- S8-S9 formal verification + pentest
- Full formal proof of alignment

## Quarterly Milestones

| Quarter | Milestone |
|---------|-----------|
| **2026 Q3** | Autodiff backward pass functional, SGD + AdamW, 100+ tests |
| **2026 Q4** | HSS parallel scan, SER learned gating, CPU training loop, Python API v0.5 |
| **2027 Q1** | WikiText-2 training end-to-end, S3 complete |
| **2027 Q2** | S4 ZK proofs: proof generation for MLP, 200+ tests |
| **2027 Q3** | CUDA kernels for tensor ops, Benchmarks |
| **2027 Q4** | Multi-GPU training, S5 on-device runtime (ARM NEON) |
| **2028** | 7B model, S6, Distributed training, S7 |
