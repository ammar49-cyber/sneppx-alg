# Roadmap

## v1.0.0 — Stable Release (2026-07-25)

### Completed
All 8 development phases complete, API frozen for stability, full regression suite passes:

| Area | Deliverable | Status |
|------|-------------|--------|
| Model Zoo | `from_pretrained()`, ModelHub, ModelConfig/Card/Registry, C/C++/Python integration | **Done** |
| Distributed Training | ZeRO-1/2/3, pipeline/tensor/expert parallel, elastic, fault tolerance | **Done** |
| Advanced Architectures | Differential Attention, Mamba-2, FlexAttention, MoD, gated activations, YaRN, ALiBi | **Done** |
| Security | S0-S9 complete (21,984+ LOC), 4-ring firewall, 123 test executables | **Done** |
| Quantization | INT8/FP8/AWQ/GPTQ, Python + CUDA kernels | **Done** |
| Checkpointing | Async save/load, heartbeat, elastic training, binary format reader | **Done** |
| Profiling | Profiler, structured logging, NVTX, sanitizer scripts | **Done** |
| CUDA Backend | Tensor-core GEMM, Flash Attention v2/v3, autodiff, memory pool, RNG | **Done** |

## v1.1 — Model Serving & Inference Optimization (Next)

| Area | Deliverable | Metric |
|------|-------------|--------|
| **Serving** | TensorRT-LLM / vLLM integration, OpenAI-compatible API | <5ms TTFT at 7B |
| **Continuing Pre-training** | Distributed checkpoint resume, FSDP integration | 10k tokens/s/node |
| **Fine-tuning** | LoRA/QLoRA adapters, DPO/GRPO trainers | 2x memory reduction |
| **Evaluation** | LM Eval Harness integration, standard benchmarks | MMLU/GSM8K/HumanEval |
| **Deployment** | pip-installable wheel, Docker Compose for serving | One-command deploy |

## v1.2 — Production Hardening

| Area | Deliverable | Metric |
|------|-------------|--------|
| **Performance** | CUDA graph capture, INT4/FP8 inference kernels | 2x inference speedup |
| **Memory** | PagedAttention v2, vLLM-style KV cache | 4x batch size increase |
| **Reliability** | Circuit breakers, retry logic, health checks for distributed training | 99.9% job success |
| **Monitoring** | Prometheus metrics, Grafana dashboards, tracing | Real-time visibility |

## v2.0 — Competitive with LLaMA-3

- 70B parameters, 1M LOC
- Multi-node training, 64+ GPUs
- Adaptive state size per HSS layer
- Hierarchical MoE with 256 experts
- Self-modifying NPE programs with meta-optimization
- Cross-datacenter federated learning
- S4 ZK proofs complete, S5 on-device runtime
- Formal verification of critical paths

## v3.0 — Competitive with GPT-4

- 1T parameters, 5M LOC
- Fully learned routing, 1000+ GPU cluster
- S6 federated contribution protocol
- On-device attestation

## v4.0 — Beyond Existing Systems

- 10T parameters, 15M LOC
- Self-evolving architecture
- S7 self-evolving NPE paths
- S8-S9 formal verification + pentest
- Full formal proof of alignment

## Quarterly Milestones (Updated)

| Quarter | Milestone | Status |
|---------|-----------|--------|
| **2026 Q3-Q4** | Phase 1-8 complete, v1.0.0 release | **Done** |
| **2027 Q1** | Model serving (vLLM/TensorRT-LLM), LoRA fine-tuning, LM Eval Harness | Pending |
| **2027 Q2** | CUDA graph capture, PagedAttention v2, Prometheus monitoring | Pending |
| **2027 Q3** | 7B model pre-training, distributed checkpoint resume | Pending |
| **2027 Q4** | Multi-GPU training, FSDP, 13B model | Pending |
| **2028** | 70B model, S6-S9 full integration, formal verification | Pending |
