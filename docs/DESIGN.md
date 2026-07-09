# Design

## Principles

1. **Security-first**: Every layer has defensive measures. No trust boundaries
   are crossed. Security is embedded into the architecture, not bolted on.

2. **O(n log n)**: All sequence modeling operations scale as O(n log n) using
   parallel associative scan. No quadratic attention.

3. **Verifiable**: All execution paths are deterministic and verifiable. The
   Neural Program Executor (NPE) provides bytecode-level verification.

4. **Federated**: Learning happens across distributed memory banks with
   differential privacy guarantees. No central data repository.

5. **Deterministic**: Given the same inputs and seed, all operations produce
   identical outputs. No non-determinism in forward passes.

## Component Interaction

The pipeline processes data through five components in sequence:

```
Input → HSS → SER → ARC → NPE → FM → Output
```

### Data Flow

1. **HSS (Hierarchical State Space)**
   - Input: Sequence of vectors (seq_len x input_dim)
   - Output: Sequence of hidden states (seq_len x state_dim)
   - Encodes temporal dependencies via parallel associative scan
   - Hierarchical layers capture multi-scale patterns

2. **SER (Sparse Expert Routing)**
   - Input: Hidden states from HSS
   - Output: Expert-weighted representations
   - Routes each token to top-k experts
   - Load balancing loss prevents expert collapse

3. **ARC (Adversarial Robustness Core)**
   - Input: SER output
   - Output: Sanitized representations
   - Z-score anomaly detection on inputs
   - Gradient obfuscation against adversarial attacks
   - Output consistency verification

4. **NPE (Neural Program Executor)**
   - Input: ARC output (or raw tensor for compiled programs)
   - Output: Verified computation result
   - Executes bytecode programs through a 16-register VM
   - Static verification before execution
   - Execution tracing for audit

5. **FM (Federated Memory)**
   - Input: NPE output
   - Output: Memory-augmented representations
   - Key-value memory banks with temporal decay
   - Distributed sync with differential privacy
   - Catastrophic forgetting protection

## Error Handling

- Fail fast: detect errors at the earliest point
- No silent errors: all error paths return error codes
- Log and abort: unrecoverable errors log diagnostic info before aborting
- Error codes: consistent integer return codes (0 = success, nonzero = error)
- Input validation: all public functions validate parameters

## Memory Management

- Pool allocator: `SNEPPX_malloc` / `SNEPPX_free` with alignment support
- Reference counting for shared tensor data
- No garbage collection in hot paths
- Memory pools for fixed-size allocations (tensor metadata, small buffers)
- Secure wipe for sensitive data (keys, gradients) on deallocation
- Guard pages around security-critical allocations
- Canary values for buffer overflow detection

## Thread Safety

- Immutable tensors: tensor data is read-only after creation
- Lock-free atomics for reference counts and shared state
- Mutex fallback for complex operations
- Thread-local storage for per-thread state (random seeds, temp buffers)
- No global mutable state

## Security Architecture

```
S0: Crypto primitives (Ed25519, ChaCha20-Poly1305, SHA-3, BLAKE3)
S1: Secure memory (guard pages, canaries, ASLR, locked memory)
S2: Obfuscation (control flow, string encryption, instruction substitution)
S3: Behavioral monitoring (anomaly detection, API hooking)
S4: Network security (planned v0.5.0)
S5: AI sanitizer (planned v0.5.0)
S6-S9: Advanced (planned v1.0+)
```

Every component includes defensive measures appropriate to its role:
- ARC: input sanitization and output verification
- NPE: bytecode verification before execution
- FM: differential privacy on gradient shares
- HSS/SER: constant-time operations where timing could leak information
