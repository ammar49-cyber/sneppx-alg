#ifndef ARIX_NPE_H
#define ARIX_NPE_H

#include "arix_tensor.h"
#include <stddef.h>

typedef enum {
    ARIX_NOP,
    ARIX_LOAD,
    ARIX_STORE,
    ARIX_ADD,
    ARIX_SUB,
    ARIX_MUL,
    ARIX_DIV,
    ARIX_MATMUL,
    ARIX_RELU,
    ARIX_SOFTMAX,
    ARIX_LAYERNORM,
    ARIX_ATTENTION,
    ARIX_BRANCH,
    ARIX_HALT,
    ARIX_EXP,
    ARIX_LOG,
    ARIX_SQRT,
    ARIX_POW,
    ARIX_SIN,
    ARIX_COS,
    ARIX_TANH,
    ARIX_SIGMOID,
    ARIX_GELU,
    ARIX_SILU,
    ARIX_DROPOUT,
    ARIX_CONV2D,
    ARIX_POOL2D,
    ARIX_BATCHNORM,
    ARIX_EMBEDDING,
    ARIX_CROSSENTROPY,
    ARIX_MSE,
    ARIX_CONCAT,
    ARIX_SPLIT
} ArixNPEOpCode;

typedef struct {
    int opcode;
    int dest_reg;
    int src_reg_a;
    int src_reg_b;
    int immediate;
    int shape_a[2];
    int shape_b[2];
} ArixNPEInstruction;

typedef struct {
    ArixNPEInstruction* instructions;
    size_t num_instructions;
    size_t max_instructions;
    ArixTensor* registers[16];
    ArixTensor* memory;
    size_t pc;
} ArixNPEProgram;

typedef struct {
    ArixNPEProgram* program;
    ArixNPEInstruction* execution_trace;
    size_t trace_length;
    size_t max_trace;
    size_t step_limit;
} ArixNPEVM;

typedef struct {
    size_t max_program_length;
    size_t register_count;
    size_t step_limit;
    int verification_mode;
    int trace_execution;
} ArixNPEConfig;

ArixNPEConfig arix_npe_config_default(void);
ArixNPEProgram* arix_npe_program_create(size_t max_instructions);
void arix_npe_program_destroy(ArixNPEProgram* prog);
void arix_npe_program_append(ArixNPEProgram* prog, ArixNPEInstruction inst);
ArixNPEVM* arix_npe_vm_create(const ArixNPEConfig* config);
void arix_npe_vm_destroy(ArixNPEVM* vm);
void arix_npe_vm_load(ArixNPEVM* vm, ArixNPEProgram* prog);
int arix_npe_vm_run(ArixNPEVM* vm, ArixTensor* input, ArixTensor** output);
int arix_npe_vm_step(ArixNPEVM* vm);
ArixNPEProgram* arix_npe_compile_attention(size_t seq_len, size_t dim);
ArixNPEProgram* arix_npe_compile_mlp(size_t dim, size_t hidden_dim);
int arix_npe_verify_program(const ArixNPEProgram* prog, char** error_msg, size_t* error_len);

#endif /* ARIX_NPE_H */
