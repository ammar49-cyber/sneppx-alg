#ifndef SNEPPX_NPE_H
#define SNEPPX_NPE_H

#include "multidimensional_tensor_engine.h"
#include "automatic_differentiation_framework.h"
#include <stddef.h>

typedef enum {
    SNEPPX_NOP,
    SNEPPX_LOAD,
    SNEPPX_STORE,
    SNEPPX_ADD,
    SNEPPX_SUB,
    SNEPPX_MUL,
    SNEPPX_DIV,
    SNEPPX_MATMUL,
    SNEPPX_RELU,
    SNEPPX_SOFTMAX,
    SNEPPX_LAYERNORM,
    SNEPPX_ATTENTION,
    SNEPPX_BRANCH,
    SNEPPX_HALT,
    SNEPPX_NEG,
    SNEPPX_EXP,
    SNEPPX_LOG,
    SNEPPX_SQRT,
    SNEPPX_POW,
    SNEPPX_SIN,
    SNEPPX_COS,
    SNEPPX_TANH,
    SNEPPX_SIGMOID,
    SNEPPX_GELU,
    SNEPPX_SILU,
    SNEPPX_DROPOUT,
    SNEPPX_CONV2D,
    SNEPPX_POOL2D,
    SNEPPX_BATCHNORM,
    SNEPPX_EMBEDDING,
    SNEPPX_CROSSENTROPY,
    SNEPPX_MSE,
    SNEPPX_CONCAT,
    SNEPPX_SPLIT
} SNEPPXNPEOpCode;

typedef struct {
    int opcode;
    int dest_reg;
    int src_reg_a;
    int src_reg_b;
    int immediate;
    int shape_a[2];
    int shape_b[2];
} SNEPPXNPEInstruction;

typedef struct {
    SNEPPXNPEInstruction* instructions;
    size_t num_instructions;
    size_t max_instructions;
    SNEPPXTensor* registers[16];
    SNEPPXTensor* memory;
    size_t pc;
    SNEPPXTensor* param_w1;
    SNEPPXTensor* param_b1;
    SNEPPXTensor* param_w2;
    SNEPPXTensor* param_b2;
} SNEPPXNPEProgram;

typedef struct {
    SNEPPXNPEProgram* program;
    SNEPPXNPEInstruction* execution_trace;
    size_t trace_length;
    size_t max_trace;
    size_t step_limit;
} SNEPPXNPEVM;

typedef struct {
    size_t max_program_length;
    size_t register_count;
    size_t step_limit;
    int verification_mode;
    int trace_execution;
} SNEPPXNPEConfig;

SNEPPXNPEConfig SNEPPX_npe_config_default(void);
SNEPPXNPEProgram* SNEPPX_npe_program_create(size_t max_instructions);
void SNEPPX_npe_program_destroy(SNEPPXNPEProgram* prog);
void SNEPPX_npe_program_append(SNEPPXNPEProgram* prog, SNEPPXNPEInstruction inst);
SNEPPXNPEVM* SNEPPX_npe_vm_create(const SNEPPXNPEConfig* config);
void SNEPPX_npe_vm_destroy(SNEPPXNPEVM* vm);
void SNEPPX_npe_vm_load(SNEPPXNPEVM* vm, SNEPPXNPEProgram* prog);
int SNEPPX_npe_vm_run(SNEPPXNPEVM* vm, SNEPPXTensor* input, SNEPPXTensor** output);
int SNEPPX_npe_vm_step(SNEPPXNPEVM* vm);
SNEPPXNPEProgram* SNEPPX_npe_compile_attention(size_t seq_len, size_t dim);
SNEPPXNPEProgram* SNEPPX_npe_compile_mlp(size_t dim, size_t hidden_dim);
int SNEPPX_npe_verify_program(const SNEPPXNPEProgram* prog, char** error_msg, size_t* error_len);

// Training graph support
size_t SNEPPX_npe_get_params(SNEPPXNPEProgram* prog, SNEPPXTensor** out_params, size_t max_params);
int SNEPPX_npe_build_train_graph(SNEPPXNPEProgram* prog, SNEPPXTape* tape,
                                SNEPPXVariable* input_var,
                                SNEPPXVariable** weight_vars, size_t num_weights,
                                SNEPPXVariable** output_var);

// JIT profiling data
typedef struct SNEPPXNPEJITProfile {
    size_t op_frequency[32];
    size_t op_latency[32];
    size_t total_instructions;
    size_t hot_threshold;
    int   is_profiling;
} SNEPPXNPEJITProfile;

SNEPPXNPEJITProfile* SNEPPX_npe_jit_profile_create(size_t hot_threshold);
void SNEPPX_npe_jit_profile_destroy(SNEPPXNPEJITProfile* profile);
void SNEPPX_npe_jit_record(SNEPPXNPEJITProfile* profile, int opcode, float latency_us);
SNEPPXNPEProgram* SNEPPX_npe_jit_compile(SNEPPXNPEJITProfile* profile, const SNEPPXNPEProgram* original);
SNEPPXNPEProgram* SNEPPX_npe_jit_specialize(const SNEPPXNPEProgram* prog, size_t batch, size_t seq_len, size_t dim);
SNEPPXNPEProgram* SNEPPX_npe_jit_fuse(const SNEPPXNPEProgram* prog);
SNEPPXNPEProgram* SNEPPX_npe_jit_constant_fold(const SNEPPXNPEProgram* prog, const SNEPPXTensor* memory);
SNEPPXNPEProgram* SNEPPX_npe_jit_dce(const SNEPPXNPEProgram* prog);

#endif /* SNEPPX_NPE_H */
