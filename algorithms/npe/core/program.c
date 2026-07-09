#include "neural_programming_engine.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>

SNEPPXNPEConfig SNEPPX_npe_config_default(void) {
    SNEPPXNPEConfig cfg;
    cfg.max_program_length = 256;
    cfg.register_count = 16;
    cfg.step_limit = 1024;
    cfg.verification_mode = 1;
    cfg.trace_execution = 1;
    return cfg;
}

SNEPPXNPEProgram* SNEPPX_npe_program_create(size_t max_instructions) {
    SNEPPXNPEProgram* prog = (SNEPPXNPEProgram*)SNEPPX_malloc(sizeof(SNEPPXNPEProgram), 64);
    if (!prog) return NULL;
    memset(prog, 0, sizeof(SNEPPXNPEProgram));

    prog->instructions = (SNEPPXNPEInstruction*)SNEPPX_malloc(max_instructions * sizeof(SNEPPXNPEInstruction), 64);
    if (!prog->instructions) { SNEPPX_free(prog, sizeof(SNEPPXNPEProgram)); return NULL; }
    memset(prog->instructions, 0, max_instructions * sizeof(SNEPPXNPEInstruction));

    prog->max_instructions = max_instructions;
    prog->num_instructions = 0;
    prog->pc = 0;

    for (int i = 0; i < 16; i++) prog->registers[i] = NULL;

    size_t mem_size[] = {65536};
    prog->memory = SNEPPX_tensor_zeros(mem_size, 1, SNEPPX_FLOAT32);

    return prog;
}

void SNEPPX_npe_program_destroy(SNEPPXNPEProgram* prog) {
    if (!prog) return;
    for (int i = 0; i < 16; i++) {
        if (prog->registers[i]) SNEPPX_tensor_destroy(prog->registers[i]);
    }
    if (prog->memory) SNEPPX_tensor_destroy(prog->memory);
    if (prog->param_w1) SNEPPX_tensor_destroy(prog->param_w1);
    if (prog->param_b1) SNEPPX_tensor_destroy(prog->param_b1);
    if (prog->param_w2) SNEPPX_tensor_destroy(prog->param_w2);
    if (prog->param_b2) SNEPPX_tensor_destroy(prog->param_b2);
    SNEPPX_free(prog->instructions, prog->max_instructions * sizeof(SNEPPXNPEInstruction));
    SNEPPX_free(prog, sizeof(SNEPPXNPEProgram));
}

void SNEPPX_npe_program_append(SNEPPXNPEProgram* prog, SNEPPXNPEInstruction inst) {
    if (prog->num_instructions >= prog->max_instructions) return;
    prog->instructions[prog->num_instructions++] = inst;
}
