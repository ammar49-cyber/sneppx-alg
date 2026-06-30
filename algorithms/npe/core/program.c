#include "neural_programming_engine.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>

ArixNPEConfig arix_npe_config_default(void) {
    ArixNPEConfig cfg;
    cfg.max_program_length = 256;
    cfg.register_count = 16;
    cfg.step_limit = 1024;
    cfg.verification_mode = 1;
    cfg.trace_execution = 1;
    return cfg;
}

ArixNPEProgram* arix_npe_program_create(size_t max_instructions) {
    ArixNPEProgram* prog = (ArixNPEProgram*)arix_malloc(sizeof(ArixNPEProgram), 64);
    if (!prog) return NULL;
    memset(prog, 0, sizeof(ArixNPEProgram));

    prog->instructions = (ArixNPEInstruction*)arix_malloc(max_instructions * sizeof(ArixNPEInstruction), 64);
    if (!prog->instructions) { arix_free(prog, sizeof(ArixNPEProgram)); return NULL; }
    memset(prog->instructions, 0, max_instructions * sizeof(ArixNPEInstruction));

    prog->max_instructions = max_instructions;
    prog->num_instructions = 0;
    prog->pc = 0;

    for (int i = 0; i < 16; i++) prog->registers[i] = NULL;

    size_t mem_size[] = {65536};
    prog->memory = arix_tensor_zeros(mem_size, 1, ARIX_FLOAT32);

    return prog;
}

void arix_npe_program_destroy(ArixNPEProgram* prog) {
    if (!prog) return;
    for (int i = 0; i < 16; i++) {
        if (prog->registers[i]) arix_tensor_destroy(prog->registers[i]);
    }
    if (prog->memory) arix_tensor_destroy(prog->memory);
    arix_free(prog->instructions, prog->max_instructions * sizeof(ArixNPEInstruction));
    arix_free(prog, sizeof(ArixNPEProgram));
}

void arix_npe_program_append(ArixNPEProgram* prog, ArixNPEInstruction inst) {
    if (prog->num_instructions >= prog->max_instructions) return;
    prog->instructions[prog->num_instructions++] = inst;
}
