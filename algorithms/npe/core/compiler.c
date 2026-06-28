#include "arix_npe.h"
#include "arix_memory.h"
#include <string.h>
#include <stdlib.h>

ArixNPEProgram* arix_npe_compile_attention(size_t seq_len, size_t dim) {
    ArixNPEProgram* prog = arix_npe_program_create(32);
    if (!prog) return NULL;

    ArixNPEInstruction inst;
    memset(&inst, 0, sizeof(ArixNPEInstruction));

    inst.opcode = ARIX_LOAD; inst.dest_reg = 1; inst.immediate = 0;
    inst.shape_a[0] = (int)seq_len; inst.shape_a[1] = (int)dim;
    arix_npe_program_append(prog, inst);

    inst.opcode = ARIX_LOAD; inst.dest_reg = 2; inst.immediate = (int)(seq_len * dim);
    inst.shape_a[0] = (int)seq_len; inst.shape_a[1] = (int)dim;
    arix_npe_program_append(prog, inst);

    inst.opcode = ARIX_LOAD; inst.dest_reg = 3; inst.immediate = (int)(2 * seq_len * dim);
    inst.shape_a[0] = (int)seq_len; inst.shape_a[1] = (int)dim;
    arix_npe_program_append(prog, inst);

    inst.opcode = ARIX_MATMUL; inst.dest_reg = 4; inst.src_reg_a = 1; inst.src_reg_b = 2;
    arix_npe_program_append(prog, inst);

    inst.opcode = ARIX_SOFTMAX; inst.dest_reg = 5; inst.src_reg_a = 4;
    arix_npe_program_append(prog, inst);

    inst.opcode = ARIX_MATMUL; inst.dest_reg = 6; inst.src_reg_a = 5; inst.src_reg_b = 3;
    arix_npe_program_append(prog, inst);

    inst.opcode = ARIX_STORE; inst.src_reg_a = 6; inst.immediate = (int)(3 * seq_len * dim);
    arix_npe_program_append(prog, inst);

    inst.opcode = ARIX_HALT;
    arix_npe_program_append(prog, inst);

    return prog;
}

ArixNPEProgram* arix_npe_compile_mlp(size_t dim, size_t hidden_dim) {
    ArixNPEProgram* prog = arix_npe_program_create(32);
    if (!prog) return NULL;

    ArixNPEInstruction inst;
    memset(&inst, 0, sizeof(ArixNPEInstruction));

    inst.opcode = ARIX_LOAD; inst.dest_reg = 1; inst.immediate = 0;
    inst.shape_a[0] = (int)dim; inst.shape_a[1] = (int)hidden_dim;
    arix_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = ARIX_LOAD; inst.dest_reg = 2; inst.immediate = (int)(dim * hidden_dim);
    inst.shape_a[0] = (int)hidden_dim;
    arix_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = ARIX_MATMUL; inst.dest_reg = 3; inst.src_reg_a = 0; inst.src_reg_b = 1;
    arix_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = ARIX_ADD; inst.dest_reg = 4; inst.src_reg_a = 3; inst.src_reg_b = 2;
    arix_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = ARIX_RELU; inst.dest_reg = 5; inst.src_reg_a = 4;
    arix_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = ARIX_LOAD; inst.dest_reg = 6; inst.immediate = (int)(hidden_dim * dim + hidden_dim);
    inst.shape_a[0] = (int)hidden_dim; inst.shape_a[1] = (int)dim;
    arix_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = ARIX_LOAD; inst.dest_reg = 7; inst.immediate = (int)(hidden_dim * dim + hidden_dim + dim * hidden_dim);
    inst.shape_a[0] = (int)dim;
    arix_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = ARIX_MATMUL; inst.dest_reg = 8; inst.src_reg_a = 5; inst.src_reg_b = 6;
    arix_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = ARIX_ADD; inst.dest_reg = 9; inst.src_reg_a = 8; inst.src_reg_b = 7;
    arix_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = ARIX_STORE; inst.src_reg_a = 9;
    inst.immediate = (int)(hidden_dim * dim + hidden_dim + dim * hidden_dim + dim);
    arix_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = ARIX_HALT;
    arix_npe_program_append(prog, inst);

    return prog;
}
