#include "neural_programming_engine.h"
#include "polymorphic_memory_allocator.h"
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

ArixNPEJITProfile* arix_npe_jit_profile_create(size_t hot_threshold) {
    ArixNPEJITProfile* profile = (ArixNPEJITProfile*)calloc(1, sizeof(ArixNPEJITProfile));
    if (!profile) return NULL;
    profile->hot_threshold = hot_threshold ? hot_threshold : 100;
    profile->is_profiling = 1;
    return profile;
}

void arix_npe_jit_profile_destroy(ArixNPEJITProfile* profile) {
    if (profile) free(profile);
}

void arix_npe_jit_record(ArixNPEJITProfile* profile, int opcode, float latency_us) {
    if (!profile || !profile->is_profiling) return;
    if (opcode >= 0 && opcode < 32) {
        profile->op_frequency[opcode]++;
        profile->op_latency[opcode] += (size_t)(latency_us * 1000.0f);
        profile->total_instructions++;
    }
}

ArixNPEProgram* arix_npe_jit_compile(ArixNPEJITProfile* profile, const ArixNPEProgram* original) {
    if (!profile || !original) return NULL;

    int hot_op = -1;
    size_t max_freq = 0;
    for (int i = 0; i < 32; i++) {
        if (profile->op_frequency[i] > max_freq) {
            max_freq = profile->op_frequency[i];
            hot_op = i;
        }
    }

    if (max_freq < profile->hot_threshold) return NULL;

    ArixNPEProgram* opt = arix_npe_program_create(original->max_instructions);
    if (!opt) return NULL;

    for (size_t i = 0; i < original->num_instructions; i++) {
        ArixNPEInstruction inst = original->instructions[i];
        if (inst.opcode == ARIX_NOP) continue;

        if (inst.opcode == ARIX_MATMUL && i + 1 < original->num_instructions) {
            ArixNPEInstruction next = original->instructions[i + 1];
            if (next.opcode == ARIX_RELU && next.src_reg_a == inst.dest_reg) {
                inst.immediate |= 0x80000000;
                arix_npe_program_append(opt, inst);
                i++;
                continue;
            }
        }

        arix_npe_program_append(opt, inst);
    }

    return opt;
}

ArixNPEProgram* arix_npe_jit_specialize(const ArixNPEProgram* prog, size_t batch, size_t seq_len, size_t dim) {
    if (!prog) return NULL;
    ArixNPEProgram* spec = arix_npe_program_create(prog->max_instructions);
    if (!spec) return NULL;

    for (size_t i = 0; i < prog->num_instructions; i++) {
        ArixNPEInstruction inst = prog->instructions[i];

        if (inst.shape_a[0] > 0) inst.shape_a[0] = (int)seq_len;
        if (inst.shape_a[1] > 0) inst.shape_a[1] = (int)dim;
        if (inst.shape_b[0] > 0) inst.shape_b[0] = (int)seq_len;
        if (inst.shape_b[1] > 0) inst.shape_b[1] = (int)dim;

        if (inst.opcode == ARIX_BRANCH && inst.immediate < 0) {
            size_t body_start = i + inst.immediate;
            size_t body_end = i;
            for (size_t u = 0; u < batch; u++) {
                for (size_t j = body_start; j < body_end; j++) {
                    arix_npe_program_append(spec, prog->instructions[j]);
                }
            }
            continue;
        }

        arix_npe_program_append(spec, inst);
    }

    return spec;
}

ArixNPEProgram* arix_npe_jit_fuse(const ArixNPEProgram* prog) {
    if (!prog) return NULL;
    ArixNPEProgram* fused = arix_npe_program_create(prog->max_instructions);
    if (!fused) return NULL;

    for (size_t i = 0; i < prog->num_instructions; i++) {
        ArixNPEInstruction inst = prog->instructions[i];

        if (inst.opcode == ARIX_MATMUL && i + 1 < prog->num_instructions) {
            ArixNPEInstruction next = prog->instructions[i + 1];
            if (next.opcode == ARIX_RELU && next.src_reg_a == inst.dest_reg) {
                inst.immediate |= 0x80000000;
                arix_npe_program_append(fused, inst);
                i++;
                continue;
            }
        }

        if (inst.opcode == ARIX_ADD && i + 1 < prog->num_instructions) {
            ArixNPEInstruction next = prog->instructions[i + 1];
            if (next.opcode == ARIX_RELU && next.src_reg_a == inst.dest_reg) {
                inst.immediate |= 0x80000000;
                arix_npe_program_append(fused, inst);
                i++;
                continue;
            }
        }

        if (inst.opcode == ARIX_MATMUL && i + 1 < prog->num_instructions) {
            ArixNPEInstruction next = prog->instructions[i + 1];
            if (next.opcode == ARIX_ADD && next.src_reg_a == inst.dest_reg) {
                inst.immediate |= 0x40000000;
                arix_npe_program_append(fused, inst);
                i++;
                continue;
            }
        }

        arix_npe_program_append(fused, inst);
    }

    return fused;
}

ArixNPEProgram* arix_npe_jit_constant_fold(const ArixNPEProgram* prog, const ArixTensor* memory) {
    if (!prog) return NULL;
    (void)memory;
    ArixNPEProgram* folded = arix_npe_program_create(prog->max_instructions);
    if (!folded) return NULL;

    int reg_const[16];
    memset(reg_const, 0, sizeof(reg_const));

    for (size_t i = 0; i < prog->num_instructions; i++) {
        ArixNPEInstruction inst = prog->instructions[i];

        if (inst.opcode == ARIX_LOAD && inst.immediate == 0) {
            reg_const[inst.dest_reg] = 1;
        } else if (inst.opcode == ARIX_MUL &&
                   ((reg_const[inst.src_reg_a] == 1) || (reg_const[inst.src_reg_b] == 1))) {
            inst.opcode = ARIX_LOAD;
            inst.immediate = 0;
            inst.src_reg_a = 0;
            inst.src_reg_b = 0;
            memset(inst.shape_a, 0, sizeof(inst.shape_a));
            memset(inst.shape_b, 0, sizeof(inst.shape_b));
            reg_const[inst.dest_reg] = 1;
        } else if (inst.opcode == ARIX_ADD &&
                   (reg_const[inst.src_reg_a] == 1 && reg_const[inst.src_reg_b] == 1)) {
            inst.opcode = ARIX_LOAD;
            inst.immediate = 0;
            inst.src_reg_a = 0;
            inst.src_reg_b = 0;
            memset(inst.shape_a, 0, sizeof(inst.shape_a));
            memset(inst.shape_b, 0, sizeof(inst.shape_b));
            reg_const[inst.dest_reg] = 1;
        } else if (inst.opcode == ARIX_SUB &&
                   (reg_const[inst.src_reg_a] == 1 && reg_const[inst.src_reg_b] == 1)) {
            inst.opcode = ARIX_LOAD;
            inst.immediate = 0;
            inst.src_reg_a = 0;
            inst.src_reg_b = 0;
            memset(inst.shape_a, 0, sizeof(inst.shape_a));
            memset(inst.shape_b, 0, sizeof(inst.shape_b));
            reg_const[inst.dest_reg] = 1;
        } else if (inst.opcode == ARIX_RELU && reg_const[inst.src_reg_a] == 1) {
            inst.opcode = ARIX_LOAD;
            inst.immediate = 0;
            inst.src_reg_a = 0;
            inst.src_reg_b = 0;
            memset(inst.shape_a, 0, sizeof(inst.shape_a));
            memset(inst.shape_b, 0, sizeof(inst.shape_b));
            reg_const[inst.dest_reg] = 1;
        } else if (inst.opcode == ARIX_LOAD) {
            reg_const[inst.dest_reg] = 0;
        } else if (inst.opcode == ARIX_STORE || inst.opcode == ARIX_HALT || inst.opcode == ARIX_BRANCH) {
        } else if (inst.dest_reg >= 0 && inst.dest_reg < 16) {
            reg_const[inst.dest_reg] = 0;
        }

        arix_npe_program_append(folded, inst);
    }

    return folded;
}

ArixNPEProgram* arix_npe_jit_dce(const ArixNPEProgram* prog) {
    if (!prog) return NULL;

    int reg_used[16];
    memset(reg_used, 0, sizeof(reg_used));

    for (size_t i = 0; i < prog->num_instructions; i++) {
        ArixNPEInstruction inst = prog->instructions[i];
        if (inst.src_reg_a >= 0 && inst.src_reg_a < 16) reg_used[inst.src_reg_a] = 1;
        if (inst.src_reg_b >= 0 && inst.src_reg_b < 16) reg_used[inst.src_reg_b] = 1;
        if (inst.opcode == ARIX_STORE || inst.opcode == ARIX_HALT || inst.opcode == ARIX_BRANCH) {
            if (inst.dest_reg >= 0 && inst.dest_reg < 16) reg_used[inst.dest_reg] = 1;
        }
    }

    ArixNPEProgram* cleaned = arix_npe_program_create(prog->max_instructions);
    if (!cleaned) return NULL;

    for (size_t i = 0; i < prog->num_instructions; i++) {
        ArixNPEInstruction inst = prog->instructions[i];

        if (inst.opcode == ARIX_LOAD || inst.opcode == ARIX_STORE ||
            inst.opcode == ARIX_HALT || inst.opcode == ARIX_BRANCH) {
            arix_npe_program_append(cleaned, inst);
            continue;
        }

        if (inst.dest_reg >= 0 && inst.dest_reg < 16 && reg_used[inst.dest_reg]) {
            arix_npe_program_append(cleaned, inst);
        }
    }

    return cleaned;
}
