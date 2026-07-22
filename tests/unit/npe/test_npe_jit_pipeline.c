#include "neural_programming_engine.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (%s)\n", msg, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

static void run_test(const char* name, void (*test_fn)(void)) {
    printf("Running %s... ", name);
    fflush(stdout);
    test_fn();
    printf("PASS\n");
    tests_passed++;
}

static void test_pipeline_identity(void) {
    SNEPPXNPEProgram* prog = SNEPPX_npe_program_create(64);
    ASSERT(prog != NULL, "create program");

    SNEPPXNPEInstruction load = {SNEPPX_LOAD, 0, -1, -1, 0, {1, 1}, {0, 0}};
    SNEPPX_npe_program_append(prog, load);
    SNEPPXNPEInstruction halt = {SNEPPX_HALT, -1, -1, -1, 0, {0, 0}, {0, 0}};
    SNEPPX_npe_program_append(prog, halt);

    float mem_data[] = {42.0f};
    SNEPPXTensor mem;
    mem.data = mem_data;
    mem.size = 1;

    SNEPPXNPEJITProfile* profile = SNEPPX_npe_jit_profile_create(100);
    ASSERT(profile != NULL, "create profile");

    SNEPPXNPEProgram* opt = SNEPPX_npe_jit_optimize(profile, prog, &mem);
    ASSERT(opt != NULL, "optimize returns program");
    ASSERT(opt->num_instructions > 0, "non-empty optimized program");

    SNEPPX_npe_program_destroy(opt);
    SNEPPX_npe_jit_profile_destroy(profile);
    SNEPPX_npe_program_destroy(prog);
}

static void test_pipeline_null_profile(void) {
    SNEPPXNPEProgram* prog = SNEPPX_npe_program_create(16);
    ASSERT(prog != NULL, "create program");

    SNEPPXNPEInstruction load = {SNEPPX_LOAD, 0, -1, -1, 0, {1, 1}, {0, 0}};
    SNEPPX_npe_program_append(prog, load);
    SNEPPXNPEInstruction halt = {SNEPPX_HALT, -1, -1, -1, 0, {0, 0}, {0, 0}};
    SNEPPX_npe_program_append(prog, halt);

    SNEPPXNPEProgram* opt = SNEPPX_npe_jit_optimize(NULL, prog, NULL);
    ASSERT(opt != NULL, "optimize with null profile works");

    SNEPPX_npe_program_destroy(opt);
    SNEPPX_npe_program_destroy(prog);
}

static void test_pipeline_null_prog(void) {
    SNEPPXNPEProgram* opt = SNEPPX_npe_jit_optimize(NULL, NULL, NULL);
    ASSERT(opt == NULL, "null prog returns null");
}

static void test_pipeline_fuses_matmul_add_relu(void) {
    SNEPPXNPEProgram* prog = SNEPPX_npe_program_create(64);
    ASSERT(prog != NULL, "create program");

    SNEPPXNPEInstruction load_in = {SNEPPX_LOAD, 0, -1, -1, 0, {2, 3}, {0, 0}};
    SNEPPX_npe_program_append(prog, load_in);
    SNEPPXNPEInstruction load_w = {SNEPPX_LOAD, 1, -1, -1, 6, {3, 2}, {0, 0}};
    SNEPPX_npe_program_append(prog, load_w);
    SNEPPXNPEInstruction load_b = {SNEPPX_LOAD, 2, -1, -1, 12, {2}, {0, 0}};
    SNEPPX_npe_program_append(prog, load_b);

    SNEPPXNPEInstruction matmul = {SNEPPX_MATMUL, 3, 0, 1, 0, {0, 0}, {0, 0}};
    SNEPPX_npe_program_append(prog, matmul);
    SNEPPXNPEInstruction add = {SNEPPX_ADD, 4, 3, 2, 0, {0, 0}, {0, 0}};
    SNEPPX_npe_program_append(prog, add);
    SNEPPXNPEInstruction relu = {SNEPPX_RELU, 5, 4, -1, 0, {0, 0}, {0, 0}};
    SNEPPX_npe_program_append(prog, relu);

    SNEPPXNPEInstruction store = {SNEPPX_STORE, -1, 5, -1, 14, {0, 0}, {0, 0}};
    SNEPPX_npe_program_append(prog, store);
    SNEPPXNPEInstruction halt = {SNEPPX_HALT, -1, -1, -1, 0, {0, 0}, {0, 0}};
    SNEPPX_npe_program_append(prog, halt);

    SNEPPXNPEProgram* opt = SNEPPX_npe_jit_optimize(NULL, prog, NULL);
    ASSERT(opt != NULL, "optimize returns program");

    int fused_count = 0;
    for (size_t i = 0; i < opt->num_instructions; i++) {
        if (opt->instructions[i].opcode == SNEPPX_MATMUL &&
            (opt->instructions[i].immediate & 0x20000000)) {
            fused_count++;
        }
    }
    ASSERT(fused_count == 1, "MATMUL+ADD+RELU triple fusion found");

    SNEPPX_npe_program_destroy(opt);
    SNEPPX_npe_program_destroy(prog);
}

static void test_vm_auto_jit(void) {
    SNEPPXNPEConfig cfg = SNEPPX_npe_config_default();
    cfg.jit_enabled = 1;
    cfg.jit_hot_threshold = 5;

    SNEPPXNPEVM* vm = SNEPPX_npe_vm_create(&cfg);
    ASSERT(vm != NULL, "vm created");
    ASSERT(vm->jit_profile != NULL, "jit profile created");

    SNEPPXNPEProgram* prog = SNEPPX_npe_compile_mlp(4, 8);
    ASSERT(prog != NULL, "mlp program created");

    float mem_data[128];
    for (int i = 0; i < 128; i++) mem_data[i] = (float)(i % 4);
    prog->memory = SNEPPX_tensor_create((size_t[]){128}, 1, SNEPPX_FLOAT32);
    memcpy(prog->memory->data, mem_data, 128 * sizeof(float));

    SNEPPX_npe_vm_load(vm, prog);

    SNEPPXTensor* input = SNEPPX_tensor_create((size_t[]){1, 4}, 2, SNEPPX_FLOAT32);
    float* id = (float*)input->data;
    id[0] = 1.0f; id[1] = 2.0f; id[2] = 3.0f; id[3] = 4.0f;

    SNEPPXTensor* output = NULL;
    int ret = SNEPPX_npe_vm_run(vm, input, &output);
    ASSERT(ret == 0, "vm run succeeded");
    ASSERT(output != NULL, "output produced");

    ASSERT(vm->jit_profile != NULL, "profile still exists");
    ASSERT(vm->jit_profile->is_profiling == 0, "profiling stopped after hot optimize");

    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(input);
    SNEPPX_npe_vm_destroy(vm);
}

static void test_vm_optimize_direct(void) {
    SNEPPXNPEConfig cfg = SNEPPX_npe_config_default();
    cfg.jit_enabled = 1;
    cfg.jit_hot_threshold = 1;

    SNEPPXNPEVM* vm = SNEPPX_npe_vm_create(&cfg);
    ASSERT(vm != NULL, "vm created");

    SNEPPXNPEProgram* prog = SNEPPX_npe_compile_mlp(4, 8);
    ASSERT(prog != NULL, "mlp program created");

    float mem_data[128];
    for (int i = 0; i < 128; i++) mem_data[i] = (float)(i % 4);
    prog->memory = SNEPPX_tensor_create((size_t[]){128}, 1, SNEPPX_FLOAT32);
    memcpy(prog->memory->data, mem_data, 128 * sizeof(float));

    SNEPPX_npe_vm_load(vm, prog);

    int ret = SNEPPX_npe_vm_optimize(vm);
    ASSERT(ret == 0, "direct optimize succeeded");
    ASSERT(vm->program != NULL, "program replaced after optimize");

    if (prog->memory) SNEPPX_tensor_destroy(prog->memory);
    SNEPPX_npe_vm_destroy(vm);
}

int main(void) {
    run_test("pipeline_identity", test_pipeline_identity);
    run_test("pipeline_null_profile", test_pipeline_null_profile);
    run_test("pipeline_null_prog", test_pipeline_null_prog);
    run_test("pipeline_fuses_matmul_add_relu", test_pipeline_fuses_matmul_add_relu);
    run_test("vm_auto_jit", test_vm_auto_jit);
    run_test("vm_optimize_direct", test_vm_optimize_direct);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
