#include "neural_programming_engine.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_vm_create(void) {
    ArixNPEConfig cfg = arix_npe_config_default();
    ArixNPEVM* vm = arix_npe_vm_create(&cfg);
    ASSERT(vm != NULL, "vm not null");
    ASSERT(vm->program == NULL, "no program loaded");
    arix_npe_vm_destroy(vm);
}

static void test_vm_nop(void) {
    ArixNPEConfig cfg = arix_npe_config_default();
    ArixNPEVM* vm = arix_npe_vm_create(&cfg);
    ArixNPEProgram* p = arix_npe_program_create(16);
    ArixNPEInstruction inst; memset(&inst, 0, sizeof(inst));
    inst.opcode = ARIX_HALT;
    arix_npe_program_append(p, inst);

    arix_npe_vm_load(vm, p);
    size_t sh[] = {2, 3};
    ArixTensor* input = arix_tensor_zeros(sh, 2, ARIX_FLOAT32);
    ArixTensor* output = NULL;
    int r = arix_npe_vm_run(vm, input, &output);
    ASSERT(r == 0, "run success");
    arix_tensor_destroy(input);
    if (output) arix_tensor_destroy(output);
    arix_npe_program_destroy(p);
    arix_npe_vm_destroy(vm);
}

static void test_vm_add(void) {
    ArixNPEConfig cfg = arix_npe_config_default();
    ArixNPEVM* vm = arix_npe_vm_create(&cfg);
    ArixNPEProgram* p = arix_npe_program_create(32);

    float a_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_vals[] = {10.0f, 20.0f, 30.0f, 40.0f};
    memcpy((float*)p->memory->data + 0, a_vals, 4 * sizeof(float));
    memcpy((float*)p->memory->data + 4, b_vals, 4 * sizeof(float));

    ArixNPEInstruction inst; memset(&inst, 0, sizeof(inst));
    inst.opcode = ARIX_LOAD; inst.dest_reg = 1; inst.immediate = 0;
    inst.shape_a[0] = 4;
    arix_npe_program_append(p, inst);

    inst.opcode = ARIX_LOAD; inst.dest_reg = 2; inst.immediate = 4;
    inst.shape_a[0] = 4;
    arix_npe_program_append(p, inst);

    inst.opcode = ARIX_ADD; inst.dest_reg = 3; inst.src_reg_a = 1; inst.src_reg_b = 2;
    arix_npe_program_append(p, inst);

    inst.opcode = ARIX_HALT;
    arix_npe_program_append(p, inst);

    arix_npe_vm_load(vm, p);
    size_t sh[] = {4};
    ArixTensor* input = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ArixTensor* output = NULL;
    int r = arix_npe_vm_run(vm, input, &output);
    ASSERT(r == 0, "run success");
    ASSERT(output != NULL, "output not null");
    float* od = (float*)output->data;
    ASSERT(fabsf(od[0] - 11.0f) < 1e-5f, "1+10=11");
    ASSERT(fabsf(od[1] - 22.0f) < 1e-5f, "2+20=22");
    ASSERT(fabsf(od[2] - 33.0f) < 1e-5f, "3+30=33");
    ASSERT(fabsf(od[3] - 44.0f) < 1e-5f, "4+40=44");

    arix_tensor_destroy(input);
    arix_tensor_destroy(output);
    arix_npe_program_destroy(p);
    arix_npe_vm_destroy(vm);
}

static void test_vm_matmul(void) {
    ArixNPEConfig cfg = arix_npe_config_default();
    ArixNPEVM* vm = arix_npe_vm_create(&cfg);
    ArixNPEProgram* p = arix_npe_program_create(32);

    float a_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_vals[] = {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    memcpy((float*)p->memory->data + 0, a_vals, 4 * sizeof(float));
    memcpy((float*)p->memory->data + 4, b_vals, 8 * sizeof(float));

    ArixNPEInstruction inst; memset(&inst, 0, sizeof(inst));
    inst.opcode = ARIX_LOAD; inst.dest_reg = 1; inst.immediate = 0;
    inst.shape_a[0] = 2; inst.shape_a[1] = 2;
    arix_npe_program_append(p, inst);

    inst.opcode = ARIX_LOAD; inst.dest_reg = 2; inst.immediate = 4;
    inst.shape_a[0] = 2; inst.shape_a[1] = 4;
    arix_npe_program_append(p, inst);

    inst.opcode = ARIX_MATMUL; inst.dest_reg = 3; inst.src_reg_a = 1; inst.src_reg_b = 2;
    arix_npe_program_append(p, inst);

    inst.opcode = ARIX_HALT;
    arix_npe_program_append(p, inst);

    arix_npe_vm_load(vm, p);
    size_t sh[] = {1};
    ArixTensor* input = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ArixTensor* output = NULL;
    int r = arix_npe_vm_run(vm, input, &output);
    ASSERT(r == 0, "run success");
    ASSERT(output != NULL, "output not null");
    ASSERT(output->shape[0] == 2 && output->shape[1] == 4, "output [2x4]");

    arix_tensor_destroy(input);
    arix_tensor_destroy(output);
    arix_npe_program_destroy(p);
    arix_npe_vm_destroy(vm);
}

int main(void) {
    run_test("test_vm_create", test_vm_create);
    run_test("test_vm_nop", test_vm_nop);
    run_test("test_vm_add", test_vm_add);
    run_test("test_vm_matmul", test_vm_matmul);
    printf("\nVM tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
