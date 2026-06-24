#include "arix_npe.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_compile_attention(void) {
    ArixNPEProgram* p = arix_npe_compile_attention(8, 16);
    ASSERT(p != NULL, "program not null");
    ASSERT(p->num_instructions > 5, "more than 5 instructions");
    int has_softmax = 0, has_matmul = 0;
    for (size_t i = 0; i < p->num_instructions; i++) {
        if (p->instructions[i].opcode == ARIX_SOFTMAX) has_softmax = 1;
        if (p->instructions[i].opcode == ARIX_MATMUL) has_matmul = 1;
    }
    ASSERT(has_softmax, "contains SOFTMAX");
    ASSERT(has_matmul, "contains MATMUL");
    arix_npe_program_destroy(p);
}

static void test_compile_mlp(void) {
    ArixNPEProgram* p = arix_npe_compile_mlp(8, 16);
    ASSERT(p != NULL, "program not null");
    ASSERT(p->num_instructions > 5, "more than 5 instructions");
    int has_relu = 0, has_matmul = 0, has_add = 0;
    for (size_t i = 0; i < p->num_instructions; i++) {
        if (p->instructions[i].opcode == ARIX_RELU) has_relu = 1;
        if (p->instructions[i].opcode == ARIX_MATMUL) has_matmul = 1;
        if (p->instructions[i].opcode == ARIX_ADD) has_add = 1;
    }
    ASSERT(has_relu, "contains RELU");
    ASSERT(has_matmul, "contains MATMUL");
    ASSERT(has_add, "contains ADD");
    arix_npe_program_destroy(p);
}

int main(void) {
    run_test("test_compile_attention", test_compile_attention);
    run_test("test_compile_mlp", test_compile_mlp);
    printf("\nCompiler tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
