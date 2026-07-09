#include "neural_programming_engine.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_verify_valid(void) {
    SNEPPXNPEProgram* p = SNEPPX_npe_program_create(16);
    SNEPPXNPEInstruction inst; memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_NOP;
    SNEPPX_npe_program_append(p, inst);
    inst.opcode = SNEPPX_HALT;
    SNEPPX_npe_program_append(p, inst);

    char* err = NULL;
    size_t err_len = 0;
    int r = SNEPPX_npe_verify_program(p, &err, &err_len);
    ASSERT(r != 0, "valid program passes");
    free(err);
    SNEPPX_npe_program_destroy(p);
}

static void test_verify_invalid_reg(void) {
    SNEPPXNPEProgram* p = SNEPPX_npe_program_create(16);
    SNEPPXNPEInstruction inst; memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_NOP; inst.dest_reg = 20;
    SNEPPX_npe_program_append(p, inst);
    inst.opcode = SNEPPX_HALT; inst.dest_reg = -1;
    SNEPPX_npe_program_append(p, inst);

    char* err = NULL;
    size_t err_len = 0;
    int r = SNEPPX_npe_verify_program(p, &err, &err_len);
    ASSERT(r == 0, "invalid reg fails");
    ASSERT(err_len > 0, "error message not empty");
    free(err);
    SNEPPX_npe_program_destroy(p);
}

int main(void) {
    run_test("test_verify_valid", test_verify_valid);
    run_test("test_verify_invalid_reg", test_verify_invalid_reg);
    printf("\nVerify tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
