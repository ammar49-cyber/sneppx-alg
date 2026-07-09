#include "neural_programming_engine.h"
#include <stdio.h>
#include <string.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_program_create(void) {
    SNEPPXNPEProgram* p = SNEPPX_npe_program_create(64);
    ASSERT(p != NULL, "program not null");
    ASSERT(p->num_instructions == 0, "empty program");
    ASSERT(p->max_instructions == 64, "max 64");
    ASSERT(p->registers[0] == NULL, "reg 0 null");
    ASSERT(p->registers[15] == NULL, "reg 15 null");
    ASSERT(p->memory != NULL, "memory not null");
    SNEPPX_npe_program_destroy(p);
}

static void test_program_append(void) {
    SNEPPXNPEProgram* p = SNEPPX_npe_program_create(64);
    ASSERT(p != NULL, "program not null");
    SNEPPXNPEInstruction inst; memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_NOP;
    SNEPPX_npe_program_append(p, inst);
    SNEPPX_npe_program_append(p, inst);
    SNEPPX_npe_program_append(p, inst);
    SNEPPX_npe_program_append(p, inst);
    SNEPPX_npe_program_append(p, inst);
    ASSERT(p->num_instructions == 5, "5 instructions");
    ASSERT(p->instructions[0].opcode == SNEPPX_NOP, "opcode NOP");
    SNEPPX_npe_program_destroy(p);
}

int main(void) {
    run_test("test_program_create", test_program_create);
    run_test("test_program_append", test_program_append);
    printf("\nProgram tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
