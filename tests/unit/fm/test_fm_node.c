#include "arix_fm.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_node_create(void) {
    ArixFMNode* node = arix_fm_node_create(0, 16, 32);
    ASSERT(node != NULL, "node not null");
    ASSERT(node->node_id == 0, "id 0");
    ASSERT(node->memory_bank != NULL, "memory bank not null");
    ASSERT(node->gradient_accumulator != NULL, "grad accum not null");
    arix_fm_node_destroy(node);
}

static void test_node_online(void) {
    ArixFMNode* node = arix_fm_node_create(1, 8, 16);
    ASSERT(node->is_online == 1, "online true");
    ASSERT(fabsf(node->trust_score - 1.0f) < 1e-5f, "trust 1.0");
    arix_fm_node_destroy(node);
}

int main(void) {
    run_test("test_node_create", test_node_create);
    run_test("test_node_online", test_node_online);
    printf("\nNode tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
