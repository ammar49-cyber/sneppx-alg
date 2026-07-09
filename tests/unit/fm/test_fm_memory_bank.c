#include "fractal_memory_orchestrator.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_bank_create(void) {
    SNEPPXFMMemoryBank* bank = SNEPPX_fm_memory_bank_create(32, 64);
    ASSERT(bank != NULL, "bank not null");
    ASSERT(bank->keys != NULL, "keys not null");
    ASSERT(bank->values != NULL, "values not null");
    ASSERT(bank->keys->shape[0] == 64, "keys rows 64");
    ASSERT(bank->keys->shape[1] == 32, "keys cols 32");
    ASSERT(bank->num_entries == 0, "empty");
    ASSERT(bank->max_entries == 64, "max 64");
    SNEPPX_fm_memory_bank_destroy(bank);
}

static void test_bank_write_read(void) {
    SNEPPXFMMemoryBank* bank = SNEPPX_fm_memory_bank_create(8, 16);
    size_t key_shape[] = {8};
    size_t val_shape[] = {8};
    for (int k = 0; k < 5; k++) {
        SNEPPXTensor* key = SNEPPX_tensor_zeros(key_shape, 1, SNEPPX_FLOAT32);
        SNEPPXTensor* val = SNEPPX_tensor_zeros(val_shape, 1, SNEPPX_FLOAT32);
        ((float*)key->data)[0] = (float)k;
        ((float*)val->data)[0] = (float)(k * 10);
        int r = SNEPPX_fm_memory_bank_write(bank, key, val);
        ASSERT(r != 0, "write ok");
        SNEPPX_tensor_destroy(key);
        SNEPPX_tensor_destroy(val);
    }
    ASSERT(bank->num_entries == 5, "5 entries");

    SNEPPXTensor* q = SNEPPX_tensor_zeros(key_shape, 1, SNEPPX_FLOAT32);
    ((float*)q->data)[0] = 2.0f;
    SNEPPXTensor* result = SNEPPX_fm_memory_bank_read(bank, q);
    ASSERT(result != NULL, "read hit");
    ASSERT(fabsf(((float*)result->data)[0] - 20.0f) < 1e-5f, "value 20");
    SNEPPX_tensor_destroy(q);
    SNEPPX_tensor_destroy(result);
    SNEPPX_fm_memory_bank_destroy(bank);
}

static void test_bank_overwrite(void) {
    SNEPPXFMMemoryBank* bank = SNEPPX_fm_memory_bank_create(4, 8);
    size_t sh[] = {4};
    SNEPPXTensor* k = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* v1 = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    ((float*)v1->data)[0] = 100.0f;
    SNEPPX_fm_memory_bank_write(bank, k, v1);
    SNEPPXTensor* v2 = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    ((float*)v2->data)[0] = 200.0f;
    SNEPPX_fm_memory_bank_write(bank, k, v2);
    SNEPPXTensor* r = SNEPPX_fm_memory_bank_read(bank, k);
    ASSERT(r != NULL, "read hit");
    ASSERT(fabsf(((float*)r->data)[0] - 200.0f) < 1e-5f, "overwritten to 200");
    ASSERT(bank->num_entries == 1, "still 1 entry");
    SNEPPX_tensor_destroy(k);
    SNEPPX_tensor_destroy(v1);
    SNEPPX_tensor_destroy(v2);
    SNEPPX_tensor_destroy(r);
    SNEPPX_fm_memory_bank_destroy(bank);
}

static void test_bank_eviction(void) {
    SNEPPXFMMemoryBank* bank = SNEPPX_fm_memory_bank_create(2, 3);
    size_t sh[] = {2};
    for (int i = 0; i < 5; i++) {
        SNEPPXTensor* k = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
        SNEPPXTensor* v = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
        ((float*)k->data)[0] = (float)i;
        ((float*)v->data)[0] = (float)i;
        SNEPPX_fm_memory_bank_write(bank, k, v);
        SNEPPX_tensor_destroy(k);
        SNEPPX_tensor_destroy(v);
    }
    ASSERT(bank->num_entries == 3, "capped at 3");
    SNEPPX_fm_memory_bank_destroy(bank);
}

static void test_bank_forget(void) {
    SNEPPXFMMemoryBank* bank = SNEPPX_fm_memory_bank_create(4, 6);
    size_t sh[] = {4};
    for (int i = 0; i < 6; i++) {
        SNEPPXTensor* k = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
        SNEPPXTensor* v = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
        ((float*)k->data)[0] = (float)i;
        ((float*)v->data)[0] = (float)i;
        SNEPPX_fm_memory_bank_write(bank, k, v);
        SNEPPX_tensor_destroy(k);
        SNEPPX_tensor_destroy(v);
    }

    SNEPPXTensor* q = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    ((float*)q->data)[0] = 3.0f;
    for (int j = 0; j < 5; j++) {
        SNEPPXTensor* r = SNEPPX_fm_memory_bank_read(bank, q);
        SNEPPX_tensor_destroy(r);
    }
    SNEPPX_tensor_destroy(q);

    SNEPPX_fm_memory_bank_forget(bank, 0.5f);
    int retained = 0;
    for (size_t i = 0; i < bank->num_entries; i++) {
        float* vd = (float*)bank->values->data + i * 4;
        if (fabsf(vd[0]) > 1e-6f) retained++;
    }
    ASSERT(retained >= 2, "some retained after forget");
    SNEPPX_fm_memory_bank_destroy(bank);
}

int main(void) {
    run_test("test_bank_create", test_bank_create);
    run_test("test_bank_write_read", test_bank_write_read);
    run_test("test_bank_overwrite", test_bank_overwrite);
    run_test("test_bank_eviction", test_bank_eviction);
    run_test("test_bank_forget", test_bank_forget);
    printf("\nMemory bank tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
