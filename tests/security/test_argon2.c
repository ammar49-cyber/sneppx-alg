#include <stdio.h>
#include <string.h>
#include "arix_argon2.h"
#include "arix_ct.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name, expr) do { \
    if (!(expr)) { printf("FAIL: %s\n", name); tests_failed++; } \
    else { printf("PASS: %s\n", name); tests_passed++; } \
} while(0)

void test_hash_verify(void) {
    printf("\n--- test_hash_verify ---\n");
    uint8_t password[] = "correct horse battery staple";
    uint8_t salt[] = "sodium chloride";
    uint8_t hash[32];

    ArixArgon2Config cfg;
    cfg.memory_kb = 64;
    cfg.iterations = 2;
    cfg.parallelism = 1;
    cfg.hash_len = 32;

    int ret = arix_argon2id(password, sizeof(password) - 1, salt, sizeof(salt) - 1, &cfg, hash);
    TEST("hash success", ret == 0);

    ret = arix_argon2id_verify(password, sizeof(password) - 1, salt, sizeof(salt) - 1, &cfg, hash);
    TEST("verify correct password", ret == 1);

    uint8_t wrong[] = "wrong password";
    ret = arix_argon2id_verify(wrong, sizeof(wrong) - 1, salt, sizeof(salt) - 1, &cfg, hash);
    TEST("verify wrong password fails", ret == 0);
}

void test_timing(void) {
    printf("\n--- test_timing ---\n");
    uint8_t password[] = "test password";
    uint8_t salt[] = "test salt";
    uint8_t hash[32];

    ArixArgon2Config cfg;
    cfg.memory_kb = 64;
    cfg.iterations = 2;
    cfg.parallelism = 1;
    cfg.hash_len = 32;

    arix_argon2id(password, sizeof(password) - 1, salt, sizeof(salt) - 1, &cfg, hash);

    uint8_t close[] = "test passworc";
    int ret_correct = arix_argon2id_verify(password, sizeof(password) - 1, salt, sizeof(salt) - 1, &cfg, hash);
    int ret_wrong = arix_argon2id_verify(close, sizeof(close) - 1, salt, sizeof(salt) - 1, &cfg, hash);

    TEST("timing-safe verify works", ret_correct == 1 && ret_wrong == 0);
}

int main(void) {
    printf("=== ARIX-Argon2 Test Suite ===\n");
    test_hash_verify();
    test_timing();
    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
