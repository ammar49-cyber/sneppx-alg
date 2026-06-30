#include "subword_tokenization_pipeline.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0, tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { printf("FAIL: %s (got %d, expected %d)\n", msg, (int)(a), (int)(b)); tests_failed++; return; } \
} while(0)

#define ASSERT_STR_EQ(a, b, msg) do { \
    if (strcmp((a), (b)) != 0) { printf("FAIL: %s (got \"%s\", expected \"%s\")\n", msg, (a), (b)); tests_failed++; return; } \
} while(0)

static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout);
    fn(); printf("PASS\n"); tests_passed++;
}

static void test_create_destroy(void) {
    ArixTokenizer* tok = arix_tokenizer_create(1000);
    ASSERT(tok != NULL, "create returned non-NULL");
    ASSERT_EQ(arix_tokenizer_vocab_size(tok), 256, "starts with 256 bytes");
    arix_tokenizer_destroy(tok);
}

static void test_add_token(void) {
    ArixTokenizer* tok = arix_tokenizer_create(500);
    int r = arix_tokenizer_add_token(tok, "hello", 256);
    ASSERT_EQ(r, 0, "add hello token");
    ASSERT_EQ(arix_tokenizer_vocab_size(tok), 257, "vocab now 257");
    arix_tokenizer_destroy(tok);
}

static void test_encode_decode_basic(void) {
    ArixTokenizer* tok = arix_tokenizer_create(1000);
    const char* text = "hello";
    int ids[256];
    int n = arix_tokenizer_encode(tok, text, ids, 256);
    ASSERT(n > 0, "encoded non-empty");
    char* decoded = arix_tokenizer_decode(tok, ids, (size_t)n);
    ASSERT(decoded != NULL, "decoded non-NULL");
    ASSERT_STR_EQ(decoded, text, "roundtrip match");
    arix_free(decoded, strlen(decoded) + 1);
    arix_tokenizer_destroy(tok);
}

static void test_train_bpe(void) {
    const char* texts[] = {"low low low low low", "lowest lowest", "newer newer"};
    ArixTokenizer* tok = arix_tokenizer_train_bpe(texts, 3, 270);
    ASSERT(tok != NULL, "trained non-NULL");
    ASSERT(arix_tokenizer_vocab_size(tok) > 256, "vocab grew");
    int ids[256];
    int n = arix_tokenizer_encode(tok, "low", ids, 256);
    ASSERT(n < 3, "low encoded with fewer tokens than letters (bpe works)");
    char* decoded = arix_tokenizer_decode(tok, ids, (size_t)n);
    ASSERT(decoded != NULL, "decoded non-NULL");
    ASSERT_STR_EQ(decoded, "low", "roundtrip low");
    arix_free(decoded, strlen(decoded) + 1);
    arix_tokenizer_destroy(tok);
}

static void test_save_load(void) {
    ArixTokenizer* tok1 = arix_tokenizer_create(500);
    arix_tokenizer_add_token(tok1, "test", 256);
    arix_tokenizer_add_token(tok1, "save", 257);
    int r = arix_tokenizer_save(tok1, "test_tokenizer.bin");
    ASSERT_EQ(r, 0, "save succeeded");
    ArixTokenizer* tok2 = arix_tokenizer_load("test_tokenizer.bin");
    ASSERT(tok2 != NULL, "load succeeded");
    ASSERT_EQ(arix_tokenizer_vocab_size(tok2), 258, "vocab size matches");
    const char* text = "test";
    int ids1[256], ids2[256];
    int n1 = arix_tokenizer_encode(tok1, text, ids1, 256);
    int n2 = arix_tokenizer_encode(tok2, text, ids2, 256);
    ASSERT_EQ(n1, n2, "encode lengths match");
    arix_tokenizer_destroy(tok1);
    arix_tokenizer_destroy(tok2);
    remove("test_tokenizer.bin");
}

static void test_special_tokens(void) {
    ArixTokenizer* tok = arix_tokenizer_create(500);
    ArixSpecialTokens sp = {0, 1, 2, 3};
    arix_tokenizer_set_special(tok, sp);
    ArixSpecialTokens got = arix_tokenizer_special(tok);
    ASSERT_EQ(got.pad_id, 0, "pad=0");
    ASSERT_EQ(got.bos_id, 1, "bos=1");
    ASSERT_EQ(got.eos_id, 2, "eos=2");
    ASSERT_EQ(got.unk_id, 3, "unk=3");
    arix_tokenizer_destroy(tok);
}

int main(void) {
    arix_mem_pool_init();
    run_test("create_destroy", test_create_destroy);
    run_test("add_token", test_add_token);
    run_test("encode_decode_basic", test_encode_decode_basic);
    run_test("train_bpe", test_train_bpe);
    run_test("save_load", test_save_load);
    run_test("special_tokens", test_special_tokens);
    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    arix_mem_pool_destroy();
    return tests_failed > 0 ? 1 : 0;
}
