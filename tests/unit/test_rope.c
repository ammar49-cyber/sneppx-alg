#include "multi_head_attention_module.h"
#include <stdio.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (%s)\n", msg, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) do { \
    if (fabsf((a) - (b)) > (eps)) { \
        printf("FAIL: %s (got %f, expected %f)\n", msg, (float)(a), (float)(b)); \
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

static void test_rope_precompute(void) {
    SNEPPXTensor* cos = SNEPPX_rope_precompute(8, 4, 10000.0f);
    ASSERT(cos != NULL, "precomputed cos table");
    ASSERT(cos->shape[0] == 8, "seq_len dim");
    ASSERT(cos->shape[1] == 4, "head_dim dim");
    SNEPPX_tensor_destroy(cos);
}

static void test_rope_apply_changes_values(void) {
    size_t shape[] = {2, 4};
    SNEPPXTensor* q = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* k = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    float* qd = (float*)q->data;
    float* kd = (float*)k->data;
    for (size_t i = 0; i < 8; i++) { qd[i] = 1.0f; kd[i] = 0.5f; }

    SNEPPXTensor* cos = SNEPPX_rope_precompute(2, 4, 10000.0f);
    SNEPPXTensor* sin = NULL;
    SNEPPX_rope_apply(q, k, cos, sin);

    float* q_out = (float*)q->data;
    int changed = 0;
    for (size_t i = 0; i < 8; i++) if (q_out[i] != 1.0f) { changed = 1; break; }
    ASSERT(changed, "rope rotates q values");

    SNEPPX_tensor_destroy(sin);
    SNEPPX_tensor_destroy(cos);
    SNEPPX_tensor_destroy(k);
    SNEPPX_tensor_destroy(q);
}

static void test_attention_self_attention(void) {
    SNEPPXAttentionConfig cfg;
    cfg.num_heads = 2;
    cfg.head_dim = 4;
    cfg.seq_len = 3;
    cfg.dropout = 0.0f;
    SNEPPXAttention* attn = SNEPPX_attention_create(&cfg, 42);
    ASSERT(attn != NULL, "attention layer created");

    size_t shape[] = {cfg.seq_len, cfg.num_heads * cfg.head_dim};
    SNEPPXTensor* x = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* mask = NULL;
    SNEPPXTensor* output = SNEPPX_attention_forward(attn, x, mask);
    ASSERT(output != NULL, "attention forward output");
    ASSERT(output->shape[0] == cfg.seq_len, "output seq_len");
    ASSERT(output->shape[1] == shape[1], "output feat dim");

    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(x);
    SNEPPX_attention_destroy(attn);
}

static void test_attention_causal_mask(void) {
    SNEPPXAttentionConfig cfg;
    cfg.num_heads = 1;
    cfg.head_dim = 2;
    cfg.seq_len = 3;
    cfg.dropout = 0.0f;
    SNEPPXAttention* attn = SNEPPX_attention_create(&cfg, 42);

    size_t shape[] = {cfg.seq_len, cfg.num_heads * cfg.head_dim};
    SNEPPXTensor* x = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* mask = SNEPPX_attention_causal_mask(cfg.seq_len);
    ASSERT(mask != NULL, "causal mask created");

    SNEPPXTensor* output = SNEPPX_attention_forward(attn, x, mask);
    ASSERT(output != NULL, "causal masked forward");

    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(mask);
    SNEPPX_tensor_destroy(x);
    SNEPPX_attention_destroy(attn);
}

int main(void) {
    run_test("rope_precompute", test_rope_precompute);
    run_test("rope_apply_changes_values", test_rope_apply_changes_values);
    run_test("attention_self_attention", test_attention_self_attention);
    run_test("attention_causal_mask", test_attention_causal_mask);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
