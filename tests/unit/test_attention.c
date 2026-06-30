#include "multi_head_attention_module.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0, tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) do { \
    if (fabsf((a) - (b)) > (eps)) { \
        printf("FAIL: %s (got %f, expected %f)\n", msg, (float)(a), (float)(b)); \
        tests_failed++; return; \
    } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        printf("FAIL: %s (got %d, expected %d)\n", msg, (int)(a), (int)(b)); \
        tests_failed++; return; \
    } \
} while(0)

static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout);
    fn(); printf("PASS\n"); tests_passed++;
}

static void test_config_default(void) {
    ArixAttentionConfig cfg = arix_attn_config_default();
    ASSERT_EQ(cfg.num_heads, 8, "8 heads");
    ASSERT_EQ(cfg.head_dim, 64, "64 head dim");
    ASSERT_EQ(cfg.d_model, 512, "512 model dim");
}

static void test_weights_create(void) {
    ArixAttentionConfig cfg = arix_attn_config_default();
    ArixAttentionWeights* w = arix_attn_weights_create(cfg, 42);
    ASSERT(w != NULL, "weights non-NULL");
    ASSERT(w->w_q != NULL, "w_q non-NULL");
    ASSERT(w->w_k != NULL, "w_k non-NULL");
    ASSERT(w->w_v != NULL, "w_v non-NULL");
    ASSERT(w->w_o != NULL, "w_o non-NULL");
    size_t expected = 4 * (cfg.d_model * cfg.num_heads * cfg.head_dim) + 4 * (cfg.num_heads * cfg.head_dim);
    ASSERT_EQ(arix_attn_num_params(w), expected, "param count matches");
    arix_attn_weights_destroy(w);
}

static void test_rope_precompute(void) {
    ArixTensor* cos = arix_rope_precompute(10, 64, 10000.0f);
    ASSERT(cos != NULL, "cos non-NULL");
    ASSERT_EQ(cos->shape[0], 10, "10 seq positions");
    ASSERT_EQ(cos->shape[1], 64, "64 head dim");
    float* d = (float*)cos->data;
    ASSERT_NEAR(d[0], 1.0f, 1e-5f, "cos(0)=1");
    arix_tensor_destroy(cos);
}

static void test_forward(void) {
    ArixAttentionConfig cfg = arix_attn_config_default();
    cfg.d_model = 64; cfg.num_heads = 2; cfg.head_dim = 32;
    ArixAttentionWeights* w = arix_attn_weights_create(cfg, 42);
    size_t shape[] = {1, 4, cfg.d_model};
    ArixTensor* x = arix_tensor_ones(shape, 3, ARIX_FLOAT32);
    ArixTensor* cos_t = arix_rope_precompute(4, cfg.head_dim, 10000.0f);
    ArixTensor* result = arix_attn_forward(w, x, cos_t, cos_t);
    ASSERT(result != NULL, "forward non-NULL");
    ASSERT_EQ(result->ndim, 3, "output is 3D");
    ASSERT_EQ(result->shape[0], 1, "batch=1");
    ASSERT_EQ(result->shape[1], 4, "seq=4");
    ASSERT_EQ(result->shape[2], cfg.d_model, "d_model matches");
    arix_tensor_destroy(result);
    arix_tensor_destroy(cos_t);
    arix_tensor_destroy(x);
    arix_attn_weights_destroy(w);
}

static void test_kv_cache(void) {
    ArixKVCache* c = arix_kv_cache_create(2, 64, 8, 64);
    ASSERT(c != NULL, "cache non-NULL");
    ASSERT_EQ(c->seq_len, 0, "starts empty");
    arix_kv_cache_clear(c);
    arix_kv_cache_destroy(c);
}

static void test_batched_matmul(void) {
    size_t sh[] = {1, 1, 3, 4};
    ArixTensor* a = arix_tensor_ones(sh, 4, ARIX_FLOAT32);
    ArixTensor* b = arix_tensor_ones(sh, 4, ARIX_FLOAT32);
    ArixTensor* out = arix_batched_matmul(a, b, 1, 0);
    ASSERT(out != NULL, "batched matmul non-NULL");
    ASSERT_EQ(out->shape[0], 1, "batch=1");
    ASSERT_EQ(out->shape[1], 1, "heads=1");
    ASSERT_EQ(out->shape[2], 3, "seq=3");
    ASSERT_EQ(out->shape[3], 3, "seq=3");
    float* d = (float*)out->data;
    ASSERT_NEAR(d[0], 4.0f, 1e-5f, "dot product of 4 ones = 4");
    arix_tensor_destroy(a); arix_tensor_destroy(b); arix_tensor_destroy(out);
}

int main(void) {
    arix_mem_pool_init();
    run_test("config_default", test_config_default);
    run_test("weights_create", test_weights_create);
    run_test("rope_precompute", test_rope_precompute);
    run_test("forward", test_forward);
    run_test("kv_cache", test_kv_cache);
    run_test("batched_matmul", test_batched_matmul);
    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    arix_mem_pool_destroy();
    return tests_failed > 0 ? 1 : 0;
}
