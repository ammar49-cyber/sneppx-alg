#include "test_common.h"
#include "data_pipeline.h"
#include "inference_engine.h"
#include "polymorphic_memory_allocator.h"
#include "multi_head_attention_module.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ===== Data Pipeline Tests ===== */

static void test_dataset_create(void) {
    ArixTokenizer* tok = arix_tokenizer_create(256);
    ASSERT_NOT_NULL(tok, "tokenizer created");
    arix_tokenizer_add_token(tok, "hello", 0);
    arix_tokenizer_add_token(tok, "world", 1);
    arix_tokenizer_add_token(tok, "test", 2);
    arix_tokenizer_add_token(tok, "data", 3);

    const char* texts[] = {"hello world", "test data", "hello test"};
    ArixTokenizer* bpe = arix_tokenizer_train_bpe(texts, 3, 270);
    ASSERT_NOT_NULL(bpe, "bpe trained");

    const char* tmpfile = "test_corpus.txt";
    FILE* f = fopen(tmpfile, "w");
    ASSERT_NOT_NULL(f, "file opened");
    fprintf(f, "hello world\ntest data\nhello test\n");
    fclose(f);

    ArixTextDataset* ds = arix_text_dataset_create(tmpfile, bpe, 8, 1);
    ASSERT_NOT_NULL(ds, "dataset created");
    ASSERT_EQ(arix_text_dataset_size(ds), 3, "3 samples");
    arix_text_dataset_destroy(ds);
    remove(tmpfile);
    arix_tokenizer_destroy(bpe);
    arix_tokenizer_destroy(tok);
}

static void test_dataset_batch(void) {
    ArixTokenizer* tok = arix_tokenizer_create(256);
    ASSERT_NOT_NULL(tok, "tokenizer created");
    const char* texts[] = {"a b c d e f g h i j k l m n o p q r s t u v w x y z"};
    ArixTokenizer* bpe = arix_tokenizer_train_bpe(texts, 1, 270);
    ASSERT_NOT_NULL(bpe, "bpe trained");

    const char* tmpfile = "test_batch.txt";
    FILE* f = fopen(tmpfile, "w");
    fprintf(f, "hello world this is a test of the data pipeline system\n");
    fclose(f);

    ArixTextDataset* ds = arix_text_dataset_create(tmpfile, bpe, 4, 0);
    ASSERT_NOT_NULL(ds, "dataset created");

    ArixTensor* in = NULL;
    ArixTensor* tgt = NULL;
    int ret = arix_text_dataset_get_batch(ds, 0, 2, &in, &tgt);
    ASSERT_EQ(ret, 0, "batch retrieved");
    ASSERT_NOT_NULL(in, "input tensor");
    ASSERT_NOT_NULL(tgt, "target tensor");
    ASSERT_EQ(in->ndim, 2, "input is 2D");
    ASSERT_EQ(in->shape[0], 2, "batch size 2");
    ASSERT_EQ(in->shape[1], 4, "seq len 4");

    arix_tensor_destroy(in);
    arix_tensor_destroy(tgt);
    arix_text_dataset_destroy(ds);
    remove(tmpfile);
    arix_tokenizer_destroy(bpe);
    arix_tokenizer_destroy(tok);
}

/* ===== Inference Engine Tests ===== */

static void test_argmax(void) {
    float logits[] = {-2.0f, 5.0f, 1.0f, 3.0f};
    int result = arix_argmax(logits, 4);
    ASSERT_EQ(result, 1, "argmax at index 1");
}

static void test_sample_logits(void) {
    float logits[] = {0.0f, 0.0f, 100.0f, 0.0f};
    ArixGenerationConfig cfg = arix_generation_config_default();
    cfg.temperature = 1.0f;
    cfg.top_k = 0;
    cfg.top_p = 0.0f;
    int result = arix_sample_from_logits(logits, 4, &cfg);
    ASSERT_EQ(result, 2, "sample picks highest prob");
}

static void test_gen_config_default(void) {
    ArixGenerationConfig cfg = arix_generation_config_default();
    ASSERT_EQ(cfg.max_new_tokens, 256, "max tokens default");
    ASSERT(cfg.temperature > 0, "temperature positive");
    ASSERT(cfg.top_p > 0, "top-p positive");
}

static void test_generate_tokens_small(void) {
    ArixAttentionConfig attn_cfg = arix_attn_config_default();
    attn_cfg.d_model = 16;
    attn_cfg.num_heads = 2;
    attn_cfg.head_dim = 8;
    attn_cfg.use_rope = 1;
    attn_cfg.use_causal_mask = 1;

    ArixAttentionWeights* attn = arix_attn_weights_create(attn_cfg, 42);
    ASSERT_NOT_NULL(attn, "attn weights created");

    size_t vocab = 16, d = 16;
    size_t emb_shape[] = {vocab, d};
    ArixTensor* embed = arix_tensor_empty(emb_shape, 2, ARIX_FLOAT32);
    ASSERT_NOT_NULL(embed, "embed created");
    for (size_t i = 0; i < vocab * d; i++) ((float*)embed->data)[i] = ((float)(i % 7) - 3.0f) * 0.1f;

    size_t unemb_shape[] = {d, vocab};
    ArixTensor* unembed = arix_tensor_create(unemb_shape, 2, ARIX_FLOAT32);
    ASSERT_NOT_NULL(unembed, "unembed created");
    for (size_t i = 0; i < d * vocab; i++) ((float*)unembed->data)[i] = ((float)(i * 3 % 11) - 5.0f) * 0.1f;

    int input[] = {0, 1, 2, 3};
    int output[32];
    ArixGenerationConfig cfg = arix_generation_config_default();
    cfg.max_new_tokens = 5;
    cfg.top_k = 1;

    int n = arix_generate_tokens(embed, unembed, attn, input, 4, output, 32, &cfg);
    ASSERT(n > 0, "tokens generated");
    ASSERT_EQ(output[0], 0, "first output token matches input");

    arix_tensor_destroy(embed);
    arix_tensor_destroy(unembed);
    arix_attn_weights_destroy(attn);
}

/* ===== Main ===== */

int main(void) {
    arix_mem_pool_init();
    run_test("dataset_create", test_dataset_create);
    run_test("dataset_batch", test_dataset_batch);
    run_test("argmax", test_argmax);
    run_test("sample_logits", test_sample_logits);
    run_test("gen_config_default", test_gen_config_default);
    run_test("generate_tokens_small", test_generate_tokens_small);
    RUN_ALL_TESTS();
}
