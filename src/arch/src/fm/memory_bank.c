#include "arix_fm.h"
#include "arix_memory.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

static float euclidean_similarity(const float* a, const float* b, size_t dim) {
    float dist_sq = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        dist_sq += d * d;
    }
    return 1.0f / (1.0f + dist_sq / (float)dim);
}

ArixFMMemoryBank* arix_fm_memory_bank_create(size_t memory_dim, size_t capacity) {
    ArixFMMemoryBank* bank = (ArixFMMemoryBank*)arix_malloc(sizeof(ArixFMMemoryBank), 64);
    if (!bank) return NULL;
    memset(bank, 0, sizeof(ArixFMMemoryBank));

    size_t keys_shape[] = {capacity, memory_dim};
    bank->keys = arix_tensor_zeros(keys_shape, 2, ARIX_FLOAT32);
    bank->values = arix_tensor_zeros(keys_shape, 2, ARIX_FLOAT32);

    size_t ts_shape[] = {capacity};
    bank->timestamps = arix_tensor_zeros(ts_shape, 1, ARIX_FLOAT32);
    bank->access_counts = arix_tensor_zeros(ts_shape, 1, ARIX_FLOAT32);

    if (!bank->keys || !bank->values || !bank->timestamps || !bank->access_counts) {
        arix_fm_memory_bank_destroy(bank);
        return NULL;
    }

    bank->num_entries = 0;
    bank->max_entries = capacity;
    return bank;
}

void arix_fm_memory_bank_destroy(ArixFMMemoryBank* bank) {
    if (!bank) return;
    if (bank->keys) arix_tensor_destroy(bank->keys);
    if (bank->values) arix_tensor_destroy(bank->values);
    if (bank->timestamps) arix_tensor_destroy(bank->timestamps);
    if (bank->access_counts) arix_tensor_destroy(bank->access_counts);
    arix_free(bank, sizeof(ArixFMMemoryBank));
}

int arix_fm_memory_bank_write(ArixFMMemoryBank* bank, const ArixTensor* key, const ArixTensor* value) {
    if (!bank || !key || !value) return 0;
    size_t dim = bank->keys->shape[1];
    float* keys_data = (float*)bank->keys->data;
    float* vals_data = (float*)bank->values->data;
    float* ts_data = (float*)bank->timestamps->data;
    float* ac_data = (float*)bank->access_counts->data;
    float* kd = (float*)key->data;
    float* vd = (float*)value->data;

    for (size_t i = 0; i < bank->num_entries; i++) {
        float sim = euclidean_similarity(keys_data + i * dim, kd, dim);
        if (sim > 0.95f) {
            memcpy(vals_data + i * dim, vd, dim * sizeof(float));
            ts_data[i] = (float)(bank->num_entries);
            ac_data[i] = 0.0f;
            return 1;
        }
    }

    if (bank->num_entries < bank->max_entries) {
        size_t idx = bank->num_entries;
        memcpy(keys_data + idx * dim, kd, dim * sizeof(float));
        memcpy(vals_data + idx * dim, vd, dim * sizeof(float));
        ts_data[idx] = (float)(bank->num_entries);
        ac_data[idx] = 0.0f;
        bank->num_entries++;
        return 1;
    }

    size_t evict_idx = 0;
    float worst_score = 1e30f;
    float current_time = (float)(bank->num_entries);
    for (size_t i = 0; i < bank->max_entries; i++) {
        float age = current_time - ts_data[i];
        if (age < 1.0f) age = 1.0f;
        float score = ac_data[i] / age;
        if (score < worst_score) {
            worst_score = score;
            evict_idx = i;
        }
    }

    memcpy(keys_data + evict_idx * dim, kd, dim * sizeof(float));
    memcpy(vals_data + evict_idx * dim, vd, dim * sizeof(float));
    ts_data[evict_idx] = current_time;
    ac_data[evict_idx] = 0.0f;
    return 1;
}

ArixTensor* arix_fm_memory_bank_read(ArixFMMemoryBank* bank, const ArixTensor* key) {
    if (!bank || !key || bank->num_entries == 0) return NULL;
    size_t dim = bank->keys->shape[1];
    float* keys_data = (float*)bank->keys->data;
    float* vals_data = (float*)bank->values->data;
    float* ts_data = (float*)bank->timestamps->data;
    float* ac_data = (float*)bank->access_counts->data;
    float* kd = (float*)key->data;

    size_t best_idx = 0;
    float best_sim = -1.0f;
    for (size_t i = 0; i < bank->num_entries; i++) {
        float sim = euclidean_similarity(keys_data + i * dim, kd, dim);
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = i;
        }
    }

    if (best_sim < 0.5f) return NULL;

    size_t out_shape[] = {dim};
    ArixTensor* result = arix_tensor_create(out_shape, 1, ARIX_FLOAT32);
    if (result) {
        memcpy((float*)result->data, vals_data + best_idx * dim, dim * sizeof(float));
    }

    ac_data[best_idx] += 1.0f;
    ts_data[best_idx] = (float)(bank->num_entries);
    return result;
}

typedef struct {
    float score;
    size_t idx;
} RetentionEntry;

static int cmp_retention(const void* a, const void* b) {
    float sa = ((const RetentionEntry*)a)->score;
    float sb = ((const RetentionEntry*)b)->score;
    if (sa < sb) return 1;
    if (sa > sb) return -1;
    return 0;
}

void arix_fm_memory_bank_forget(ArixFMMemoryBank* bank, float forget_rate) {
    if (!bank || bank->num_entries == 0) return;
    size_t dim = bank->keys->shape[1];
    float* keys_data = (float*)bank->keys->data;
    float* vals_data = (float*)bank->values->data;
    float* ts_data = (float*)bank->timestamps->data;
    float* ac_data = (float*)bank->access_counts->data;

    float current_time = (float)(bank->num_entries);
    size_t n = bank->num_entries;
    RetentionEntry* entries = (RetentionEntry*)malloc(n * sizeof(RetentionEntry));
    if (!entries) return;

    for (size_t i = 0; i < n; i++) {
        float age = current_time - ts_data[i];
        if (age < 1.0f) age = 1.0f;
        entries[i].score = (ac_data[i] * ac_data[i]) / age;
        entries[i].idx = i;
    }

    qsort(entries, n, sizeof(RetentionEntry), cmp_retention);

    size_t to_forget = (size_t)(forget_rate * (float)n);
    if (to_forget > n) to_forget = n;
    if (to_forget >= n) to_forget = n - 1;

    for (size_t i = n - to_forget; i < n; i++) {
        size_t idx = entries[i].idx;
        memset(keys_data + idx * dim, 0, dim * sizeof(float));
        memset(vals_data + idx * dim, 0, dim * sizeof(float));
        ts_data[idx] = 0.0f;
        ac_data[idx] = 0.0f;
    }

    free(entries);
}
