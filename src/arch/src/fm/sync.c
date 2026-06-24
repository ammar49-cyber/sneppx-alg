#include "arix_fm.h"
#include "arix_memory.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

static float uniform_01(void) {
    static unsigned long seed = 123456789;
    seed = seed * 1103515245UL + 12345UL;
    return (float)((seed >> 16) & 0x7FFF) / 32767.0f;
}

static float laplace_noise(float epsilon) {
    float u = uniform_01() - 0.5f;
    return -epsilon * (u > 0 ? 1.0f : -1.0f) * logf(1.0f - 2.0f * fabsf(u) + 1e-10f);
}

void arix_fm_add_privacy_noise(ArixTensor* data, float epsilon) {
    if (!data || epsilon <= 0.0f) return;
    float* d = (float*)data->data;
    for (size_t i = 0; i < data->size; i++) {
        d[i] += laplace_noise(epsilon);
    }
}

typedef struct {
    float val;
    size_t idx;
} MagEntry;

static int cmp_mag(const void* a, const void* b) {
    float fa = fabsf(((const MagEntry*)a)->val);
    float fb = fabsf(((const MagEntry*)b)->val);
    if (fa < fb) return 1;
    if (fa > fb) return -1;
    return 0;
}

ArixTensor* arix_fm_compress_gradients(const ArixTensor* gradients, float ratio) {
    if (!gradients || ratio <= 0.0f) return NULL;
    if (ratio >= 1.0f) {
        ArixTensor* copy = arix_tensor_create(gradients->shape, gradients->ndim, ARIX_FLOAT32);
        if (copy) memcpy((float*)copy->data, (float*)gradients->data, gradients->size * sizeof(float));
        return copy;
    }

    ArixTensor* result = arix_tensor_create(gradients->shape, gradients->ndim, ARIX_FLOAT32);
    if (!result) return NULL;
    memcpy((float*)result->data, (float*)gradients->data, gradients->size * sizeof(float));

    size_t total = gradients->size;
    size_t keep = (size_t)(ratio * (float)total);
    if (keep == 0) keep = 1;
    if (keep > total) keep = total;

    MagEntry* entries = (MagEntry*)malloc(total * sizeof(MagEntry));
    if (!entries) { arix_tensor_destroy(result); return NULL; }

    float* src = (float*)gradients->data;
    for (size_t i = 0; i < total; i++) {
        entries[i].val = src[i];
        entries[i].idx = i;
    }

    qsort(entries, total, sizeof(MagEntry), cmp_mag);

    float* dst = (float*)result->data;
    int* keep_mask = (int*)calloc(total, sizeof(int));
    if (!keep_mask) { free(entries); arix_tensor_destroy(result); return NULL; }

    for (size_t i = 0; i < keep; i++) {
        keep_mask[entries[i].idx] = 1;
    }

    for (size_t i = 0; i < total; i++) {
        if (!keep_mask[i]) dst[i] = 0.0f;
    }

    free(keep_mask);
    free(entries);
    return result;
}

static void average_two_banks(ArixFMMemoryBank* a, ArixFMMemoryBank* b, float alpha) {
    size_t dim = a->keys->shape[1];
    size_t min_entries = a->num_entries < b->num_entries ? a->num_entries : b->num_entries;
    if (min_entries == 0) return;

    float* a_keys = (float*)a->keys->data;
    float* a_vals = (float*)a->values->data;
    float* b_keys = (float*)b->keys->data;
    float* b_vals = (float*)b->values->data;

    for (size_t i = 0; i < min_entries; i++) {
        for (size_t j = 0; j < dim; j++) {
            float avg_key = a_keys[i * dim + j] * alpha + b_keys[i * dim + j] * (1.0f - alpha);
            float avg_val = a_vals[i * dim + j] * alpha + b_vals[i * dim + j] * (1.0f - alpha);
            a_keys[i * dim + j] = avg_key;
            a_vals[i * dim + j] = avg_val;
            b_keys[i * dim + j] = avg_key;
            b_vals[i * dim + j] = avg_val;
        }
    }
}

int arix_fm_sync_all_reduce(ArixFMController* ctrl) {
    if (!ctrl || ctrl->config.num_nodes == 0) return 1;
    size_t dim = ctrl->config.memory_dim;
    size_t cap = ctrl->config.memory_capacity;
    size_t n_nodes = ctrl->config.num_nodes;

    for (size_t slot = 0; slot < cap; slot++) {
        float sum_keys[512], sum_vals[512];
        float total_weight = 0.0f;
        memset(sum_keys, 0, dim * sizeof(float));
        memset(sum_vals, 0, dim * sizeof(float));

        for (size_t n = 0; n < n_nodes; n++) {
            if (!ctrl->nodes[n]->is_online) continue;
            ArixFMMemoryBank* bank = ctrl->nodes[n]->memory_bank;
            if (slot >= bank->num_entries) continue;
            float w = ctrl->nodes[n]->trust_score;
            total_weight += w;
            float* kd = (float*)bank->keys->data + slot * dim;
            float* vd = (float*)bank->values->data + slot * dim;
            for (size_t j = 0; j < dim; j++) {
                sum_keys[j] += kd[j] * w;
                sum_vals[j] += vd[j] * w;
            }
        }

        if (total_weight < 1e-10f) continue;

        float variance = 0.0f;
        for (size_t n = 0; n < n_nodes; n++) {
            if (!ctrl->nodes[n]->is_online) continue;
            ArixFMMemoryBank* bank = ctrl->nodes[n]->memory_bank;
            if (slot >= bank->num_entries) continue;
            float* vd = (float*)bank->values->data + slot * dim;
            for (size_t j = 0; j < dim; j++) {
                float diff = vd[j] - sum_vals[j] / total_weight;
                variance += diff * diff;
            }
        }
        variance /= (float)(n_nodes * dim);

        if (variance > 0.1f && ctrl->sync_state.conflict_log) {
            ctrl->sync_state.conflict_log[ctrl->sync_state.conflict_count++] = slot;
        }

        for (size_t n = 0; n < n_nodes; n++) {
            if (!ctrl->nodes[n]->is_online) continue;
            ArixFMMemoryBank* bank = ctrl->nodes[n]->memory_bank;
            float* kd = (float*)bank->keys->data + slot * dim;
            float* vd = (float*)bank->values->data + slot * dim;
            for (size_t j = 0; j < dim; j++) {
                kd[j] = sum_keys[j] / total_weight;
                vd[j] = sum_vals[j] / total_weight;
            }
        }
    }

    arix_fm_add_privacy_noise(ctrl->sync_state.global_memory, ctrl->config.privacy_epsilon);
    ctrl->sync_state.sync_round++;

    for (size_t n = 0; n < n_nodes; n++) {
        if (!ctrl->nodes[n]->is_online) continue;
        ctrl->nodes[n]->last_sync_time = ctrl->step_counter;
        if (ctrl->sync_state.node_contributions) {
            ctrl->sync_state.node_contributions[n] += 0.1f;
        }
    }

    return 0;
}

int arix_fm_sync_gossip(ArixFMController* ctrl, size_t num_pairs) {
    if (!ctrl || ctrl->config.num_nodes < 2) return 1;
    size_t n_nodes = ctrl->config.num_nodes;

    for (size_t p = 0; p < num_pairs; p++) {
        size_t a = (size_t)(uniform_01() * (float)n_nodes) % n_nodes;
        size_t b = (size_t)(uniform_01() * (float)n_nodes) % n_nodes;
        if (a == b) { b = (a + 1) % n_nodes; }
        if (!ctrl->nodes[a]->is_online || !ctrl->nodes[b]->is_online) continue;

        average_two_banks(ctrl->nodes[a]->memory_bank, ctrl->nodes[b]->memory_bank, 0.5f);
        ctrl->nodes[a]->last_sync_time = ctrl->step_counter;
        ctrl->nodes[b]->last_sync_time = ctrl->step_counter;
    }

    ctrl->sync_state.sync_round++;
    return 0;
}

int arix_fm_sync_topology(ArixFMController* ctrl) {
    if (!ctrl || ctrl->config.num_nodes < 2) return 1;
    size_t n_nodes = ctrl->config.num_nodes;

    for (size_t i = 0; i < n_nodes; i++) {
        size_t next = (i + 1) % n_nodes;
        if (!ctrl->nodes[i]->is_online || !ctrl->nodes[next]->is_online) continue;
        average_two_banks(ctrl->nodes[i]->memory_bank, ctrl->nodes[next]->memory_bank, 0.5f);
        ctrl->nodes[i]->last_sync_time = ctrl->step_counter;
        ctrl->nodes[next]->last_sync_time = ctrl->step_counter;
    }

    ctrl->sync_state.sync_round++;
    return 0;
}
