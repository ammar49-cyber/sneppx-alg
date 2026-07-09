#ifndef SNEPPX_FM_H
#define SNEPPX_FM_H

#include "multidimensional_tensor_engine.h"
#include "automatic_differentiation_framework.h"
#include <stddef.h>
#include <stdint.h>

typedef enum {
    SNEPPX_SYNC_ALL_REDUCE,
    SNEPPX_SYNC_GOSSIP,
    SNEPPX_SYNC_TOPOLOGY
} SNEPPXFMSyncMethod;

typedef struct {
    SNEPPXTensor* keys;
    SNEPPXTensor* values;
    SNEPPXTensor* timestamps;
    SNEPPXTensor* access_counts;
    size_t num_entries;
    size_t max_entries;
} SNEPPXFMMemoryBank;

typedef struct {
    size_t node_id;
    SNEPPXFMMemoryBank* memory_bank;
    SNEPPXTensor* gradient_accumulator;
    uint64_t last_sync_time;
    int is_online;
    float trust_score;
} SNEPPXFMNode;

typedef struct {
    size_t num_nodes;
    size_t memory_dim;
    size_t memory_capacity;
    size_t sync_interval;
    SNEPPXFMSyncMethod sync_method;
    float compression_ratio;
    float privacy_epsilon;
    int catastrophic_forgetting_protection;
    float ewm_alpha;
} SNEPPXFMConfig;

typedef struct {
    SNEPPXTensor* global_memory;
    size_t sync_round;
    float* node_contributions;
    size_t* conflict_log;
    size_t conflict_count;
} SNEPPXFMSyncState;

typedef struct {
    SNEPPXFMNode** nodes;
    SNEPPXFMConfig config;
    SNEPPXFMSyncState sync_state;
    size_t step_counter;
} SNEPPXFMController;

SNEPPXFMConfig SNEPPX_fm_config_default(void);
SNEPPXFMMemoryBank* SNEPPX_fm_memory_bank_create(size_t memory_dim, size_t capacity);
void SNEPPX_fm_memory_bank_destroy(SNEPPXFMMemoryBank* bank);
int SNEPPX_fm_memory_bank_write(SNEPPXFMMemoryBank* bank, const SNEPPXTensor* key, const SNEPPXTensor* value);
SNEPPXTensor* SNEPPX_fm_memory_bank_read(SNEPPXFMMemoryBank* bank, const SNEPPXTensor* key);
void SNEPPX_fm_memory_bank_forget(SNEPPXFMMemoryBank* bank, float forget_rate);
SNEPPXFMNode* SNEPPX_fm_node_create(size_t node_id, size_t memory_dim, size_t capacity);
void SNEPPX_fm_node_destroy(SNEPPXFMNode* node);
SNEPPXFMController* SNEPPX_fm_controller_create(const SNEPPXFMConfig* config);
void SNEPPX_fm_controller_destroy(SNEPPXFMController* ctrl);
int SNEPPX_fm_sync_all_reduce(SNEPPXFMController* ctrl);
int SNEPPX_fm_sync_gossip(SNEPPXFMController* ctrl, size_t num_pairs);
int SNEPPX_fm_sync_topology(SNEPPXFMController* ctrl);
SNEPPXTensor* SNEPPX_fm_compress_gradients(const SNEPPXTensor* gradients, float ratio);
void SNEPPX_fm_add_privacy_noise(SNEPPXTensor* data, float epsilon);
int SNEPPX_fm_forward(SNEPPXFMController* ctrl, size_t node_id, const SNEPPXTensor* input, SNEPPXTensor** output);

// Training graph support (FM has no trainable params, pass-through only)
size_t SNEPPX_fm_get_params(const SNEPPXFMController* ctrl, SNEPPXTensor** out_params, size_t max_params);
int SNEPPX_fm_build_train_graph(SNEPPXFMController* ctrl, SNEPPXTape* tape,
                               SNEPPXVariable* input_var,
                               SNEPPXVariable** weight_vars, size_t num_weights,
                               SNEPPXVariable** output_var);

// ── Error-compensated gradient compression (EF-SGD) ──────────────────────────

typedef struct {
    SNEPPXTensor* error_buffer;
    SNEPPXTensor* compressed_grad;
    float compression_ratio;
    size_t dim;
} SNEPPXFMErrorFeedback;

SNEPPXFMErrorFeedback* SNEPPX_fm_error_feedback_create(size_t dim, float ratio);
void SNEPPX_fm_error_feedback_destroy(SNEPPXFMErrorFeedback* ef);
SNEPPXTensor* SNEPPX_fm_compress_with_error(SNEPPXFMErrorFeedback* ef, const SNEPPXTensor* gradient);

// ── Exponential moving average (catastrophic forgetting) ─────────────────────

void SNEPPX_fm_ewm_update(SNEPPXFMMemoryBank* bank, float alpha);

// ── Adaptive sync frequency ──────────────────────────────────────────────────

float SNEPPX_fm_compute_change_rate(SNEPPXFMMemoryBank* bank, const SNEPPXTensor* new_values);
size_t SNEPPX_fm_adaptive_sync_interval(SNEPPXFMController* ctrl, float base_interval);

// ── Gradient send / receive ──────────────────────────────────────────────────

int SNEPPX_fm_send_gradients(SNEPPXFMController* ctrl, size_t node_id, const SNEPPXTensor* gradients);
int SNEPPX_fm_receive_gradients(SNEPPXFMController* ctrl, size_t node_id, SNEPPXTensor* aggregated);

#endif /* SNEPPX_FM_H */
