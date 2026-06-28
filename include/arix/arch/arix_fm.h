#ifndef ARIX_FM_H
#define ARIX_FM_H

#include "arix_tensor.h"
#include <stddef.h>
#include <stdint.h>

typedef enum {
    ARIX_SYNC_ALL_REDUCE,
    ARIX_SYNC_GOSSIP,
    ARIX_SYNC_TOPOLOGY
} ArixFMSyncMethod;

typedef struct {
    ArixTensor* keys;
    ArixTensor* values;
    ArixTensor* timestamps;
    ArixTensor* access_counts;
    size_t num_entries;
    size_t max_entries;
} ArixFMMemoryBank;

typedef struct {
    size_t node_id;
    ArixFMMemoryBank* memory_bank;
    ArixTensor* gradient_accumulator;
    uint64_t last_sync_time;
    int is_online;
    float trust_score;
} ArixFMNode;

typedef struct {
    size_t num_nodes;
    size_t memory_dim;
    size_t memory_capacity;
    size_t sync_interval;
    ArixFMSyncMethod sync_method;
    float compression_ratio;
    float privacy_epsilon;
    int catastrophic_forgetting_protection;
    float ewm_alpha;
} ArixFMConfig;

typedef struct {
    ArixTensor* global_memory;
    size_t sync_round;
    float* node_contributions;
    size_t* conflict_log;
    size_t conflict_count;
} ArixFMSyncState;

typedef struct {
    ArixFMNode** nodes;
    ArixFMConfig config;
    ArixFMSyncState sync_state;
    size_t step_counter;
} ArixFMController;

ArixFMConfig arix_fm_config_default(void);
ArixFMMemoryBank* arix_fm_memory_bank_create(size_t memory_dim, size_t capacity);
void arix_fm_memory_bank_destroy(ArixFMMemoryBank* bank);
int arix_fm_memory_bank_write(ArixFMMemoryBank* bank, const ArixTensor* key, const ArixTensor* value);
ArixTensor* arix_fm_memory_bank_read(ArixFMMemoryBank* bank, const ArixTensor* key);
void arix_fm_memory_bank_forget(ArixFMMemoryBank* bank, float forget_rate);
ArixFMNode* arix_fm_node_create(size_t node_id, size_t memory_dim, size_t capacity);
void arix_fm_node_destroy(ArixFMNode* node);
ArixFMController* arix_fm_controller_create(const ArixFMConfig* config);
void arix_fm_controller_destroy(ArixFMController* ctrl);
int arix_fm_sync_all_reduce(ArixFMController* ctrl);
int arix_fm_sync_gossip(ArixFMController* ctrl, size_t num_pairs);
int arix_fm_sync_topology(ArixFMController* ctrl);
ArixTensor* arix_fm_compress_gradients(const ArixTensor* gradients, float ratio);
void arix_fm_add_privacy_noise(ArixTensor* data, float epsilon);
int arix_fm_forward(ArixFMController* ctrl, size_t node_id, const ArixTensor* input, ArixTensor** output);

#endif /* ARIX_FM_H */
