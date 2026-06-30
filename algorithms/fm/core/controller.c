#include "fractal_memory_orchestrator.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>

ArixFMConfig arix_fm_config_default(void) {
    ArixFMConfig cfg;
    cfg.num_nodes = 4;
    cfg.memory_dim = 512;
    cfg.memory_capacity = 1024;
    cfg.sync_interval = 100;
    cfg.sync_method = ARIX_SYNC_ALL_REDUCE;
    cfg.compression_ratio = 0.1f;
    cfg.privacy_epsilon = 1.0f;
    cfg.catastrophic_forgetting_protection = 1;
    cfg.ewm_alpha = 0.9f;
    return cfg;
}

ArixFMController* arix_fm_controller_create(const ArixFMConfig* config) {
    if (!config) return NULL;
    ArixFMController* ctrl = (ArixFMController*)arix_malloc(sizeof(ArixFMController), 64);
    if (!ctrl) return NULL;
    memset(ctrl, 0, sizeof(ArixFMController));

    ctrl->config = *config;
    ctrl->step_counter = 0;

    ctrl->nodes = (ArixFMNode**)arix_malloc(config->num_nodes * sizeof(ArixFMNode*), 64);
    if (!ctrl->nodes) {
        arix_free(ctrl, sizeof(ArixFMController));
        return NULL;
    }
    memset(ctrl->nodes, 0, config->num_nodes * sizeof(ArixFMNode*));

    for (size_t i = 0; i < config->num_nodes; i++) {
        ctrl->nodes[i] = arix_fm_node_create(i, config->memory_dim, config->memory_capacity);
        if (!ctrl->nodes[i]) {
            arix_fm_controller_destroy(ctrl);
            return NULL;
        }
    }

    size_t gm_shape[] = {config->memory_capacity, config->memory_dim};
    ctrl->sync_state.global_memory = arix_tensor_zeros(gm_shape, 2, ARIX_FLOAT32);
    if (!ctrl->sync_state.global_memory) {
        arix_fm_controller_destroy(ctrl);
        return NULL;
    }
    ctrl->sync_state.sync_round = 0;
    ctrl->sync_state.node_contributions = (float*)arix_malloc(config->num_nodes * sizeof(float), 64);
    if (ctrl->sync_state.node_contributions) {
        memset(ctrl->sync_state.node_contributions, 0, config->num_nodes * sizeof(float));
    }
    ctrl->sync_state.conflict_log = NULL;
    ctrl->sync_state.conflict_count = 0;

    return ctrl;
}

void arix_fm_controller_destroy(ArixFMController* ctrl) {
    if (!ctrl) return;
    if (ctrl->nodes) {
        for (size_t i = 0; i < ctrl->config.num_nodes; i++) {
            if (ctrl->nodes[i]) arix_fm_node_destroy(ctrl->nodes[i]);
        }
        arix_free(ctrl->nodes, ctrl->config.num_nodes * sizeof(ArixFMNode*));
    }
    if (ctrl->sync_state.global_memory) arix_tensor_destroy(ctrl->sync_state.global_memory);
    if (ctrl->sync_state.node_contributions) arix_free(ctrl->sync_state.node_contributions, ctrl->config.num_nodes * sizeof(float));
    if (ctrl->sync_state.conflict_log) arix_free(ctrl->sync_state.conflict_log, ctrl->sync_state.conflict_count * sizeof(size_t));
    arix_free(ctrl, sizeof(ArixFMController));
}
