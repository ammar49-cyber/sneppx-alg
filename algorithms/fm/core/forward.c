#include "fractal_memory_orchestrator.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

int arix_fm_forward(ArixFMController* ctrl, size_t node_id, const ArixTensor* input, ArixTensor** output) {
    if (!ctrl || !input || !output) return 1;
    if (node_id >= ctrl->config.num_nodes) return 1;
    if (!ctrl->nodes[node_id]->is_online) return 1;

    ArixFMNode* node = ctrl->nodes[node_id];
    ArixFMMemoryBank* bank = node->memory_bank;

    ArixTensor* result = NULL;

    ArixTensor* retrieved = arix_fm_memory_bank_read(bank, input);
    if (retrieved) {
        float alpha = ctrl->config.ewm_alpha;
        size_t sz = input->size < retrieved->size ? input->size : retrieved->size;
        result = arix_tensor_create(input->shape, input->ndim, ARIX_FLOAT32);
        if (result) {
            float* rd = (float*)result->data;
            float* id = (float*)input->data;
            float* vd = (float*)retrieved->data;
            for (size_t i = 0; i < sz; i++) {
                rd[i] = alpha * id[i] + (1.0f - alpha) * vd[i];
            }
        }
        arix_tensor_destroy(retrieved);
    } else {
        result = arix_tensor_create(input->shape, input->ndim, ARIX_FLOAT32);
        if (result) {
            memcpy((float*)result->data, (float*)input->data, input->size * sizeof(float));
        }
        arix_fm_memory_bank_write(bank, input, input);
    }

    if (!result) return 1;

    if (node->gradient_accumulator && result->size == node->gradient_accumulator->size) {
        float* ga = (float*)node->gradient_accumulator->data;
        float* rd = (float*)result->data;
        for (size_t i = 0; i < result->size; i++) {
            ga[i] += rd[i] * 0.01f;
        }
    }

    *output = result;

    ctrl->step_counter++;

    if (ctrl->config.sync_interval > 0 && ctrl->step_counter % ctrl->config.sync_interval == 0) {
        switch (ctrl->config.sync_method) {
            case ARIX_SYNC_ALL_REDUCE:
                arix_fm_sync_all_reduce(ctrl);
                break;
            case ARIX_SYNC_GOSSIP:
                arix_fm_sync_gossip(ctrl, ctrl->config.num_nodes);
                break;
            case ARIX_SYNC_TOPOLOGY:
                arix_fm_sync_topology(ctrl);
                break;
        }
    }

    return 0;
}
