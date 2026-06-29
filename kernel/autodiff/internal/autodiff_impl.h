#ifndef ARIX_AUTODIFF_INTERNAL_H
#define ARIX_AUTODIFF_INTERNAL_H
/*
 * Autodiff Internal Implementation — v0.5
 *
 * PURPOSE: Internal gradient graph representation, tape node structures,
 * and backward-pass scheduler.  The external API (arix_autodiff.h) exposes
 * a high-level Variable/GradFn interface; this header defines the internal
 * graph topology, reference-counted tensors, and topological sort utilities.
 *
 * DEPENDENCIES: arix_autodiff.h, arix_tensor.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ARIX_GRAD_NODE_MUL,
    ARIX_GRAD_NODE_ADD,
    ARIX_GRAD_NODE_SUB,
    ARIX_GRAD_NODE_MATMUL,
    ARIX_GRAD_NODE_CONV2D,
    ARIX_GRAD_NODE_RELU,
    ARIX_GRAD_NODE_SIGMOID,
    ARIX_GRAD_NODE_TANH,
    ARIX_GRAD_NODE_SOFTMAX,
    ARIX_GRAD_NODE_CROSS_ENTROPY,
    ARIX_GRAD_NODE_RESHAPE,
    ARIX_GRAD_NODE_CUSTOM,
} ArixGradNodeType;

typedef struct ArixGradNode {
    ArixGradNodeType   type;
    void*              grad_fn;
    struct ArixGradNode** inputs;
    int                num_inputs;
    struct ArixGradNode** consumers;
    int                num_consumers;
    void*              saved_tensors;      /* for backward */
    int                ref_count;
    uint64_t           id;
} ArixGradNode;

typedef struct {
    ArixGradNode**    nodes;
    size_t            num_nodes;
    size_t            capacity;
    size_t            num_leafs;
    ArixGradNode**    topological_order;
    int               needs_rebuild;
} ArixGradGraph;

int  arix_grad_graph_init(ArixGradGraph* graph);
void arix_grad_graph_destroy(ArixGradGraph* graph);

int  arix_grad_node_create(ArixGradNodeType type, ArixGradNode** node);
void arix_grad_node_destroy(ArixGradNode* node);
void arix_grad_node_add_input(ArixGradNode* node, ArixGradNode* input);
void arix_grad_node_add_consumer(ArixGradNode* node, ArixGradNode* consumer);

int  arix_grad_graph_toposort(ArixGradGraph* graph);
int  arix_grad_graph_backward(ArixGradGraph* graph, ArixGradNode* root);

/* ---------- Saved tensor helpers ---------- */
int  arix_grad_save_tensor(ArixGradNode* node, void* tensor);
void* arix_grad_restore_tensor(ArixGradNode* node, int idx);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_AUTODIFF_INTERNAL_H */
