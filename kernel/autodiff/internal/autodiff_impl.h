#ifndef SNEPPX_AUTODIFF_INTERNAL_H
#define SNEPPX_AUTODIFF_INTERNAL_H
/*
 * Autodiff Internal Implementation — v0.5
 *
 * PURPOSE: Internal gradient graph representation, tape node structures,
 * and backward-pass scheduler.  The external API (automatic_differentiation_framework.h) exposes
 * a high-level Variable/GradFn interface; this header defines the internal
 * graph topology, reference-counted tensors, and topological sort utilities.
 *
 * DEPENDENCIES: automatic_differentiation_framework.h, multidimensional_tensor_engine.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SNEPPX_GRAD_NODE_MUL,
    SNEPPX_GRAD_NODE_ADD,
    SNEPPX_GRAD_NODE_SUB,
    SNEPPX_GRAD_NODE_MATMUL,
    SNEPPX_GRAD_NODE_CONV2D,
    SNEPPX_GRAD_NODE_RELU,
    SNEPPX_GRAD_NODE_SIGMOID,
    SNEPPX_GRAD_NODE_TANH,
    SNEPPX_GRAD_NODE_SOFTMAX,
    SNEPPX_GRAD_NODE_CROSS_ENTROPY,
    SNEPPX_GRAD_NODE_RESHAPE,
    SNEPPX_GRAD_NODE_CUSTOM,
} SNEPPXGradNodeType;

typedef struct SNEPPXGradNode {
    SNEPPXGradNodeType   type;
    void*              grad_fn;
    struct SNEPPXGradNode** inputs;
    int                num_inputs;
    struct SNEPPXGradNode** consumers;
    int                num_consumers;
    void*              saved_tensors;      /* for backward */
    int                ref_count;
    uint64_t           id;
} SNEPPXGradNode;

typedef struct {
    SNEPPXGradNode**    nodes;
    size_t            num_nodes;
    size_t            capacity;
    size_t            num_leafs;
    SNEPPXGradNode**    topological_order;
    int               needs_rebuild;
} SNEPPXGradGraph;

int  SNEPPX_grad_graph_init(SNEPPXGradGraph* graph);
void SNEPPX_grad_graph_destroy(SNEPPXGradGraph* graph);

int  SNEPPX_grad_node_create(SNEPPXGradNodeType type, SNEPPXGradNode** node);
void SNEPPX_grad_node_destroy(SNEPPXGradNode* node);
void SNEPPX_grad_node_add_input(SNEPPXGradNode* node, SNEPPXGradNode* input);
void SNEPPX_grad_node_add_consumer(SNEPPXGradNode* node, SNEPPXGradNode* consumer);

int  SNEPPX_grad_graph_toposort(SNEPPXGradGraph* graph);
int  SNEPPX_grad_graph_backward(SNEPPXGradGraph* graph, SNEPPXGradNode* root);

/* ---------- Saved tensor helpers ---------- */
int  SNEPPX_grad_save_tensor(SNEPPXGradNode* node, void* tensor);
void* SNEPPX_grad_restore_tensor(SNEPPXGradNode* node, int idx);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_AUTODIFF_INTERNAL_H */
