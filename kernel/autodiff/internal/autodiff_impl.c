/*
 * Autodiff Internal Implementation — SKELETON
 * VERSION: v0.5
 */

#include "autodiff_impl.h"
#include <stdlib.h>
#include <string.h>

int arix_grad_graph_init(ArixGradGraph* graph) {
    if (!graph) return -1;
    memset(graph, 0, sizeof(*graph));
    return 0;
}

void arix_grad_graph_destroy(ArixGradGraph* graph) {
    if (!graph) return;
    for (size_t i = 0; i < graph->num_nodes; i++)
        arix_grad_node_destroy(graph->nodes[i]);
    free(graph->nodes);
    free(graph->topological_order);
}

int arix_grad_node_create(ArixGradNodeType type, ArixGradNode** node) {
    if (!node) return -1;
    *node = (ArixGradNode*)calloc(1, sizeof(ArixGradNode));
    if (*node) (*node)->type = type;
    return *node ? 0 : -1;
}

void arix_grad_node_destroy(ArixGradNode* node) {
    if (!node) return;
    free(node->inputs);
    free(node->consumers);
    free(node->saved_tensors);
    free(node);
}

void arix_grad_node_add_input(ArixGradNode* node, ArixGradNode* input) {
    (void)node; (void)input;
}

void arix_grad_node_add_consumer(ArixGradNode* node, ArixGradNode* consumer) {
    (void)node; (void)consumer;
}

int arix_grad_graph_toposort(ArixGradGraph* graph) {
    (void)graph; return 0;
}

int arix_grad_graph_backward(ArixGradGraph* graph, ArixGradNode* root) {
    (void)graph; (void)root; return 0;
}

int arix_grad_save_tensor(ArixGradNode* node, void* tensor) {
    (void)node; (void)tensor; return 0;
}

void* arix_grad_restore_tensor(ArixGradNode* node, int idx) {
    (void)node; (void)idx; return NULL;
}
