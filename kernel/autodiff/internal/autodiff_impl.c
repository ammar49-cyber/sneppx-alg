#include "autodiff_impl.h"
#include <stdlib.h>
#include <string.h>

int arix_grad_graph_init(ArixGradGraph* graph) {
    if (!graph) return -1;
    memset(graph, 0, sizeof(*graph));
    graph->capacity = 64;
    graph->nodes = (ArixGradNode**)calloc(graph->capacity, sizeof(ArixGradNode*));
    if (!graph->nodes) return -1;
    graph->topological_order = NULL;
    graph->needs_rebuild = 1;
    return 0;
}

void arix_grad_graph_destroy(ArixGradGraph* graph) {
    if (!graph) return;
    for (size_t i = 0; i < graph->num_nodes; i++)
        arix_grad_node_destroy(graph->nodes[i]);
    free(graph->nodes);
    free(graph->topological_order);
    memset(graph, 0, sizeof(*graph));
}

static uint64_t g_next_id = 1;

int arix_grad_node_create(ArixGradNodeType type, ArixGradNode** node) {
    if (!node) return -1;
    *node = (ArixGradNode*)calloc(1, sizeof(ArixGradNode));
    if (!*node) return -1;
    (*node)->type = type;
    (*node)->ref_count = 1;
    (*node)->id = g_next_id++;
    return 0;
}

void arix_grad_node_destroy(ArixGradNode* node) {
    if (!node) return;
    free(node->inputs);
    free(node->consumers);
    free(node->saved_tensors);
    free(node);
}

void arix_grad_node_add_input(ArixGradNode* node, ArixGradNode* input) {
    if (!node || !input) return;
    int n = node->num_inputs;
    ArixGradNode** new_inputs = (ArixGradNode**)realloc(node->inputs, (n + 1) * sizeof(ArixGradNode*));
    if (!new_inputs) return;
    node->inputs = new_inputs;
    node->inputs[n] = input;
    node->num_inputs = n + 1;
    if (input) input->ref_count++;
}

void arix_grad_node_add_consumer(ArixGradNode* node, ArixGradNode* consumer) {
    if (!node || !consumer) return;
    int n = node->num_consumers;
    ArixGradNode** new_consumers = (ArixGradNode**)realloc(node->consumers, (n + 1) * sizeof(ArixGradNode*));
    if (!new_consumers) return;
    node->consumers = new_consumers;
    node->consumers[n] = consumer;
    node->num_consumers = n + 1;
}

static void toposort_dfs(ArixGradNode* node, ArixGradNode*** order, size_t* count, size_t* capacity, int* visited) {
    if (!node || visited[node->id % 65536]) return;
    visited[node->id % 65536] = 1;
    for (int i = 0; i < node->num_inputs; i++) {
        if (node->inputs[i])
            toposort_dfs(node->inputs[i], order, count, capacity, visited);
    }
    if (*count >= *capacity) {
        *capacity *= 2;
        *order = (ArixGradNode**)realloc(*order, *capacity * sizeof(ArixGradNode*));
    }
    if (*order)
        (*order)[(*count)++] = node;
}

int arix_grad_graph_toposort(ArixGradGraph* graph) {
    if (!graph || graph->num_nodes == 0) return 0;
    free(graph->topological_order);
    graph->topological_order = NULL;
    size_t capacity = graph->num_nodes > 0 ? graph->num_nodes : 1;
    graph->topological_order = (ArixGradNode**)malloc(capacity * sizeof(ArixGradNode*));
    if (!graph->topological_order) return -1;
    size_t count = 0;
    int visited[65536] = {0};
    for (size_t i = 0; i < graph->num_nodes; i++) {
        if (graph->nodes[i])
            toposort_dfs(graph->nodes[i], &graph->topological_order, &count, &capacity, visited);
    }
    graph->needs_rebuild = 0;
    return 0;
}

int arix_grad_graph_backward(ArixGradGraph* graph, ArixGradNode* root) {
    if (!graph || !root) return -1;
    if (graph->needs_rebuild) arix_grad_graph_toposort(graph);
    if (!graph->topological_order) return -1;

    for (size_t i = graph->num_nodes; i > 0; i--) {
        ArixGradNode* node = graph->topological_order[i - 1];
        if (!node || !node->grad_fn) continue;
        ((void (*)(void*, void*))node->grad_fn)(node->saved_tensors, NULL);
    }
    return 0;
}

int arix_grad_save_tensor(ArixGradNode* node, void* tensor) {
    if (!node) return -1;
    node->saved_tensors = tensor;
    return 0;
}

void* arix_grad_restore_tensor(ArixGradNode* node, int idx) {
    (void)idx;
    if (!node) return NULL;
    return node->saved_tensors;
}
