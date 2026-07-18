#ifndef SNEPPX_GRAPH_CONVOLUTIONAL_NETWORK_H
#define SNEPPX_GRAPH_CONVOLUTIONAL_NETWORK_H
#include <stddef.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXGCN SNEPPXGCN;

SNEPPXGCN* SNEPPX_gcn_create(size_t in_features, size_t out_features, size_t hidden_features,
    int num_layers, float dropout);
void SNEPPX_gcn_destroy(void* gcn);
int SNEPPX_gcn_forward(void* gcn, const float* adj, const float* features, size_t num_nodes, float* output);

#ifdef __cplusplus
}
#endif
#endif
