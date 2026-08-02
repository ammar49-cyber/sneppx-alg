#ifndef SNEPPX_GRAPH_CONVOLUTIONAL_NETWORK_H
#define SNEPPX_GRAPH_CONVOLUTIONAL_NETWORK_H
#include <stddef.h>
#include <stdbool.h>
/*
 * SNEPPX - Graph Convolutional Network
 *
 * WHAT
 *   Graph Convolutional Network.
 *
 * CONCEPT
 *   GCN configuration and forward API for graph-structured data.
 *
 * ROLE
 *   Declares the GCN API for graph neural network operations within the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal module).
 */


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
