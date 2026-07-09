/*
 * Network Topology Implementation — SKELETON
 * VERSION: v1.0
 */

#include "topology.h"
#include <stdlib.h>
#include <string.h>

SNEPPXTopology* SNEPPX_topology_create_ring(int world_size) {
    (void)world_size;
    SNEPPXTopology* t = (SNEPPXTopology*)calloc(1, sizeof(SNEPPXTopology));
    if (t) t->type = SNEPPX_TOPOLOGY_RING;
    return t;
}

SNEPPXTopology* SNEPPX_topology_create_tree(int world_size, int branching_factor) {
    (void)world_size; (void)branching_factor;
    SNEPPXTopology* t = (SNEPPXTopology*)calloc(1, sizeof(SNEPPXTopology));
    if (t) t->type = SNEPPX_TOPOLOGY_TREE;
    return t;
}

SNEPPXTopology* SNEPPX_topology_create_graph(int world_size, const int* adjacency_matrix) {
    (void)world_size; (void)adjacency_matrix;
    SNEPPXTopology* t = (SNEPPXTopology*)calloc(1, sizeof(SNEPPXTopology));
    if (t) t->type = SNEPPX_TOPOLOGY_GRAPH;
    return t;
}

void SNEPPX_topology_destroy(SNEPPXTopology* topo) {
    if (topo) free(topo->nodes);
    free(topo);
}

int SNEPPX_topology_get_prev(const SNEPPXTopology* topo, int rank) {
    (void)topo; (void)rank; return -1;
}

int SNEPPX_topology_get_next(const SNEPPXTopology* topo, int rank) {
    (void)topo; (void)rank; return -1;
}

int SNEPPX_topology_get_parent(const SNEPPXTopology* topo, int rank) {
    (void)topo; (void)rank; return -1;
}

int SNEPPX_topology_get_children(const SNEPPXTopology* topo, int rank, int** children, int* count) {
    (void)topo; (void)rank;
    if (children) *children = NULL;
    if (count) *count = 0;
    return 0;
}

int SNEPPX_topology_compute_route(const SNEPPXTopology* topo, int src, int dst, int** path, int* path_len) {
    (void)topo; (void)src; (void)dst;
    if (path) *path = NULL;
    if (path_len) *path_len = 0;
    return 0;
}

void SNEPPX_topology_free_route(int* path) { free(path); }
