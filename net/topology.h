#ifndef SNEPPX_NET_TOPOLOGY_H
#define SNEPPX_NET_TOPOLOGY_H
/*
 * Network Topology Abstraction — v1.0 (distributed communication patterns)
 *
 * PURPOSE: Maps nodes to logical network positions for collective
 * communication.  Supports ring (all-reduce), tree (broadcast), and
 * arbitrary graph (gossip) topologies.  Used by the distributed
 * training coordinator to choose communication schedules.
 *
 * DEPENDENCIES: SNEPPX_grpc_service.h, SNEPPX_socket_comm.h
 * VERSION: v1.0
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SNEPPX_TOPOLOGY_RING,
    SNEPPX_TOPOLOGY_TREE,
    SNEPPX_TOPOLOGY_GRAPH,
} SNEPPXTopologyType;

typedef struct {
    int    rank;
    int    prev_rank;
    int    next_rank;
    int    parent_rank;
    int    children[16];
    int    num_children;
    char   host[256];
    int    data_port;
} SNEPPXTopologyNode;

typedef struct {
    SNEPPXTopologyType  type;
    int               world_size;
    SNEPPXTopologyNode* nodes;
} SNEPPXTopology;

/* ---------- Topology construction ---------- */
SNEPPXTopology* SNEPPX_topology_create_ring(int world_size);
SNEPPXTopology* SNEPPX_topology_create_tree(int world_size, int branching_factor);
SNEPPXTopology* SNEPPX_topology_create_graph(int world_size, const int* adjacency_matrix);
void          SNEPPX_topology_destroy(SNEPPXTopology* topo);

/* ---------- Neighbor queries ---------- */
int SNEPPX_topology_get_prev(const SNEPPXTopology* topo, int rank);
int SNEPPX_topology_get_next(const SNEPPXTopology* topo, int rank);
int SNEPPX_topology_get_parent(const SNEPPXTopology* topo, int rank);
int SNEPPX_topology_get_children(const SNEPPXTopology* topo, int rank, int** children, int* count);

/* ---------- Route calculation (v1.0) ---------- */
int  SNEPPX_topology_compute_route(const SNEPPXTopology* topo, int src, int dst, int** path, int* path_len);
void SNEPPX_topology_free_route(int* path);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_NET_TOPOLOGY_H */
