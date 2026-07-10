#include "../../net/topology.h"
#include "../../net/topology.c"
#include <stdio.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (%s)\n", msg, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

static void run_test(const char* name, void (*test_fn)(void)) {
    printf("Running %s... ", name);
    fflush(stdout);
    test_fn();
    printf("PASS\n");
    tests_passed++;
}

static void test_topology_create_destroy(void) {
    SNEPPXTopology* topo = SNEPPX_topology_create_ring(4);
    ASSERT(topo != NULL, "topology created");
    ASSERT(topo->type == SNEPPX_TOPOLOGY_RING, "ring type");
    SNEPPX_topology_destroy(topo);
}

static void test_topology_ring(void) {
    SNEPPXTopology* topo = SNEPPX_topology_create_ring(8);
    ASSERT(topo != NULL, "ring topology");
    ASSERT(topo->type == SNEPPX_TOPOLOGY_RING, "ring type");
    SNEPPX_topology_destroy(topo);
}

static void test_topology_tree(void) {
    SNEPPXTopology* topo = SNEPPX_topology_create_tree(8, 2);
    ASSERT(topo != NULL, "tree topology");
    ASSERT(topo->type == SNEPPX_TOPOLOGY_TREE, "tree type");
    SNEPPX_topology_destroy(topo);
}

static void test_topology_graph(void) {
    int adj[9 * 9];
    memset(adj, 0, sizeof(adj));
    adj[0 * 9 + 1] = 1;
    adj[1 * 9 + 2] = 1;
    adj[2 * 9 + 0] = 1;
    SNEPPXTopology* topo = SNEPPX_topology_create_graph(9, adj);
    ASSERT(topo != NULL, "graph topology");
    ASSERT(topo->type == SNEPPX_TOPOLOGY_GRAPH, "graph type");
    SNEPPX_topology_destroy(topo);
}

int main(void) {
    run_test("topology_create_destroy", test_topology_create_destroy);
    run_test("topology_ring", test_topology_ring);
    run_test("topology_tree", test_topology_tree);
    run_test("topology_graph", test_topology_graph);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
