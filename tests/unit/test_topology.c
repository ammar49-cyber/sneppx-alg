#include "network_topology.h"
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
    SNEPPXTopology* topo = SNEPPX_topology_create(4);
    ASSERT(topo != NULL, "topology created");
    ASSERT(topo->num_nodes == 4, "4 nodes");
    SNEPPX_topology_destroy(topo);
}

static void test_topology_connect_disconnect(void) {
    SNEPPXTopology* topo = SNEPPX_topology_create(3);
    SNEPPX_topology_connect(topo, 0, 1);
    SNEPPX_topology_connect(topo, 1, 2);
    ASSERT(SNEPPX_topology_is_connected(topo, 0, 1), "0-1 connected");
    ASSERT(SNEPPX_topology_is_connected(topo, 1, 0), "1-0 connected (undirected)");
    ASSERT(!SNEPPX_topology_is_connected(topo, 0, 2), "0-2 not connected");
    SNEPPX_topology_disconnect(topo, 0, 1);
    ASSERT(!SNEPPX_topology_is_connected(topo, 0, 1), "0-1 disconnected");
    SNEPPX_topology_destroy(topo);
}

static void test_topology_ring(void) {
    SNEPPXTopology* topo = SNEPPX_topology_create_ring(8);
    ASSERT(topo != NULL, "ring topology");
    ASSERT(SNEPPX_topology_is_connected(topo, 0, 1), "ring 0-1");
    ASSERT(SNEPPX_topology_is_connected(topo, 7, 0), "ring 7-0");
    SNEPPX_topology_destroy(topo);
}

static void test_topology_mesh(void) {
    SNEPPXTopology* topo = SNEPPX_topology_create_mesh(4, 4);
    ASSERT(topo != NULL, "mesh topology 4x4");
    ASSERT(topo->num_nodes == 16, "16 nodes in 4x4 mesh");
    SNEPPX_topology_destroy(topo);
}

int main(void) {
    run_test("topology_create_destroy", test_topology_create_destroy);
    run_test("topology_connect_disconnect", test_topology_connect_disconnect);
    run_test("topology_ring", test_topology_ring);
    run_test("topology_mesh", test_topology_mesh);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
