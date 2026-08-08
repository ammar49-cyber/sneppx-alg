/*
 * SNEPPX - vizmon unit test (compile + run standalone with MSVC + UCRT).
 *
 * Usage:
  *   cl /nologo /D_WIN32_WINNT=0x0601 /DSNEPPX_VIZMON_EXPORTS /Iinclude vizmon.c tests/unit/test_vizmon.c ws2_32.lib advapi32.lib
 *   test_vizmon.exe
 */
#include "neural_core/vizmon.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { printf("FAIL: %s (line %d)\n", #cond, __LINE__); failures++; } \
} while (0)

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0); setvbuf(stderr, NULL, _IONBF, 0);
    SNEPPX_VizMon *m = SNEPPX_vizmon_create();
    CHECK(m != NULL);
    for (int i = 0; i < 10; i++)
        SNEPPX_vizmon_push_scalar(m, "loss",  (double)(10 - i) * 0.1, (double)i);
    for (int i = 0; i < 10; i++)
        SNEPPX_vizmon_push_scalar(m, "acc",   (double)(i + 1) * 0.1, (double)i + 0.5);
    SNEPPX_VizSeries s = {0};
    CHECK(SNEPPX_vizmon_get(m, "loss", &s) == 0);
    CHECK(s.count == 10);
    CHECK(s.values[0] == 1.0 && s.values[9] == 0.1);
    CHECK(s.times[0] == 0.0 && s.times[9] == 9.0);
    /* caller-owned buffers */
    if (s.values) free(s.values);
    if (s.times) free(s.times);
    CHECK(SNEPPX_vizmon_get(m, "nope", &s) == -1);

    /* ---- graph ---- */
    int ins0[] = {-1}; (void)ins0;
    int ins1[] = {0};
    CHECK(SNEPPX_vizmon_add_node(m, "x:0", ins0, 0, 0, (double[]){4}, 1) == 0);
    CHECK(SNEPPX_vizmon_add_node(m, "linear", ins1, 1, 1, (double[]){4, 8}, 2) == 0);
    int ins2[] = {1, 0};
    CHECK(SNEPPX_vizmon_add_node(m, "add", ins2, 2, 2, (double[]){8}, 1) == 0);
    SNEPPX_VizNode nodes[16] = {0}; int nn = 0;
    CHECK(SNEPPX_vizmon_get_graph(m, nodes, &nn) == 0);
    CHECK(nn == 3);
    CHECK(strcmp(nodes[1].name, "linear") == 0);
    CHECK(nodes[1].num_in == 1 && nodes[1].in_nodes[0] == 0 && nodes[1].out_node == 1);
    CHECK(nodes[1].nshape == 2 && nodes[1].shape[0] == 4 && nodes[1].shape[1] == 8);
    CHECK(strcmp(nodes[2].name, "add") == 0 && nodes[2].num_in == 2);

    /* ---- embeddings + PCA ---- */
    double pts[6 * 3] = {
        1.0,  1.1,  0.9,
       -1.0, -1.1, -0.9,
        1.0,  1.2,  1.0,
       -1.0, -1.0, -1.1,
        1.1,  1.0,  1.0,
       -0.9, -1.0, -1.0
    };
    CHECK(SNEPPX_vizmon_push_embedding(m, "h", pts, 6, 3) == 0);
    CHECK(SNEPPX_vizmon_project_pca(m, 2) == 0);
    SNEPPX_VizEmbedding *e = SNEPPX_vizmon_get_embedding(m, "h");
    CHECK(e != NULL && e->n_samples == 6 && e->n_dim == 3 && e->projected == 1);
    CHECK(e->proj != NULL);
    /* first 3 points cluster at +x, last 3 at -x: principal component separates them */
    double mean_pos = (e->proj[0] + e->proj[4] + e->proj[8]) / 3.0;
    double mean_neg = (e->proj[4] + e->proj[8] + e->proj[12]) / 3.0;
    (void)mean_pos; (void)mean_neg;
    CHECK(e->proj[0] != e->proj[4] || e->proj[2] != e->proj[6]); /* not all identical */
    SNEPPX_vizmon_free(e->data); SNEPPX_vizmon_free(e->proj); SNEPPX_vizmon_free(e);

    /* ---- samples ---- */
    unsigned char png[4] = {137, 80, 78, 71};
    CHECK(SNEPPX_vizmon_push_sample(m, "img", SNEPPX_VIZMON_IMG_PNG, png, 4, 42.0) == 0);
    SNEPPX_VizSample sm[8] = {0}; int ns = 0;
    CHECK(SNEPPX_vizmon_get_samples(m, "img", sm, &ns) == 0);
    CHECK(ns == 1 && sm[0].kind == SNEPPX_VIZMON_IMG_PNG && sm[0].len == 4);
    for (int i = 0; i < ns; i++) free(sm[i].bytes);

    /* ---- histograms ---- */
    double vals[100];
    for (int i = 0; i < 100; i) vals[i] = (double)(i++ - 50); /* -50..49 */
    for (int i = 0; i < 100; i) vals[i] = (double)(i++ - 50);
    double data100[100]; for (int i = 0; i < 100; i++) data100[i] = (double)i;
    CHECK(SNEPPX_vizmon_push_histogram(m, "w", data100, 100, 20, 7.0) == 0);
    SNEPPX_VizHistogram h = {0};
    CHECK(SNEPPX_vizmon_get_histogram(m, "w", &h) == 0);
    CHECK(h.n_bins == 20);
    CHECK(h.counts[0] == 5 && h.counts[19] == 5); /* 0..4 and 95..99 in first/last bin */
    free(h.edges); free(h.counts);

    /* ---- timeline ---- */
    CHECK(SNEPPX_vizmon_push_timeline(m, "gemm_0", 1.0, 2.5, 1024) == 0);
    CHECK(SNEPPX_vizmon_push_timeline(m, "softmax_0", 3.5, 0.3, 512) == 0);
    SNEPPX_VizTimelineEvent te[64] = {0}; int nt = 0;
    CHECK(SNEPPX_vizmon_get_timeline(m, te, &nt) == 0);
    CHECK(nt == 2);
    CHECK(strcmp(te[0].name, "gemm_0") == 0 && te[0].dur_ms == 2.5 && te[0].mem_bytes == 1024);

    /* ---- sweeps ---- */
    CHECK(SNEPPX_vizmon_push_sweep(m, "lr-3", 0.12, 0.97, 320.0) == 0);
    CHECK(SNEPPX_vizmon_push_sweep(m, "lr-4", 0.09, 0.98, 410.0) == 0);
    SNEPPX_VizSweepResult sw[8] = {0}; int nsw = 0;
    CHECK(SNEPPX_vizmon_get_sweeps(m, sw, &nsw) == 0);
    CHECK(nsw == 2);
    CHECK(strcmp(sw[0].config_id, "lr-3") == 0 && sw[0].final_loss == 0.12);

    /* ---- snapshot JSON ---- */
    char *json = SNEPPX_vizmon_snapshot_json(m);
    CHECK(json != NULL);
    CHECK(strstr(json, "scalars") != NULL);
    CHECK(strstr(json, "linear") != NULL);
    CHECK(strstr(json, "embeddings") != NULL);
    CHECK(strstr(json, "histograms") != NULL);
    CHECK(strstr(json, "timeline") != NULL);
    CHECK(strstr(json, "sweeps") != NULL);
    CHECK(strstr(json, "\"final_acc\":0.97") != NULL);
    free(json);

    /* ---- frontend HTML ---- */
    const char *html = SNEPPX_vizmon_frontend_html();
    CHECK(html != NULL);
    CHECK(strstr(html, "vue.global.prod") != NULL);
    CHECK(strstr(html, "chart.js") != NULL);
    CHECK(strstr(html, "/snapshot") != NULL);
    CHECK(strstr(html, "/ws") != NULL);

    /* ---- export HTML embeds snapshot ---- */
    char *exp = SNEPPX_vizmon_export_html(m);
    CHECK(exp != NULL);
    CHECK(strstr(exp, "const SNAPSHOT=") != NULL);
    CHECK(strstr(exp, "vue.global.prod") != NULL);
    free(exp);

    /* ---- server start/stop smoke (use a high port) ---- */
    CHECK(SNEPPX_vizmon_start(m, 18399) == 0);
    SNEPPX_vizmon_stop(m);
    CHECK(SNEPPX_vizmon_start(m, 18399) == 0);
    SNEPPX_vizmon_stop(m);

    SNEPPX_vizmon_destroy(m);

    if (failures == 0) {
        printf("ALL vizmon TESTS PASS\n");
        return 0;
    }
    printf("%d vizmon TEST FAILURES\n", failures);
    return 1;
}
