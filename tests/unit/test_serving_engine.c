#include "serving_engine.h"
#include "http_server.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/*
 * SNEPPX - Test Serving Engine
 *
 * WHAT
 *   Tests the model serving control plane without binding a socket.
 *
 * CONCEPT
 *   Exercises versioning, rolling updates, A/B routing, dynamic batching,
 *   metrics, health/readiness, warm-up, and config hot-reload directly.
 */

static int pass = 0, fail = 0;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s (line %d)\n", m, __LINE__); fail++; } else { pass++; } } while (0)

int main(void) {
    /* ---- lifecycle + config ---- */
    SNEPPX_ServingEngine* e = SNEPPX_serving_engine_create();
    CHECK(e != NULL, "engine create");
    SNEPPX_serving_engine_configure(e, 4, 2, 2, 3);
    CHECK(SNEPPX_serving_max_batch_size(e) == 4, "config max_batch_size");
    CHECK(SNEPPX_serving_worker_count(e) == 2, "config n_workers");
    CHECK(SNEPPX_serving_warmup_iters(e) == 3, "config warmup_iters");
    CHECK(SNEPPX_serving_is_ready(e) == 0, "not ready by default");

    /* ---- register + rolling update ---- */
    CHECK(SNEPPX_serving_register_model(e, "gpt", "v1", "first release") == 0, "register v1");
    CHECK(SNEPPX_serving_register_model(e, "gpt", "v2", "canary") == 0, "register v2 canary");
    CHECK(SNEPPX_serving_active_version(e, "gpt") != NULL, "active version exists");
    CHECK(strcmp(SNEPPX_serving_active_version(e, "gpt"), "v1") == 0, "active is v1");
    CHECK(SNEPPX_serving_model_count(e) == 1, "one model registered");

    SNEPPX_ServingModel* m = SNEPPX_serving_get_model(e, "gpt");
    CHECK(m != NULL, "get model");
    CHECK(m->num_versions == 2, "two versions");
    CHECK(m->versions[m->active].weight == 100, "active has full weight");
    CHECK(m->versions[1].weight == 0, "canary weight is 0");

    /* rolling deploy: shift 30% to v2 */
    CHECK(SNEPPX_serving_set_weight(e, "gpt", "v2", 30) == 0, "shift weight to v2");
    CHECK(SNEPPX_serving_set_weight(e, "gpt", "v1", 70) == 0, "shift weight to v1");
    {
        int n_v1 = 0, n_v2 = 0;
        char ver[64];
        for (long long i = 0; i < 10000; i++) {
            char rid[32];
            snprintf(rid, sizeof(rid), "req-%lld", i);
            CHECK(SNEPPX_serving_route(e, "gpt", rid, ver) == 0, "route");
            if (strcmp(ver, "v1") == 0) n_v1++;
            else if (strcmp(ver, "v2") == 0) n_v2++;
            else { CHECK(0, "unknown version returned"); break; }
        }
        CHECK(n_v1 + n_v2 == 10000, "all routed");
        CHECK(n_v1 > 6500 && n_v1 < 7500, "v1 share ~70%");
        CHECK(n_v2 > 2500 && n_v2 < 3500, "v2 share ~30%");
    }

    /* promote v2 (rolling complete) */
    CHECK(SNEPPX_serving_set_active_version(e, "gpt", "v2") == 0, "promote v2");
    CHECK(strcmp(SNEPPX_serving_active_version(e, "gpt"), "v2") == 0, "active now v2");
    CHECK(m->versions[m->active].weight == 100, "promoted weight 100");
    {
        char ver[64];
        int routed_v2 = 0;
        for (int i = 0; i < 1000; i++) {
            char rid[32];
            snprintf(rid, sizeof(rid), "post-%d", i);
            SNEPPX_serving_route(e, "gpt", rid, ver);
            if (strcmp(ver, "v2") == 0) routed_v2++;
        }
        CHECK(routed_v2 == 1000, "100% traffic to v2 after promote");
    }

    /* rollback drops canary */
    CHECK(SNEPPX_serving_rollback(e, "gpt") == 0, "rollback");
    CHECK(m->num_versions == 1, "rollback left one version");
    CHECK(strcmp(SNEPPX_serving_active_version(e, "gpt"), "v1") == 0, "active restored to v1");
    {
        char ver[64];
        CHECK(SNEPPX_serving_route(e, "gpt", "anything", ver) == 0, "route after rollback");
        CHECK(strcmp(ver, "v1") == 0, "rollback routes to v1");
    }

    /* ---- dynamic batching ---- */
    SNEPPX_ServingEngine* b = SNEPPX_serving_engine_create();
    SNEPPX_serving_engine_configure(b, 3, 10, 1, 0);
    CHECK(SNEPPX_serving_is_ready(b) == 0, "b not ready by default");
    CHECK(SNEPPX_serving_batch_submit(b, "r1") == 1, "submit r1");
    CHECK(SNEPPX_serving_batch_submit(b, "r2") == 2, "submit r2");
    {
        char ids[8][SNEPPX_SERVING_MAX_ID_LEN] = {0};
        int count = 0;
        CHECK(SNEPPX_serving_batch_drain(b, ids, 8, &count) == 0, "not drained on size");
        CHECK(count == 0, "count 0 not ready");
        CHECK(SNEPPX_serving_batch_submit(b, "r3") == 3, "submit r3 (full)");
        CHECK(SNEPPX_serving_batch_drain(b, ids, 8, &count) == 1, "drained on size");
        CHECK(count == 3, "drained 3");
        CHECK(strcmp(ids[0], "r1") == 0, "id order r1");
        CHECK(strcmp(ids[2], "r3") == 0, "id order r3");
        SNEPPX_ServingCounts cc = SNEPPX_serving_counts(b);
        CHECK(cc.batches == 1, "one batch recorded");
    }

    /* ---- metrics + warm-up ---- */
    SNEPPX_serving_engine_configure(b, 8, 5, 1, 4);
    SNEPPX_serving_warmup_start(b);
    int became_ready = 0;
    for (int i = 0; i < 4; i++) {
        /* warmup_tick records the synthetic request internally */
        if (SNEPPX_serving_warmup_tick(b, 100 + i * 10, 8)) became_ready = 1;
    }
    CHECK(became_ready == 1, "warm-up made engine ready");
    CHECK(SNEPPX_serving_is_ready(b) == 1, "warm engine ready");
    SNEPPX_serving_record_request(b, 5000, 16, 0);
    SNEPPX_serving_record_request(b, 9000, 16, 1); /* one error */
    SNEPPX_ServingCounts c = SNEPPX_serving_counts(b);
    CHECK(c.requests == 6 && c.errors == 1 && c.tokens == 64, "metrics counts");
    CHECK(SNEPPX_serving_latency_p50(b) > 0.0, "p50 reported");
    CHECK(SNEPPX_serving_latency_p99(b) >= SNEPPX_serving_latency_p50(b), "p99 >= p50");
    {
        double tp = SNEPPX_serving_throughput_rps(b);
        CHECK(tp > 0.0, "throughput reported");
    }
    {
        char buf[65536];
        size_t n = SNEPPX_serving_metrics_prometheus(b, buf, sizeof(buf));
        CHECK(n > 0, "prometheus metrics rendered");
        CHECK(strstr(buf, "sneppx_serving_requests_total") != NULL, "prom has requests");
        size_t j = SNEPPX_serving_metrics_json(b, buf, sizeof(buf));
        CHECK(j > 0 && strstr(buf, "uptime_ms") != NULL, "json metrics rendered");
    }
    SNEPPX_serving_set_ready(b, 0);
    CHECK(SNEPPX_serving_is_ready(b) == 0, "manually set not ready");

    /* unknown model routing */
    {
        char ver[64];
        CHECK(SNEPPX_serving_route(e, "nope", "x", ver) == -1, "route unknown model fails");
        CHECK(SNEPPX_serving_set_weight(e, "nope", "v1", 50) == -1, "set_weight unknown fails");
    }

    /* ---- config hot-reload ---- */
    {
        char tmppath[] = "tmp_serving_test.yaml";
        FILE* f = fopen(tmppath, "w");
        CHECK(f != NULL, "create tmp yaml");
        if (f) {
            fputs("serving.max_batch_size: 16\n", f);
            fputs("serving.batch_timeout_ms: 20\n", f);
            fputs("serving.n_workers: 4\n", f);
            fputs("serving.warmup_iters: 10\n", f);
            fputs("model.gpt.canary: 50\n", f);
            fclose(f);
        }
        SNEPPX_serving_engine_configure(b, 8, 5, 1, 0);
        SNEPPX_serving_set_config_path(b, tmppath);
        int reloaded = SNEPPX_serving_config_reload(b);
        CHECK(reloaded == 1, "config reloaded");
        CHECK(SNEPPX_serving_max_batch_size(b) == 16, "yaml max_batch_size applied");
        CHECK(SNEPPX_serving_batch_timeout_ms(b) == 20, "yaml batch_timeout_ms applied");
        CHECK(SNEPPX_serving_worker_count(b) == 4, "yaml n_workers applied");
        CHECK(SNEPPX_serving_warmup_iters(b) == 10, "yaml warmup_iters applied");
        CHECK(SNEPPX_serving_config_reload(b) == 0, "no reload when unchanged");
        remove(tmppath);
    }

    SNEPPX_serving_engine_destroy(b);
    SNEPPX_serving_engine_destroy(e);

    printf("\n%d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}
