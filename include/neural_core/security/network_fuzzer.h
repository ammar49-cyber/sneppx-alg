#ifndef ARIX_NETWORK_FUZZER_H
#define ARIX_NETWORK_FUZZER_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    FUZZ_PROTO_RAW = 0,
    FUZZ_PROTO_TLS,
    FUZZ_PROTO_HTTP,
    FUZZ_PROTO_QUIC,
    FUZZ_PROTO_DNS,
    FUZZ_PROTO_SSH
} fuzz_protocol_t;

typedef enum {
    FUZZ_CRASH_NONE = 0,
    FUZZ_CRASH_SEGFAULT,
    FUZZ_CRASH_ABORT,
    FUZZ_CRASH_TIMEOUT,
    FUZZ_CRASH_OOM,
    FUZZ_CRASH_ASSERTION
} fuzz_crash_type_t;

typedef struct {
    size_t max_input_size;
    int max_mutations;
    int timeout_ms;
    int enable_coverage;
    int enable_crash_detection;
    double mutate_prob_flip;
    double mutate_prob_erase;
    double mutate_prob_insert;
    double mutate_prob_splice;
    double mutate_prob_havoc;
} fuzz_config_t;

typedef struct {
    uint32_t input_crc;
    uint64_t execution_id;
    int crashed;
    fuzz_crash_type_t crash_type;
    int new_coverage;
    uint64_t coverage_edges;
} fuzz_result_t;

typedef struct {
    uint64_t total_executions;
    uint64_t total_crashes;
    uint64_t total_mutations;
    int corpus_size;
    int coverage_edges;
    int coverage_total;
    double coverage_percent;
    double crashes_per_k;
} fuzz_stats_t;

int arix_fuzzer_init(fuzz_protocol_t protocol);
int arix_fuzzer_add_seed(const uint8_t *data, size_t len);
int arix_fuzzer_generate(uint8_t *buf, size_t *len, size_t max_len);
int arix_fuzzer_execute(const uint8_t *input, size_t len, fuzz_result_t *result);
int arix_fuzzer_get_stats(fuzz_stats_t *stats);
int arix_fuzzer_set_config(const fuzz_config_t *config);
int arix_fuzzer_reset(void);
int arix_fuzzer_fuzz_target(const uint8_t *target_data, size_t target_len, int iterations, fuzz_result_t *results, int max_results);

#endif
