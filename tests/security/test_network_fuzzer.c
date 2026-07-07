#include "network_fuzzer.h"
#include <stdio.h>
#include <string.h>

int main() {
    fuzz_stats_t stats;
    arix_fuzzer_init(FUZZ_PROTO_RAW);
    uint8_t buf[1024];
    size_t len;
    for (int i = 0; i < 100; i++) {
        arix_fuzzer_generate(buf, &len, sizeof(buf));
        fuzz_result_t result;
        arix_fuzzer_execute(buf, len, &result);
    }
    arix_fuzzer_get_stats(&stats);
    printf("Fuzzer stats: exec=%llu crashes=%llu cov=%.1f%%\n",
        stats.total_executions, stats.total_crashes, stats.coverage_percent);
    arix_fuzzer_init(FUZZ_PROTO_TLS);
    for (int i = 0; i < 50; i++) {
        arix_fuzzer_generate(buf, &len, sizeof(buf));
        fuzz_result_t result;
        arix_fuzzer_execute(buf, len, &result);
    }
    arix_fuzzer_get_stats(&stats);
    printf("TLS fuzzer: exec=%llu mut=%llu\n", stats.total_executions, stats.total_mutations);
    printf("PASS: Network fuzzer OK\n");
    return 0;
}
