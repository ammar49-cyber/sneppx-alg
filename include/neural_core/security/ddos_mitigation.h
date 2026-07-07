#ifndef ARIX_DDOS_MITIGATION_H
#define ARIX_DDOS_MITIGATION_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    int syn_flood_threshold;
    int conn_rate_threshold;
    int bandwidth_threshold;
    int block_duration;
    int enable_adaptive;
    int enable_app_layer;
    int enable_syn_cookies;
} ddos_config_t;

typedef struct {
    uint64_t total_packets_dropped;
    uint64_t total_blocks_issued;
    uint64_t total_sources_blocked;
    int active_flows;
    int active_sources;
    int syn_flood_threshold;
    int conn_rate_threshold;
    int block_duration;
} ddos_stats_t;

int arix_ddos_detect_syn_flood(uint32_t src_ip, uint32_t dst_ip);
int arix_ddos_detect_app_layer(uint32_t src_ip, const uint8_t *payload, size_t len);
uint32_t arix_ddos_generate_syn_cookie(uint32_t src_ip, uint32_t dst_ip, uint16_t src_port);
int arix_ddos_validate_syn_cookie(uint32_t cookie, uint32_t src_ip, uint32_t dst_ip, uint16_t src_port);
int arix_ddos_apply_rate_limit(uint32_t src_ip);
int arix_ddos_track_connection(uint32_t src_ip, uint32_t dst_ip, uint16_t src_port, uint16_t dst_port, uint64_t bytes);
int arix_ddos_get_stats(ddos_stats_t *stats);
int arix_ddos_update_config(const ddos_config_t *config);
int arix_ddos_reset(void);

#endif
