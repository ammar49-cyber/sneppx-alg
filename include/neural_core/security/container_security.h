#ifndef ARIX_CONTAINER_SECURITY_H
#define ARIX_CONTAINER_SECURITY_H

#include <stdint.h>
#include <stddef.h>
#include <time.h>

typedef struct {
    char media_type[64];
    int schema_version;
    int num_layers;
    char digest[72];
    uint64_t size;
    uint8_t config_hash[32];
} container_manifest_t;

typedef struct {
    char name[128];
    char version[64];
    char type[32];
    char supplier[128];
    uint8_t hash[32];
    uint64_t size;
    int has_checksum;
} sbom_component_t;

typedef struct {
    char image_name[128];
    char version[32];
    char format[32];
    char namespace_str[128];
    char created[32];
    int num_components;
    sbom_component_t components[4096];
    int verified;
} sbom_doc_t;

typedef struct {
    char cve_id[32];
    char severity[16];
    double cvss_score;
    int fixed;
    char fix_version[64];
    char package[128];
} vuln_result_t;

typedef struct {
    int num_images;
    int num_vulns_in_db;
    int total_vulns_found;
    int critical_count;
    int high_count;
    int medium_count;
    int low_count;
} container_stats_t;

int arix_container_verify_image(const uint8_t *manifest, size_t manifest_len, const uint8_t *signature, size_t sig_len, const uint8_t *pubkey);
int arix_container_parse_manifest(const uint8_t *manifest, size_t len, container_manifest_t *out);
int arix_container_verify_layer(const uint8_t *layer_data, size_t layer_len, const char *expected_digest);
int arix_sbom_generate(sbom_doc_t *doc, const char *image_name, const char *version);
int arix_sbom_add_component(sbom_doc_t *doc, const char *name, const char *version, const char *type, const char *supplier);
int arix_sbom_validate(sbom_doc_t *doc);
int arix_sbom_export_json(sbom_doc_t *doc, char *out, size_t out_len);
int arix_container_scan_vulns(const char *image_name, const uint8_t *layer_data, size_t layer_len, vuln_result_t *results, int max_results);
int arix_container_add_vuln(const char *cve_id, const char *package, const char *severity, double cvss);
int arix_container_get_stats(container_stats_t *stats);
int arix_container_init_vuln_db(void);

#endif
