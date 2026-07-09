#ifndef SNEPPX_S4_EXTENSIONS_H
#define SNEPPX_S4_EXTENSIONS_H
/* S4 Network Security extensions: TLS 1.3 full handshake, Noise NK/XX/IK, QUIC,
   mTLS, OCSP stapling, CT, DoH, WireGuard, IP blocklist, NIDS, traffic analysis,
   rate limiting, port knocking, gRPC auth */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_TLS13_MAX_EXTENSIONS 16
#define SNEPPX_NOISE_MAX_PATTERNS 8
#define SNEPPX_IP_BLOCKLIST_MAX 256

/* TLS 1.3 full handshake */
typedef struct {
    uint8_t random[32];
    uint8_t session_id[32];
    int session_id_len;
    uint16_t cipher_suites[16];
    int cipher_count;
    uint16_t supported_groups[8];
    int group_count;
    uint8_t* extensions[SNEPPX_TLS13_MAX_EXTENSIONS];
    size_t ext_lens[SNEPPX_TLS13_MAX_EXTENSIONS];
    int ext_count;
} SNEPPXTLS13ClientHello;

typedef struct {
    int handshake_complete;
    uint8_t master_secret[48];
    uint8_t server_random[32];
} SNEPPXTLS13Session;

int  SNEPPX_tls13_client_hello_init(SNEPPXTLS13ClientHello* ch);
int  SNEPPX_tls13_server_hello_parse(SNEPPXTLS13Session* sess, const uint8_t* data, size_t len);
int  SNEPPX_tls13_derive_keys(SNEPPXTLS13Session* sess, const uint8_t* psk, size_t psk_len);

/* Noise protocol patterns NK, XX, IK */
typedef struct {
    int pattern; /* 0=NK, 1=XX, 2=IK */
    uint8_t s[32], e[32], rs[32], re[32];
    int initiator;
    int step;
} SNEPPXNoiseHandshake;

int  SNEPPX_noise_init(SNEPPXNoiseHandshake* nh, int pattern, int initiator);
int  SNEPPX_noise_write_msg(SNEPPXNoiseHandshake* nh, uint8_t* msg, size_t* msg_len);
int  SNEPPX_noise_read_msg(SNEPPXNoiseHandshake* nh, const uint8_t* msg, size_t msg_len);

/* QUIC connection manager */
typedef struct {
    int connection_id;
    uint8_t* stream_buffers[16];
    size_t stream_sizes[16];
    int stream_count;
    int established;
} SNEPPXQUICConn;

int  SNEPPX_quic_conn_init(SNEPPXQUICConn* qc);
int  SNEPPX_quic_conn_handshake(SNEPPXQUICConn* qc, const uint8_t* params, size_t params_len);
int  SNEPPX_quic_stream_send(SNEPPXQUICConn* qc, int stream_id, const uint8_t* data, size_t len);
int  SNEPPX_quic_stream_recv(SNEPPXQUICConn* qc, int stream_id, uint8_t* data, size_t* len);

/* mTLS */
int  SNEPPX_mtls_authenticate(const uint8_t* cert_der, size_t cert_len, const uint8_t* key_der, size_t key_len);

/* OCSP stapling */
int  SNEPPX_ocsp_request(const uint8_t* issuer_cert, size_t issuer_len, const uint8_t* cert, size_t cert_len, uint8_t* response, size_t* resp_len);
int  SNEPPX_ocsp_verify(const uint8_t* response, size_t resp_len);

/* Certificate Transparency */
int  SNEPPX_ct_verify_sct(const uint8_t* sct, size_t sct_len, const uint8_t* cert, size_t cert_len);

/* DNS over HTTPS */
int  SNEPPX_doh_resolve(const char* hostname, uint8_t* ip_out, size_t* ip_len);

/* WireGuard */
typedef struct {
    uint8_t private_key[32];
    uint8_t public_key[32];
    uint8_t preshared_key[32];
    int established;
} SNEPPXWireGuardSession;

int  SNEPPX_wireguard_init(SNEPPXWireGuardSession* wg);
int  SNEPPX_wireguard_handshake(SNEPPXWireGuardSession* wg, const uint8_t* peer_key, size_t key_len);

/* IP blocklist */
typedef struct {
    uint32_t networks[SNEPPX_IP_BLOCKLIST_MAX];
    uint32_t masks[SNEPPX_IP_BLOCKLIST_MAX];
    int count;
} SNEPPXIPBlocklist;

int  SNEPPX_ip_blocklist_init(SNEPPXIPBlocklist* bl);
int  SNEPPX_ip_blocklist_add(SNEPPXIPBlocklist* bl, const char* cidr);
int  SNEPPX_ip_blocklist_check(SNEPPXIPBlocklist* bl, uint32_t ip);

/* NIDS */
int  SNEPPX_nids_init(void);
int  SNEPPX_nids_analyze_packet(const uint8_t* packet, size_t len);

/* Traffic analysis mitigation */
int  SNEPPX_traffic_pad(uint8_t* data, size_t* len, size_t max_len, size_t block_size);

/* Connection rate limiting */
typedef struct {
    uint32_t connection_counts[256];
    uint64_t windows[256];
    int max_per_window;
} SNEPPXRateLimiter;

int  SNEPPX_rate_limiter_init(SNEPPXRateLimiter* rl, int max_per_window);
int  SNEPPX_rate_limiter_check(SNEPPXRateLimiter* rl, uint32_t src_ip);

/* Port knocking */
int  SNEPPX_port_knock_sequence(const uint16_t* ports, int port_count);
int  SNEPPX_port_knock_verify(const uint16_t* received, int count, const uint16_t* expected, int expected_count);

/* gRPC auth interceptor */
typedef struct {
    uint8_t token[64];
    size_t token_len;
    int authenticated;
} SNEPPXGRPCAuth;

int  SNEPPX_grpc_auth_init(SNEPPXGRPCAuth* ga, const uint8_t* token, size_t token_len);
int  SNEPPX_grpc_auth_verify(SNEPPXGRPCAuth* ga, const uint8_t* received_token, size_t token_len);

#ifdef __cplusplus
}
#endif
#endif
