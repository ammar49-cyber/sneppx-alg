#ifndef SNEPPX_TRANSPORT_SECURITY_H
#define SNEPPX_TRANSPORT_SECURITY_H
/*
 * S4 Network Security — Transport Security Layer
 * TLS 1.3 wrappers, Noise protocol handshake, QUIC session management.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_TLS_MAX_SESSIONS 64
#define SNEPPX_TLS_KEY_LEN 32
#define SNEPPX_TLS_NONCE_LEN 12

typedef struct {
    int session_id;
    int is_active;
    uint8_t session_key[SNEPPX_TLS_KEY_LEN];
    uint64_t creation_time;
    uint64_t last_used;
} SNEPPXTLSSession;

typedef struct {
    int enabled;
    SNEPPXTLSSession sessions[SNEPPX_TLS_MAX_SESSIONS];
    int session_count;
    int use_noise_protocol;
    int use_quic;
} SNEPPXTransportSecurity;

int  SNEPPX_transport_init(SNEPPXTransportSecurity* ts);
void SNEPPX_transport_shutdown(SNEPPXTransportSecurity* ts);
int  SNEPPX_transport_new_session(SNEPPXTransportSecurity* ts, const uint8_t* psk, size_t psk_len);
int  SNEPPX_transport_close_session(SNEPPXTransportSecurity* ts, int session_id);
int  SNEPPX_transport_encrypt(SNEPPXTransportSecurity* ts, int session_id,
                             const uint8_t* plaintext, size_t len,
                             uint8_t* ciphertext, uint8_t nonce[SNEPPX_TLS_NONCE_LEN]);
int  SNEPPX_transport_decrypt(SNEPPXTransportSecurity* ts, int session_id,
                             const uint8_t* ciphertext, size_t len,
                             const uint8_t nonce[SNEPPX_TLS_NONCE_LEN],
                             uint8_t* plaintext);
int  SNEPPX_transport_noise_handshake(SNEPPXTransportSecurity* ts,
                                     const uint8_t* prologue, size_t prologue_len,
                                     uint8_t* handshake_msg, size_t* msg_len);

#ifdef __cplusplus
}
#endif
#endif
