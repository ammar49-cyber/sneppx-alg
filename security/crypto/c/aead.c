#include "arix_aead.h"
#include "arix_chacha20.h"
#include "arix_poly1305.h"
#include "arix_ct.h"
#include <string.h>
#include <stdlib.h>

static void poly1305_key_gen(const uint8_t key[32], const uint8_t nonce[12], uint8_t poly_key[32]) {
    ArixChaCha20State state;
    uint8_t block[64];
    arix_chacha20_init(&state, key, nonce, 0);
    arix_chacha20_block(&state, block);
    memcpy(poly_key, block, 32);
}

int arix_aead_encrypt(uint8_t* ciphertext, uint8_t tag[16], const uint8_t* plaintext, size_t len,
                      const uint8_t* aad, size_t aad_len, const uint8_t key[32], const uint8_t nonce[12]) {
    if (!ciphertext || !tag || !plaintext || !key || !nonce) return -1;
    uint8_t poly_key[32];
    poly1305_key_gen(key, nonce, poly_key);
    ArixChaCha20State state;
    arix_chacha20_init(&state, key, nonce, 1);
    if (ciphertext != plaintext) memcpy(ciphertext, plaintext, len);
    arix_chacha20_encrypt(&state, ciphertext, len);
    ArixPoly1305State mac;
    arix_poly1305_init(&mac, poly_key);
    if (aad && aad_len) arix_poly1305_update(&mac, aad, aad_len);
    size_t pad_aad = aad_len ? (16 - aad_len % 16) % 16 : 0;
    uint8_t zeros[16] = {0};
    if (pad_aad) arix_poly1305_update(&mac, zeros, pad_aad);
    if (len) arix_poly1305_update(&mac, ciphertext, len);
    size_t pad_ct = (16 - len % 16) % 16;
    if (pad_ct) arix_poly1305_update(&mac, zeros, pad_ct);
    uint8_t len_block[16];
    for (int i = 0; i < 8; i++) {
        len_block[i] = (uint8_t)(aad_len >> (i * 8));
        len_block[i + 8] = (uint8_t)(len >> (i * 8));
    }
    arix_poly1305_update(&mac, len_block, 16);
    arix_poly1305_finish(&mac, tag);
    return 0;
}

int arix_aead_decrypt(uint8_t* plaintext, const uint8_t* ciphertext, size_t len,
                      const uint8_t tag[16], const uint8_t* aad, size_t aad_len,
                      const uint8_t key[32], const uint8_t nonce[12]) {
    if (!plaintext || !ciphertext || !tag || !key || !nonce) return -1;
    uint8_t poly_key[32];
    uint8_t block[64];
    {   ArixChaCha20State ks;
        arix_chacha20_init(&ks, key, nonce, 0);
        arix_chacha20_block(&ks, block);
    }
    memcpy(poly_key, block, 32);
    ArixPoly1305State mac;
    arix_poly1305_init(&mac, poly_key);
    if (aad && aad_len) arix_poly1305_update(&mac, aad, aad_len);
    size_t pad_aad = aad_len ? (16 - aad_len % 16) % 16 : 0;
    uint8_t zeros[16] = {0};
    if (pad_aad) arix_poly1305_update(&mac, zeros, pad_aad);
    if (len) arix_poly1305_update(&mac, ciphertext, len);
    size_t pad_ct = (16 - len % 16) % 16;
    if (pad_ct) arix_poly1305_update(&mac, zeros, pad_ct);
    uint8_t len_block[16];
    for (int i = 0; i < 8; i++) {
        len_block[i] = (uint8_t)(aad_len >> (i * 8));
        len_block[i + 8] = (uint8_t)(len >> (i * 8));
    }
    arix_poly1305_update(&mac, len_block, 16);
    uint8_t expected_tag[16];
    arix_poly1305_finish(&mac, expected_tag);
    int ok = arix_ct_equal(tag, expected_tag, 16);
    if (ok) {
        ArixChaCha20State state;
        arix_chacha20_init(&state, key, nonce, 1);
        if (plaintext != ciphertext) memcpy(plaintext, ciphertext, len);
        arix_chacha20_encrypt(&state, plaintext, len);
    }
    return ok ? 0 : -1;
}
