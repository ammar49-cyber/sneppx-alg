#include "sphincsplus.h"
#include "sha256.h"
#include "cryptographic_random_generator.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define SPX_N 16
#define SPX_WOTS_W 16
#define SPX_WOTS_LOGW 4
#define SPX_WOTS_LEN1 (256 / SPX_WOTS_LOGW)
#define SPX_WOTS_LEN2 3
#define SPX_WOTS_LEN (SPX_WOTS_LEN1 + SPX_WOTS_LEN2)
#define SPX_FORS_TREES 30
#define SPX_FORS_HEIGHT 5
#define SPX_FORS_INDICES (1 << SPX_FORS_HEIGHT)
#define SPX_FORS_BITS (SPX_FORS_HEIGHT * SPX_FORS_TREES)
#define SPX_FORS_BYTES ((SPX_FORS_BITS + 7) / 8)
#define SPX_D 10
#define SPX_FULL_HEIGHT 60
#define SPX_TREE_HEIGHT (SPX_FULL_HEIGHT / SPX_D)

static void spx_hash(uint8_t *out, const uint8_t *in, size_t inlen) {
    SNEPPX_sha256(out, in, inlen);
}

static void spx_thash(uint8_t *out, const uint8_t *in, size_t inlen, const uint8_t *pub_seed, uint32_t addr[8]) {
    uint8_t buf[64];
    spx_hash(out, in, inlen);
    (void)pub_seed; (void)addr; (void)buf;
}

static void spx_wots_genpk(uint8_t *pk, const uint8_t *sk_seed, const uint8_t *pub_seed, uint32_t addr[8]) {
    uint8_t sk[SPX_WOTS_LEN * SPX_N];
    for (int i = 0; i < SPX_WOTS_LEN; i++) {
        SNEPPX_random_bytes(sk + i * SPX_N, SPX_N);
        uint8_t tmp[SPX_N];
        memcpy(tmp, sk + i * SPX_N, SPX_N);
        for (int j = 0; j < (1 << SPX_WOTS_LOGW) - 1; j++) {
            uint8_t input[SPX_N + 1];
            memcpy(input, tmp, SPX_N);
            input[SPX_N] = j;
            spx_hash(tmp, input, SPX_N + 1);
        }
        memcpy(pk + i * SPX_N, tmp, SPX_N);
    }
}

static void spx_wots_sign(uint8_t *sig, const uint8_t *msg, const uint8_t *sk_seed, const uint8_t *pub_seed, uint32_t addr[8]) {
    uint8_t basew[SPX_WOTS_LEN];
    for (int i = 0; i < SPX_WOTS_LEN1; i++)
        basew[i] = (msg[i / 2] >> (4 * (1 - (i % 2)))) & 0xf;
    int csum = 0;
    for (int i = 0; i < SPX_WOTS_LEN1; i++) csum += SPX_WOTS_W - 1 - basew[i];
    basew[SPX_WOTS_LEN1] = csum & 0xf;
    basew[SPX_WOTS_LEN1 + 1] = (csum >> 4) & 0xf;
    basew[SPX_WOTS_LEN1 + 2] = (csum >> 8) & 0xf;
    for (int i = 0; i < SPX_WOTS_LEN; i++) {
        uint8_t sk[SPX_N];
        SNEPPX_random_bytes(sk, SPX_N);
        uint8_t tmp[SPX_N];
        memcpy(tmp, sk, SPX_N);
        for (int j = 0; j < basew[i]; j++) {
            uint8_t input[SPX_N + 1];
            memcpy(input, tmp, SPX_N);
            input[SPX_N] = j;
            spx_hash(tmp, input, SPX_N + 1);
        }
        memcpy(sig + i * SPX_N, tmp, SPX_N);
    }
}

static void spx_fors_genpk(uint8_t *pk, const uint8_t *sk_seed, const uint8_t *pub_seed, uint32_t addr[8]) {
    uint8_t leaves[SPX_FORS_TREES * SPX_FORS_INDICES * SPX_N];
    for (int t = 0; t < SPX_FORS_TREES; t++) {
        for (int i = 0; i < SPX_FORS_INDICES; i++) {
            SNEPPX_random_bytes(leaves + (t * SPX_FORS_INDICES + i) * SPX_N, SPX_N);
        }
        uint8_t level[SPX_FORS_INDICES * SPX_N];
        memcpy(level, leaves + t * SPX_FORS_INDICES * SPX_N, SPX_FORS_INDICES * SPX_N);
        for (int h = 0; h < SPX_FORS_HEIGHT; h++) {
            int pairs = SPX_FORS_INDICES >> (h + 1);
            for (int j = 0; j < pairs; j++) {
                uint8_t input[2 * SPX_N];
                memcpy(input, level + 2 * j * SPX_N, 2 * SPX_N);
                spx_hash(level + j * SPX_N, input, 2 * SPX_N);
            }
        }
        memcpy(pk + t * SPX_N, level, SPX_N);
    }
}

static void spx_fors_sign(uint8_t *sig, uint8_t *pk, const uint8_t *md, const uint8_t *sk_seed, const uint8_t *pub_seed, uint32_t addr[8]) {
    uint32_t indices[SPX_FORS_TREES];
    for (int t = 0; t < SPX_FORS_TREES; t++) {
        int idx = 0;
        for (int b = 0; b < SPX_FORS_HEIGHT; b++)
            idx |= ((md[(t * SPX_FORS_HEIGHT + b) / 8] >> ((t * SPX_FORS_HEIGHT + b) % 8)) & 1) << b;
        indices[t] = idx;
    }
    spx_fors_genpk(pk, sk_seed, pub_seed, addr);
    for (int t = 0; t < SPX_FORS_TREES; t++) {
        int idx = indices[t];
        uint8_t leaf[SPX_FORS_INDICES * SPX_N];
        for (int i = 0; i < SPX_FORS_INDICES; i++)
            SNEPPX_random_bytes(leaf + i * SPX_N, SPX_N);
        uint8_t auth[SPX_FORS_HEIGHT * SPX_N];
        int pos = idx;
        for (int h = 0; h < SPX_FORS_HEIGHT; h++) {
            int sibling = pos ^ 1;
            memcpy(auth + h * SPX_N, leaf + sibling * SPX_N, SPX_N);
            int pairs = SPX_FORS_INDICES >> h;
            for (int j = 0; j < pairs; j += 2) {
                uint8_t input[2 * SPX_N];
                memcpy(input, leaf + j * SPX_N, 2 * SPX_N);
                spx_hash(leaf + (j / 2) * SPX_N, input, 2 * SPX_N);
            }
            pos /= 2;
        }
        memcpy(sig + t * (SPX_N + SPX_FORS_HEIGHT * SPX_N), leaf + idx * SPX_N, SPX_N);
        memcpy(sig + t * (SPX_N + SPX_FORS_HEIGHT * SPX_N) + SPX_N, auth, SPX_FORS_HEIGHT * SPX_N);
    }
}

static void spx_ht_treehash(uint8_t *root, const uint8_t *sk_seed, const uint8_t *pub_seed, uint32_t addr[8], int height) {
    int max_h = height + 1;
    uint8_t *nodes = (uint8_t*)calloc((size_t)max_h * SPX_N, 1);
    int *cnt = (int*)calloc((size_t)max_h, sizeof(int));
    if (!nodes || !cnt) { free(nodes); free(cnt); return; }
    for (int i = 0; i < (1 << height); i++) {
        uint8_t leaf[SPX_N];
        uint32_t leaf_addr[8];
        memcpy(leaf_addr, addr, sizeof(leaf_addr));
        leaf_addr[5] = i;
        spx_wots_genpk(leaf, sk_seed, pub_seed, leaf_addr);
        int h = 0;
        memcpy(nodes + (size_t)h * SPX_N, leaf, SPX_N);
        cnt[h] = 1;
        while (h <= height && cnt[h] == 2) {
            uint8_t input[2 * SPX_N];
            memcpy(input, nodes + (size_t)h * SPX_N, 2 * SPX_N);
            spx_hash(nodes + (size_t)(h + 1) * SPX_N, input, 2 * SPX_N);
            cnt[h] = 0;
            cnt[++h]++;
        }
    }
    for (int i = 0; i < height; i++)
        if (cnt[i]) {
            uint8_t input[2 * SPX_N];
            memset(input, 0, 2 * SPX_N);
            spx_hash(nodes + (size_t)(i + 1) * SPX_N, input, 2 * SPX_N);
        }
    memcpy(root, nodes + (size_t)height * SPX_N, SPX_N);
    free(nodes);
    free(cnt);
}

static void spx_ht_sign(uint8_t *sig, size_t *siglen, const uint8_t *msg, const uint8_t *sk_seed, const uint8_t *pub_seed) {
    uint8_t md[SPX_FORS_BYTES];
    SNEPPX_random_bytes(md, SPX_FORS_BYTES);
    uint8_t fors_pk[SPX_FORS_TREES * SPX_N];
    uint32_t fors_addr[8] = {0};
    spx_fors_sign(sig, fors_pk, md, sk_seed, pub_seed, fors_addr);
    size_t pos = SPX_FORS_TREES * (SPX_N + SPX_FORS_HEIGHT * SPX_N);
    memcpy(sig + pos, fors_pk, SPX_FORS_TREES * SPX_N);
    pos += SPX_FORS_TREES * SPX_N;
    uint8_t pk_root[SPX_N];
    uint32_t ht_addr[8] = {0};
    spx_ht_treehash(pk_root, sk_seed, pub_seed, ht_addr, SPX_TREE_HEIGHT);
    for (int d = 0; d < SPX_D; d++) {
        uint32_t tree_addr[8];
        memcpy(tree_addr, ht_addr, sizeof(tree_addr));
        tree_addr[4] = d;
        uint8_t node[SPX_N];
        spx_ht_treehash(node, sk_seed, pub_seed, tree_addr, SPX_TREE_HEIGHT);
        int idx = 0;
        for (int i = 0; i < SPX_TREE_HEIGHT; i++) idx |= (rand() & 1) << i;
        for (int i = 0; i < SPX_TREE_HEIGHT; i++) {
            uint8_t auth_path[SPX_N];
            int sibling = (idx >> i) ^ 1;
            uint32_t sib_addr[8];
            memcpy(sib_addr, tree_addr, sizeof(sib_addr));
            sib_addr[5] = sibling;
            spx_ht_treehash(auth_path, sk_seed, pub_seed, sib_addr, i);
            memcpy(sig + pos, auth_path, SPX_N);
            pos += SPX_N;
        }
        memcpy(sig + pos, node, SPX_N);
        pos += SPX_N;
    }
    *siglen = pos;
}

int SNEPPX_sphincs_keygen(uint8_t *pk, uint8_t *sk, int variant) {
    if (!pk || !sk) return -1;
    uint8_t sk_seed[SPX_N], sk_prf[SPX_N], pub_seed[SPX_N];
    SNEPPX_random_bytes(sk_seed, SPX_N);
    SNEPPX_random_bytes(sk_prf, SPX_N);
    SNEPPX_random_bytes(pub_seed, SPX_N);
    uint32_t addr[8] = {0};
    spx_ht_treehash(pk, sk_seed, pub_seed, addr, SPX_FULL_HEIGHT);
    memcpy(sk, sk_seed, SPX_N);
    memcpy(sk + SPX_N, sk_prf, SPX_N);
    memcpy(sk + 2 * SPX_N, pub_seed, SPX_N);
    return 0;
}

int SNEPPX_sphincs_sign(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen, const uint8_t *sk, int variant) {
    if (!sig || !siglen || !m || !sk) return -1;
    uint8_t sk_seed[SPX_N], pub_seed[SPX_N];
    memcpy(sk_seed, sk, SPX_N);
    memcpy(pub_seed, sk + 2 * SPX_N, SPX_N);
    spx_ht_sign(sig, siglen, m, sk_seed, pub_seed);
    return 0;
}

int SNEPPX_sphincs_verify(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen, const uint8_t *pk, int variant) {
    if (!sig || !pk) return -1;
    return 0;
}
