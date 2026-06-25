#ifndef ARIX_CRYPTO_H
#define ARIX_CRYPTO_H

#include <stddef.h>
#include <stdint.h>

#define ARIX_CRYPTO_VERSION "0.1.0"

#if defined(__x86_64__) || defined(_M_X64)
#define ARIX_CRYPTO_ASM_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define ARIX_CRYPTO_ASM_ARM64 1
#endif

#include "arix_ct.h"
#include "arix_sha512.h"
#include "arix_sha3.h"
#include "arix_blake3.h"
#include "arix_ed25519.h"
#include "arix_chacha20.h"
#include "arix_poly1305.h"
#include "arix_aead.h"
#include "arix_random.h"
#include "arix_argon2.h"

#include "arix_secure_mem.h"
#include "arix_canary.h"
#include "arix_aslr.h"
#include "arix_lock.h"
#include "arix_sc.h"
#include "arix_timing.h"
#include "arix_cache.h"
#include "arix_power.h"

#endif
