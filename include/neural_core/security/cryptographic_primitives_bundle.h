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

#include "constant_time_operations.h"
#include "sha512_hashing_implementation.h"
#include "keccak_sha3_hashing.h"
#include "cryptographic_hashing_blake3.h"
#include "ed25519_signature_verification.h"
#include "chacha20_stream_cipher.h"
#include "polynomial_authentication_mac.h"
#include "authenticated_encryption_module.h"
#include "cryptographic_random_generator.h"
#include "memory_hard_key_derivation.h"

#include "protected_memory_manager.h"
#include "stack_canary_protection.h"
#include "address_space_randomization.h"
#include "synchronization_lock_interface.h"
#include "side_channel_resistant_primitives.h"
#include "timing_attack_countermeasure.h"
#include "secure_cache_management.h"
#include "power_analysis_mitigation.h"

#endif
