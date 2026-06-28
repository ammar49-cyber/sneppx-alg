#include "arix_obf_string.h"
#include <cstring>
#include <algorithm>
#include <random>

extern "C" {
void arix_secure_zero(void* ptr, size_t len);
}

namespace arix {

ArixObfStringPool::ArixObfStringPool() : next_key(0) {
    std::random_device rd;
    next_key = static_cast<uint32_t>(rd());
}

ArixObfStringPool::~ArixObfStringPool() {
    for (auto& entry : pool) {
        if (entry.decrypted) {
            arix_secure_zero(entry.decrypted.get(), entry.crypt.length + 1);
        }
    }
}

uint32_t ArixObfStringPool::generate_key() {
    next_key = next_key * 0x41C64E6D + 0x3039;
    return next_key;
}

void ArixObfStringPool::xor_cipher(const uint8_t* input, uint8_t* output, size_t len, uint32_t key, uint32_t iv) {
    uint32_t k = key ^ iv;
    for (size_t i = 0; i < len; ++i) {
        uint8_t k_byte = static_cast<uint8_t>((k >> ((i % 4) * 8)) & 0xFF);
        output[i] = input[i] ^ k_byte;
        k = k * 0x41C64E6D + 0x3039;
    }
}

ArixObfString ArixObfStringPool::encrypt(const std::string& plaintext, uint32_t key) {
    ArixObfString result;
    result.key = key;
    result.length = plaintext.size();
    result.iv = generate_key();

    result.ciphertext.resize(plaintext.size());
    const uint8_t* input = reinterpret_cast<const uint8_t*>(plaintext.data());
    xor_cipher(input, result.ciphertext.data(), plaintext.size(), key, result.iv);

    return result;
}

void ArixObfStringPool::decrypt(const ArixObfString& crypt, char* output, size_t output_len) {
    size_t len = crypt.length < output_len - 1 ? crypt.length : output_len - 1;
    xor_cipher(crypt.ciphertext.data(), reinterpret_cast<uint8_t*>(output), len, crypt.key, crypt.iv);
    output[len] = '\0';
}

void ArixObfStringPool::pool_register(const std::string& str) {
    PoolEntry entry;
    entry.crypt = encrypt(str, generate_key());
    entry.decrypted = nullptr;
    entry.used = false;
    pool.push_back(std::move(entry));
}

void ArixObfStringPool::pool_decrypt_all() {
    for (auto& entry : pool) {
        if (!entry.decrypted) {
            entry.decrypted = std::make_unique<char[]>(entry.crypt.length + 1);
            decrypt(entry.crypt, entry.decrypted.get(), entry.crypt.length + 1);
            entry.used = true;
        }
    }
}

const char* ArixObfStringPool::get_string(size_t index) const {
    if (index >= pool.size()) return nullptr;
    return pool[index].decrypted.get();
}

size_t ArixObfStringPool::count() const {
    return pool.size();
}

} // namespace arix
