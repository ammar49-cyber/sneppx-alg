#pragma once
#ifndef ARIX_OBF_STRING_H
#define ARIX_OBF_STRING_H

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <memory>

namespace arix {

struct ArixObfString {
    std::vector<uint8_t> ciphertext;
    uint32_t key;
    size_t length;
    uint32_t iv;
};

class ArixObfStringPool {
public:
    ArixObfStringPool();
    ~ArixObfStringPool();

    ArixObfString encrypt(const std::string& plaintext, uint32_t key);
    void decrypt(const ArixObfString& crypt, char* output, size_t output_len);
    void pool_register(const std::string& str);
    void pool_decrypt_all();
    const char* get_string(size_t index) const;
    size_t count() const;

private:
    struct PoolEntry {
        ArixObfString crypt;
        std::unique_ptr<char[]> decrypted;
        bool used;
    };

    std::vector<PoolEntry> pool;
    uint32_t next_key;

    uint32_t generate_key();
    void xor_cipher(const uint8_t* input, uint8_t* output, size_t len, uint32_t key, uint32_t iv);
};

constexpr uint32_t arix_obf_compile_time_key(uint32_t base) {
    return base ^ 0xA71A75A7;
}

template<size_t N>
constexpr std::array<uint8_t, N> arix_obf_encrypt_literal(const char(&str)[N], uint32_t key) {
    std::array<uint8_t, N> result{};
    for (size_t i = 0; i < N; ++i) {
        result[i] = static_cast<uint8_t>(str[i]) ^ static_cast<uint8_t>((key >> ((i % 4) * 8)) & 0xFF);
    }
    return result;
}

} // namespace arix

#endif // ARIX_OBF_STRING_H
