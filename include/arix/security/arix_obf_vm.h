#pragma once
#ifndef ARIX_OBF_VM_H
#define ARIX_OBF_VM_H

#include "arix_obf_cfg.h"
#include <cstdint>
#include <vector>
#include <array>

namespace arix {

enum class ArixObfBytecode : uint8_t {
    NOP    = 0x00,
    ADD    = 0x01,
    SUB    = 0x02,
    MUL    = 0x03,
    DIV    = 0x04,
    LOAD   = 0x05,
    STORE  = 0x06,
    PUSH   = 0x07,
    POP    = 0x08,
    JMP    = 0x09,
    CALL   = 0x0A,
    RET    = 0x0B,
    CMP    = 0x0C,
    JZ     = 0x0D,
    JNZ    = 0x0E,
    HALT   = 0xFF
};

struct ArixObfVMState {
    std::array<uint64_t, 256> regs;
    std::vector<uint64_t> stack;
    size_t ip;
    bool running;
    uint64_t flags;

    ArixObfVMState() : ip(0), running(true), flags(0) {
        regs.fill(0);
    }
};

using ArixObfHandlerFunc = void(*)(ArixObfVMState& state, uint8_t op1, uint8_t op2);

struct ArixObfHandler {
    ArixObfHandlerFunc func;
    bool initialized;
};

class ArixObfVM {
public:
    ArixObfVM();
    ~ArixObfVM();

    void add_handler(ArixObfBytecode opcode, ArixObfHandler handler);
    void compile_to_bytecode(ArixObfCFG& cfg);
    bool vm_execute(const uint8_t* bytecode, size_t len);
    bool load_bytecode(const std::vector<uint8_t>& bc);

    ArixObfVMState& state() { return vm_state; }
    const std::vector<uint8_t>& bytecode() const { return bytecode_; }

    void encrypt_handler_table();
    void decrypt_handler_table();

private:
    ArixObfVMState vm_state;
    std::vector<uint8_t> bytecode_;
    std::array<ArixObfHandler, 16> handlers;
    std::array<uint8_t, 8> handler_xor_key;
    bool table_encrypted;

    void dispatch(uint8_t opcode, uint8_t op1, uint8_t op2);
    static void handler_nop(ArixObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_add(ArixObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_sub(ArixObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_mul(ArixObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_load(ArixObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_store(ArixObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_push(ArixObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_pop(ArixObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_jmp(ArixObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_call(ArixObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_ret(ArixObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_cmp(ArixObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_jz(ArixObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_jnz(ArixObfVMState& state, uint8_t op1, uint8_t op2);
};

} // namespace arix

#endif // ARIX_OBF_VM_H
