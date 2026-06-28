#include "arix_obf_vm.h"
#include <cstring>
#include <algorithm>
#include <iostream>

extern "C" {
void arix_secure_zero(void* ptr, size_t len);
}

namespace arix {

ArixObfVM::ArixObfVM() : table_encrypted(false) {
    handler_xor_key.fill(0xAB);

    handlers[0] = { handler_nop, true };
    handlers[1] = { handler_add, true };
    handlers[2] = { handler_sub, true };
    handlers[3] = { handler_mul, true };
    handlers[4] = { handler_nop, true };
    handlers[5] = { handler_load, true };
    handlers[6] = { handler_store, true };
    handlers[7] = { handler_push, true };
    handlers[8] = { handler_pop, true };
    handlers[9] = { handler_jmp, true };
    handlers[10] = { handler_call, true };
    handlers[11] = { handler_ret, true };
    handlers[12] = { handler_cmp, true };
    handlers[13] = { handler_jz, true };
    handlers[14] = { handler_jnz, true };
}

ArixObfVM::~ArixObfVM() {
    arix_secure_zero(handler_xor_key.data(), handler_xor_key.size());
}

void ArixObfVM::add_handler(ArixObfBytecode opcode, ArixObfHandler handler) {
    size_t idx = static_cast<size_t>(opcode);
    if (idx < handlers.size()) {
        handlers[idx] = handler;
    }
}

void ArixObfVM::encrypt_handler_table() {
    if (table_encrypted) return;
    table_encrypted = true;
}

void ArixObfVM::decrypt_handler_table() {
    if (!table_encrypted) return;
    table_encrypted = false;
}

void ArixObfVM::compile_to_bytecode(ArixObfCFG& cfg) {
    bytecode_.clear();

    bytecode_.push_back(static_cast<uint8_t>(ArixObfBytecode::LOAD));
    bytecode_.push_back(0);
    bytecode_.push_back(1);
    bytecode_.push_back(0);

    for (auto& pair : cfg.blocks) {
        if (pair.second->is_exit) {
            bytecode_.push_back(static_cast<uint8_t>(ArixObfBytecode::HALT));
            bytecode_.push_back(0);
            bytecode_.push_back(0);
            bytecode_.push_back(0);
            continue;
        }

        for (auto& inst : pair.second->instructions) {
            switch (inst.type) {
                case ArixObfInstType::ADD:
                    bytecode_.push_back(static_cast<uint8_t>(ArixObfBytecode::ADD));
                    bytecode_.push_back(0);
                    bytecode_.push_back(1);
                    bytecode_.push_back(0);
                    break;
                case ArixObfInstType::SUB:
                    bytecode_.push_back(static_cast<uint8_t>(ArixObfBytecode::SUB));
                    bytecode_.push_back(0);
                    bytecode_.push_back(1);
                    bytecode_.push_back(0);
                    break;
                case ArixObfInstType::MUL:
                    bytecode_.push_back(static_cast<uint8_t>(ArixObfBytecode::MUL));
                    bytecode_.push_back(0);
                    bytecode_.push_back(1);
                    bytecode_.push_back(0);
                    break;
                default:
                    bytecode_.push_back(static_cast<uint8_t>(ArixObfBytecode::NOP));
                    bytecode_.push_back(0);
                    bytecode_.push_back(0);
                    bytecode_.push_back(0);
                    break;
            }
        }
    }

    bytecode_.push_back(static_cast<uint8_t>(ArixObfBytecode::HALT));
    bytecode_.push_back(0);
    bytecode_.push_back(0);
    bytecode_.push_back(0);
}

bool ArixObfVM::load_bytecode(const std::vector<uint8_t>& bc) {
    bytecode_ = bc;
    return true;
}

bool ArixObfVM::vm_execute(const uint8_t* bytecode, size_t len) {
    vm_state = ArixObfVMState();
    vm_state.ip = 0;
    vm_state.running = true;

    decrypt_handler_table();

    while (vm_state.running && vm_state.ip < len) {
        if (vm_state.ip + 4 > len) break;

        uint8_t opcode = bytecode[vm_state.ip];
        uint8_t op1 = bytecode[vm_state.ip + 1];
        uint8_t op2 = bytecode[vm_state.ip + 2];

        dispatch(opcode, op1, op2);
        vm_state.ip += 4;
    }

    encrypt_handler_table();

    return true;
}

void ArixObfVM::dispatch(uint8_t opcode, uint8_t op1, uint8_t op2) {
    if (opcode == static_cast<uint8_t>(ArixObfBytecode::HALT)) {
        vm_state.running = false;
        return;
    }
    if (opcode < handlers.size() && handlers[opcode].initialized) {
        handlers[opcode].func(vm_state, op1, op2);
    }
}

void ArixObfVM::handler_nop(ArixObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)state; (void)op1; (void)op2;
}

void ArixObfVM::handler_add(ArixObfVMState& state, uint8_t op1, uint8_t op2) {
    state.regs[op1] += state.regs[op2];
}

void ArixObfVM::handler_sub(ArixObfVMState& state, uint8_t op1, uint8_t op2) {
    state.regs[op1] -= state.regs[op2];
}

void ArixObfVM::handler_mul(ArixObfVMState& state, uint8_t op1, uint8_t op2) {
    state.regs[op1] *= state.regs[op2];
}

void ArixObfVM::handler_load(ArixObfVMState& state, uint8_t op1, uint8_t op2) {
    state.regs[op1] = static_cast<uint64_t>(op2);
}

void ArixObfVM::handler_store(ArixObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    state.regs[op1] = state.regs[op1];
}

void ArixObfVM::handler_push(ArixObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    state.stack.push_back(state.regs[op1]);
}

void ArixObfVM::handler_pop(ArixObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    if (!state.stack.empty()) {
        state.regs[op1] = state.stack.back();
        state.stack.pop_back();
    }
}

void ArixObfVM::handler_jmp(ArixObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    state.ip = static_cast<size_t>(state.regs[op1]);
}

void ArixObfVM::handler_call(ArixObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    state.stack.push_back(state.ip + 4);
    state.ip = static_cast<size_t>(state.regs[op1]);
}

void ArixObfVM::handler_ret(ArixObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op1; (void)op2;
    if (!state.stack.empty()) {
        state.ip = state.stack.back();
        state.stack.pop_back();
    }
}

void ArixObfVM::handler_cmp(ArixObfVMState& state, uint8_t op1, uint8_t op2) {
    if (state.regs[op1] == state.regs[op2]) {
        state.flags = 0;
    } else if (state.regs[op1] < state.regs[op2]) {
        state.flags = 1;
    } else {
        state.flags = 2;
    }
}

void ArixObfVM::handler_jz(ArixObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    if (state.flags == 0) {
        state.ip = static_cast<size_t>(state.regs[op1]);
    }
}

void ArixObfVM::handler_jnz(ArixObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    if (state.flags != 0) {
        state.ip = static_cast<size_t>(state.regs[op1]);
    }
}

} // namespace arix
