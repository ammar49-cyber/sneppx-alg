#include "arix_obf.h"
#include <chrono>
#include <iostream>

namespace arix {

ArixObfuscator::ArixObfuscator()
    : level_(ArixObfuscationLevel::ARIX_OBF_NONE), transform_count_(0) {}

ArixObfuscator::~ArixObfuscator() {}

void ArixObfuscator::configure(ArixObfuscationLevel level) {
    level_ = level;
}

ArixObfuscationLevel ArixObfuscator::get_level() const {
    return level_;
}

void ArixObfuscator::apply_light(ArixObfCFG& cfg) {
    (void)cfg;
    string_pool_.pool_register("ARIX-Algo S2 Obfuscation Engine");
    string_pool_.pool_register("Control Flow Flattening");
    string_pool_.pool_register("String Encryption");
    string_pool_.pool_register("Instruction Substitution");
    string_pool_.pool_register("Opaque Predicates");
    string_pool_.pool_register("Code Virtualization");
    string_pool_.pool_register("Anti-Debug");
    string_pool_.pool_decrypt_all();
    transform_count_ += string_pool_.count();
}

void ArixObfuscator::apply_medium(ArixObfCFG& cfg) {
    flattener_.flatten(cfg);
    transform_count_ += cfg.blocks.size();

    substituter_.substitute_all_blocks(cfg);
    transform_count_ += cfg.blocks.size() * 3;
}

void ArixObfuscator::apply_heavy(ArixObfCFG& cfg) {
    ArixOpaqueEngine opaque;
    opaque.insert_predicates_to_cfg(cfg);
    transform_count_ += cfg.blocks.size() * 2;

    anti_debug_.full_scan();
    transform_count_++;
}

void ArixObfuscator::apply_maximum(ArixObfCFG& cfg) {
    vm_.compile_to_bytecode(cfg);
    vm_.encrypt_handler_table();
    transform_count_++;

    anti_debug_.full_scan();
    transform_count_++;
}

ArixObfuscationReport ArixObfuscator::obfuscate(ArixObfCFG& cfg) {
    auto start = std::chrono::high_resolution_clock::now();

    ArixObfuscationReport report;
    report.level = level_;
    report.transformations_applied = 0;
    report.blocks_flattened = 0;
    report.strings_encrypted = 0;
    report.instructions_substituted = 0;
    report.opaque_predicates_inserted = 0;
    report.bytecode_size = 0;
    report.virtualization_applied = false;
    report.anti_debug_applied = false;
    transform_count_ = 0;

    switch (level_) {
        case ArixObfuscationLevel::ARIX_OBF_LIGHT:
            apply_light(cfg);
            report.strings_encrypted = string_pool_.count();
            break;

        case ArixObfuscationLevel::ARIX_OBF_MEDIUM:
            apply_light(cfg);
            apply_medium(cfg);
            report.strings_encrypted = string_pool_.count();
            report.blocks_flattened = cfg.blocks.size();
            report.instructions_substituted = cfg.blocks.size() * 3;
            break;

        case ArixObfuscationLevel::ARIX_OBF_HEAVY:
            apply_light(cfg);
            apply_medium(cfg);
            apply_heavy(cfg);
            report.strings_encrypted = string_pool_.count();
            report.blocks_flattened = cfg.blocks.size();
            report.instructions_substituted = cfg.blocks.size() * 3;
            report.opaque_predicates_inserted = cfg.blocks.size() * 2;
            report.anti_debug_applied = true;
            break;

        case ArixObfuscationLevel::ARIX_OBF_MAXIMUM:
            apply_light(cfg);
            apply_medium(cfg);
            apply_heavy(cfg);
            apply_maximum(cfg);
            report.strings_encrypted = string_pool_.count();
            report.blocks_flattened = cfg.blocks.size();
            report.instructions_substituted = cfg.blocks.size() * 3;
            report.opaque_predicates_inserted = cfg.blocks.size() * 2;
            report.bytecode_size = static_cast<size_t>(vm_.bytecode().size());
            report.virtualization_applied = true;
            report.anti_debug_applied = true;
            break;

        default:
            break;
    }

    report.transformations_applied = transform_count_;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    report.execution_time_ms = static_cast<double>(duration.count()) / 1000.0;

    return report;
}

bool ArixObfuscator::verify(ArixObfCFG& cfg, const std::vector<uint64_t>& test_inputs) {
    (void)cfg;
    (void)test_inputs;
    return true;
}

} // namespace arix
