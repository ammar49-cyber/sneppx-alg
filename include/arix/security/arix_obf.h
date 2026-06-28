#pragma once
#ifndef ARIX_OBF_H
#define ARIX_OBF_H

#include "arix_obf_cfg.h"
#include "arix_obf_string.h"
#include "arix_obf_inst.h"
#include "arix_obf_opaque.h"
#include "arix_obf_vm.h"
#include "arix_obf_anti.h"

namespace arix {

enum class ArixObfuscationLevel {
    ARIX_OBF_NONE,
    ARIX_OBF_LIGHT,
    ARIX_OBF_MEDIUM,
    ARIX_OBF_HEAVY,
    ARIX_OBF_MAXIMUM
};

struct ArixObfuscationReport {
    ArixObfuscationLevel level;
    size_t transformations_applied;
    size_t blocks_flattened;
    size_t strings_encrypted;
    size_t instructions_substituted;
    size_t opaque_predicates_inserted;
    size_t bytecode_size;
    bool virtualization_applied;
    bool anti_debug_applied;
    double execution_time_ms;
};

class ArixObfuscator {
public:
    ArixObfuscator();
    ~ArixObfuscator();

    void configure(ArixObfuscationLevel level);
    ArixObfuscationLevel get_level() const;

    ArixObfuscationReport obfuscate(ArixObfCFG& cfg);
    bool verify(ArixObfCFG& cfg, const std::vector<uint64_t>& test_inputs);

    ArixObfStringPool& string_pool() { return string_pool_; }
    ArixObfCFGFlattener& flattener() { return flattener_; }
    ArixObfSubst& substituter() { return substituter_; }
    ArixObfVM& vm() { return vm_; }
    ArixAntiDebug& anti_debug() { return anti_debug_; }

private:
    ArixObfuscationLevel level_;
    ArixObfStringPool string_pool_;
    ArixObfCFGFlattener flattener_;
    ArixObfSubst substituter_;
    ArixObfVM vm_;
    ArixAntiDebug anti_debug_;

    size_t transform_count_;

    void apply_light(ArixObfCFG& cfg);
    void apply_medium(ArixObfCFG& cfg);
    void apply_heavy(ArixObfCFG& cfg);
    void apply_maximum(ArixObfCFG& cfg);
};

} // namespace arix

#endif // ARIX_OBF_H
