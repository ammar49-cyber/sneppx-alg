#pragma once
#ifndef ARIX_OBF_ANTI_H
#define ARIX_OBF_ANTI_H

#include <cstdint>
#include <string>
#include <functional>

namespace arix {

enum class ArixAntiDebugAction {
    CRASH,
    FAKE_DATA,
    DELAY,
    SILENT_EXIT,
    WIPE_AND_EXIT
};

class ArixAntiDebug {
public:
    ArixAntiDebug();
    ~ArixAntiDebug();

    void set_action(ArixAntiDebugAction action);
    ArixAntiDebugAction get_action() const;

    bool detect_ptrace();
    bool detect_debugger_present();
    bool detect_timing_anomaly();
    bool detect_breakpoint(const uint8_t* code_start, size_t code_len);
    bool detect_vm();

    void react();
    void full_scan();

    bool is_debugger_attached() const { return debugger_detected; }

private:
    ArixAntiDebugAction action;
    bool debugger_detected;

    uint64_t read_timestamp();
    uint32_t compute_code_hash(const uint8_t* code, size_t len);
    void wipe_sensitive_data();
};

} // namespace arix

#endif // ARIX_OBF_ANTI_H
