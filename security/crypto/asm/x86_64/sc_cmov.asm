; ARIX_SC_CMOV — Side-channel resistant operations using CMOV
; Targeted microarchitectures: Intel Core (Nehalem+), AMD Ryzen, any x86_64 with CMOV
; Guaranteed constant-time: no input-dependent branches or cache accesses

section .text
global arix_sc_select_u64_asm
global arix_sc_equal_u64_asm

; uint64_t arix_sc_select_u64_asm(uint64_t condition, uint64_t a, uint64_t b)
; Returns a if condition == ~0, b if condition == 0
arix_sc_select_u64_asm:
    mov     rax, rdx        ; rax = b (default)
    test    rdi, rdi        ; test condition
    cmovnz  rax, rsi        ; if condition != 0, rax = a
    ret

; uint64_t arix_sc_equal_u64_asm(uint64_t a, uint64_t b)
; Returns ~0 if a == b, 0 otherwise
arix_sc_equal_u64_asm:
    xor     rax, rax        ; rax = 0
    cmp     rdi, rsi        ; compare a, b
    sete    al              ; al = 1 if equal
    neg     rax             ; rax = ~0 if equal, 0 if not
    ret
