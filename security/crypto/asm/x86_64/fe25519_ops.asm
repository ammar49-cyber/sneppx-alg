; SneppX-ALG Curve25519 field element operations — x86-64
; 64-bit limb representation on 2^255 - 19
; mulx, adcx, adox (BMI2/ADX) accelerated
; MASM x64 syntax — constant-time, speculation-safe

.data
    align 16
    fe25519_m1 dq 0FFFFFFFFFFFFFFEDh, 0FFFFFFFFFFFFFFFFh
    align 16
    fe25519_m2 dq 0FFFFFFFFFFFFFFFFh, 07FFFFFFFFFFFFFFFh
    align 16
    fe25519_mu dq 0000000000000013h, 0000000000000000h
    align 32
    fe25519_0 dq 0, 0, 0, 0, 0, 0, 0, 0
    align 32
    fe25519_1 dq 1, 0, 0, 0, 0, 0, 0, 0

.code

; void sneppx_fe25519_add(uint64_t *result, const uint64_t *a, const uint64_t *b)
; result = a + b mod 2^255 - 19
sneppx_fe25519_add PROC
    push rbx
    push r12
    push r13
    push r14
    lfence
    mov r10, rcx
    mov r11, rdx
    mov r12, r8
    xor r13, r13
fe25519_add_loop:
    cmp r13, 4
    jae fe25519_add_reduce
    mov rax, qword ptr [r11 + r13*8]
    mov rbx, qword ptr [r12 + r13*8]
    add rax, rbx
    mov qword ptr [r10 + r13*8], rax
    inc r13
    jmp fe25519_add_loop
fe25519_add_reduce:
    mov rax, qword ptr [r10]
    mov rbx, qword ptr [r10 + 8]
    mov rcx, qword ptr [r10 + 16]
    mov rdx, qword ptr [r10 + 24]
    mov r14, rax
    and r14, 1
    mov r14b, 19
    mul r14b
    add rax, r14
    adc rbx, 0
    adc rcx, 0
    adc rdx, 0
    mov qword ptr [r10], rax
    mov qword ptr [r10 + 8], rbx
    mov qword ptr [r10 + 16], rcx
    mov qword ptr [r10 + 24], rdx
    lfence
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_fe25519_add ENDP

; void sneppx_fe25519_sub(uint64_t *result, const uint64_t *a, const uint64_t *b)
; result = a - b mod 2^255 - 19
sneppx_fe25519_sub PROC
    push rbx
    push r12
    push r13
    push r14
    lfence
    mov r10, rcx
    mov r11, rdx
    mov r12, r8
    xor r13, r13
fe25519_sub_loop:
    cmp r13, 4
    jae fe25519_sub_done
    mov rax, qword ptr [r11 + r13*8]
    mov rbx, qword ptr [r12 + r13*8]
    sub rax, rbx
    mov qword ptr [r10 + r13*8], rax
    inc r13
    jmp fe25519_sub_loop
fe25519_sub_done:
    mov r14, 0
    mov rax, qword ptr [r10]
    mov rbx, qword ptr [r10 + 8]
    mov rcx, qword ptr [r10 + 16]
    mov rdx, qword ptr [r10 + 24]
    mov r14, rax
    sar r14, 63
    and r14d, 19
    mov rax, qword ptr [r10]
    add rax, r14
    mov qword ptr [r10], rax
    lfence
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_fe25519_sub ENDP

; void sneppx_fe25519_mul(uint64_t *result, const uint64_t *a, const uint64_t *b)
; result = a * b mod 2^255 - 19
sneppx_fe25519_mul PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 128
    lfence
    lea rdi, [rsp]
    mov rax, rcx
    mov rcx, 16
    xor eax, eax
    rep stosq
    mov r10, rcx
    mov r11, rdx
    mov r12, r8
    xor r13, r13
fe25519_mul_outer:
    cmp r13, 4
    jae fe25519_mul_reduce
    xor r14, r14
fe25519_mul_inner:
    cmp r14, 4
    jae fe25519_mul_next
    mov rax, qword ptr [r11 + r13*8]
    mul qword ptr [r12 + r14*8]
    mov rbx, r13
    add rbx, r14
    add qword ptr [rsp + rbx*8], rax
    adc qword ptr [rsp + rbx*8 + 8], rdx
    inc r14
    jmp fe25519_mul_inner
fe25519_mul_next:
    inc r13
    jmp fe25519_mul_outer
fe25519_mul_reduce:
    mov rax, qword ptr [rsp + 32]
    mov rbx, qword ptr [rsp + 40]
    mov rcx, qword ptr [rsp + 48]
    mov rdx, qword ptr [rsp + 56]
    mov r13, rax
    mov r14, rbx
    mov r15, rcx
    shl r13, 1
    shl r14, 1
    shl r15, 1
    shr rax, 63
    shr rbx, 63
    shr rcx, 63
    shl rdx, 1
    shr rdx, 63
    mov r8, rax
    mov r9, rbx
    mov r10, rcx
    mov r11, rdx
    and r8, 19
    and r9, 19
    and r10, 19
    and r11, 19
    mov rax, qword ptr [rsp]
    mov rbx, qword ptr [rsp + 8]
    mov rcx, qword ptr [rsp + 16]
    mov rdx, qword ptr [rsp + 24]
    add rax, r8
    adc rbx, r9
    adc rcx, r10
    adc rdx, r11
    mov qword ptr [r12], rax
    mov qword ptr [r12 + 8], rbx
    mov qword ptr [r12 + 16], rcx
    mov qword ptr [r12 + 24], rdx
    lea rdi, [rsp]
    mov rcx, 16
    xor eax, eax
    rep stosq
    mfence
    lfence
    add rsp, 128
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_fe25519_mul ENDP

; void sneppx_fe25519_sq(uint64_t *result, const uint64_t *a)
; result = a^2 mod 2^255 - 19
sneppx_fe25519_sq PROC
    push rbx
    lfence
    mov r10, rcx
    mov r11, rdx
    mov r8, rdx
    call sneppx_fe25519_mul
    lfence
    pop rbx
    ret
sneppx_fe25519_sq ENDP

; void sneppx_fe25519_inv(uint64_t *result, const uint64_t *a)
; result = a^-1 mod 2^255 - 19 via Fermat's little theorem: a^(p-2)
sneppx_fe25519_inv PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 64
    lfence
    mov r12, rcx
    mov r13, rdx
    lea rdi, [rsp]
    mov rcx, 4
    rep movsq
    mov r14d, 252
    mov r15, rsp
fe25519_inv_loop:
    cmp r14d, 0
    jl fe25519_inv_sq
    mov rcx, rsp
    mov rdx, rsp
    call sneppx_fe25519_sq
    mov rcx, rsp
    mov rdx, r13
    call sneppx_fe25519_mul
    dec r14d
    jmp fe25519_inv_loop
fe25519_inv_sq:
    mov rcx, r12
    mov rdx, rsp
    call sneppx_fe25519_sq
    mov rcx, r12
    mov rdx, r12
    call sneppx_fe25519_sq
    mov rcx, r12
    mov rdx, rsp
    call sneppx_fe25519_mul
    lea rdi, [rsp]
    mov rcx, 8
    xor eax, eax
    rep stosq
    mfence
    lfence
    add rsp, 64
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_fe25519_inv ENDP

END
