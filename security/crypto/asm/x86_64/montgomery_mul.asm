; SneppX-ALG Montgomery multiplication for RSA/ECC — x86-64
; Uses BMI2 (mulx, adcx, adox) for MULX-based multiplication
; Constant-time: fixed number of iterations, no secret-dependent branches
; MASM x64 syntax — speculation-safe, memory-wiping

.data
    align 16
    mont_mask dq 0FFFFFFFFFFFFFFFFh, 0FFFFFFFFFFFFFFFFh
    align 16
    mont_zero dq 0000000000000000h, 0000000000000000h
    align 32
    mont_wipe dq 0, 0, 0, 0

.code

; void sneppx_montgomery_mul(uint64_t *result, const uint64_t *a, const uint64_t *b, const uint64_t *mod, uint64_t inv, size_t len)
; Computes result = a * b * R^-1 mod mod
; inv = -mod[0]^-1 mod 2^64 (Montgomery constant)
; len = number of 64-bit limbs
sneppx_montgomery_mul PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 256
    lfence
    lea rdi, [rsp]
    mov r10, r9
    mov rcx, 32
    xor eax, eax
    rep stosq
    mov r11, qword ptr [rsp + 288]
    mov r12, rcx
    mov r13, rdx
    mov r14, r8
    mov r15, r9
    xor rbx, rbx
mont_mul_outer:
    cmp rbx, r11
    jae mont_mul_carry
    xor rcx, rcx
    xor r8, r8
    xor r9, r9
    mov rax, qword ptr [r13 + rbx*8]
    xor r10, r10
mont_mul_inner:
    cmp r10, r11
    jae mont_mul_reduce
    mulx rdx, rax, qword ptr [r14 + r10*8]
    mov rcx, qword ptr [rsp + r10*8]
    adcx rcx, rax
    adox rcx, rdx
    mov qword ptr [rsp + r10*8], rcx
    inc r10
    jmp mont_mul_inner
mont_mul_reduce:
    mov rax, qword ptr [rsp]
    mulx rdx, rax, r15
    xor r10, r10
mont_reduce_inner:
    cmp r10, r11
    jae mont_mul_next
    mulx rdx, rax, qword ptr [r14 + r10*8]
    mov rcx, qword ptr [rsp + r10*8]
    adcx rcx, rax
    adox rcx, rdx
    mov qword ptr [rsp + r10*8], rcx
    inc r10
    jmp mont_reduce_inner
mont_mul_next:
    mov rcx, qword ptr [rsp + r11*8]
    mov qword ptr [rsp], rcx
    xor r10, r10
mont_shift_loop:
    cmp r10, r11
    jae mont_shift_done
    mov rax, qword ptr [rsp + r10*8 + 8]
    mov qword ptr [rsp + r10*8], rax
    inc r10
    jmp mont_shift_loop
mont_shift_done:
    mov qword ptr [rsp + r11*8], 0
    inc rbx
    jmp mont_mul_outer
mont_mul_carry:
    xor rbx, rbx
    xor rcx, rcx
mont_final_sub_loop:
    cmp rbx, r11
    jae mont_final_sub_done
    mov rax, qword ptr [rsp + rbx*8]
    mov rdx, qword ptr [r14 + rbx*8]
    sbb rax, rdx
    mov qword ptr [rsp + rbx*8], rax
    inc rbx
    jmp mont_final_sub_loop
mont_final_sub_done:
    sbb rcx, rcx
    xor rbx, rbx
mont_final_add_loop:
    cmp rbx, r11
    jae mont_copy_out
    mov rax, qword ptr [rsp + rbx*8]
    mov rdx, qword ptr [r14 + rbx*8]
    and rdx, rcx
    add rax, rdx
    mov qword ptr [rsp + rbx*8], rax
    inc rbx
    jmp mont_final_add_loop
mont_copy_out:
    mov rdi, r12
    lea rsi, [rsp]
    mov rcx, r11
    rep movsq
    lea rdi, [rsp]
    mov rcx, 32
    xor eax, eax
    rep stosq
    mfence
    lfence
    add rsp, 256
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_montgomery_mul ENDP

; void sneppx_montgomery_redc(uint64_t *result, const uint64_t *t, const uint64_t *mod, uint64_t inv, size_t len)
; Montgomery reduction: result = t * R^-1 mod mod
sneppx_montgomery_redc PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 256
    lfence
    lea rdi, [rsp]
    mov r10, r9
    mov rcx, 32
    xor eax, eax
    rep stosq
    mov r11, qword ptr [rsp + 288]
    mov r12, rcx
    mov r13, rdx
    mov r14, r8
    mov r15, r9
    xor rbx, rbx
mont_redc_outer:
    cmp rbx, r11
    jae mont_redc_final
    mov rax, qword ptr [r13]
    mulx rdx, rax, r15
    xor r10, r10
mont_redc_inner:
    cmp r10, r11
    jae mont_redc_next
    mulx rdx, rax, qword ptr [r14 + r10*8]
    mov rcx, qword ptr [r13 + r10*8]
    adcx rcx, rax
    adox rcx, rdx
    mov qword ptr [r13 + r10*8], rcx
    inc r10
    jmp mont_redc_inner
mont_redc_next:
    mov rcx, qword ptr [r13 + r11*8]
    mov qword ptr [r13], rcx
    xor r10, r10
mont_redc_shift:
    cmp r10, r11
    jae mont_redc_shift_done
    mov rax, qword ptr [r13 + r10*8 + 8]
    mov qword ptr [r13 + r10*8], rax
    inc r10
    jmp mont_redc_shift
mont_redc_shift_done:
    mov qword ptr [r13 + r11*8], 0
    inc rbx
    jmp mont_redc_outer
mont_redc_final:
    xor rbx, rbx
    xor rcx, rcx
mont_redc_sub_loop:
    cmp rbx, r11
    jae mont_redc_add
    mov rax, qword ptr [r13 + rbx*8]
    mov rdx, qword ptr [r14 + rbx*8]
    sbb rax, rdx
    mov qword ptr [r13 + rbx*8], rax
    inc rbx
    jmp mont_redc_sub_loop
mont_redc_add:
    sbb rcx, rcx
    xor rbx, rbx
mont_redc_add_loop:
    cmp rbx, r11
    jae mont_redc_done
    mov rax, qword ptr [r13 + rbx*8]
    mov rdx, qword ptr [r14 + rbx*8]
    and rdx, rcx
    add rax, rdx
    mov qword ptr [r12 + rbx*8], rax
    inc rbx
    jmp mont_redc_add_loop
mont_redc_done:
    lea rdi, [rsp]
    mov rcx, 32
    xor eax, eax
    rep stosq
    mfence
    lfence
    add rsp, 256
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_montgomery_redc ENDP

; void sneppx_montgomery_to_mont(uint64_t *result, const uint64_t *a, const uint64_t *mod, uint64_t inv, size_t len)
; Convert to Montgomery form: result = a * R mod mod
sneppx_montgomery_to_mont PROC
    push rbx
    sub rsp, 64
    lfence
    mov r10, rcx
    mov r11, rdx
    mov rbx, rsp
    mov qword ptr [rbx], 1
    xor rcx, rcx
    mov rcx, r9
    lea rdi, [rbx + 8]
    mov rax, rcx
    dec rax
    mov qword ptr [rdi + rax*8], 0
    xor rax, rax
mont_to_mont_shift:
    cmp rax, rcx
    jae mont_to_mont_shift_done
    mov qword ptr [rdi + rax*8], 0
    inc rax
    jmp mont_to_mont_shift
mont_to_mont_shift_done:
    mov qword ptr [rbx], 1
    mov rcx, r10
    mov rdx, r11
    mov r8, r9
    lea r9, [rsp]
    mov qword ptr [rsp + 64], r8
    mov qword ptr [rsp + 72], r9
    mov rax, qword ptr [rsp + 88]
    mov qword ptr [rsp + 80], rax
    call sneppx_montgomery_mul
    lea rdi, [rsp]
    mov rcx, 8
    xor eax, eax
    rep stosq
    lfence
    add rsp, 64
    pop rbx
    ret
sneppx_montgomery_to_mont ENDP

; void sneppx_montgomery_from_mont(uint64_t *result, const uint64_t *a, const uint64_t *mod, uint64_t inv, size_t len)
; Convert from Montgomery form: result = a * 1 * R^-1 mod mod
sneppx_montgomery_from_mont PROC
    push rbx
    sub rsp, 64
    lfence
    mov r10, rcx
    mov r11, rdx
    mov rbx, rsp
    mov qword ptr [rbx], 1
    xor rax, rax
    mov rcx, r9
    lea rdi, [rbx + 8]
mont_from_mont_loop:
    cmp rax, rcx
    jae mont_from_mont_done
    mov qword ptr [rdi + rax*8], 0
    inc rax
    jmp mont_from_mont_loop
mont_from_mont_done:
    mov rcx, r10
    mov rdx, r11
    mov r8, r9
    lea r9, [rsp]
    mov qword ptr [rsp + 64], r8
    mov qword ptr [rsp + 72], r9
    mov rax, qword ptr [rsp + 88]
    mov qword ptr [rsp + 80], rax
    call sneppx_montgomery_mul
    lea rdi, [rsp]
    mov rcx, 8
    xor eax, eax
    rep stosq
    lfence
    add rsp, 64
    pop rbx
    ret
sneppx_montgomery_from_mont ENDP

END
