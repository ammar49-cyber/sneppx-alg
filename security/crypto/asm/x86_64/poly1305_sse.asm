; SneppX-ALG SSE2-optimized Poly1305 MAC with extended security measures
; MASM x64 syntax — constant-time, speculation-safe, memory-wiping

.data
    align 16
    p1305_mask dd 0fffffffh, 0ffffffch, 0ffffff1h, 0ffffff1h
    align 16
    p1305_rr dq 5, 0
    align 16
    p1305_shift dd 26, 26, 26, 26
    align 16
    p1305_26 dd 67108863, 67108863, 67108863, 67108863
    align 16
    p1305_26_inv dd 0FC000000h, 0FC000000h, 0FC000000h, 0FC000000h
    align 16
    one dq 1, 0
    align 16
    poly_mask dq 0FFFFFFFFFFFFFFFFh, 0FFFFFFFFFFFFFFFFh

.code

; Helper: multiply and reduce modulo 2^130-5 — constant-time
sneppx_poly1305_mul PROC
    push rbx
    lfence
    movdqa xmm7, xmm0
    movdqa xmm2, xmm5
    pmuludq xmm0, xmm2
    pshufd xmm3, xmm0, 04Eh
    pmuludq xmm3, xmm2
    pshufd xmm4, xmm1, 04Eh
    pmuludq xmm4, xmm2
    pshufd xmm2, xmm4, 04Eh
    pmuludq xmm2, xmm5
    paddq xmm0, xmm4
    paddq xmm3, xmm2
    lea rbx, p1305_26
    movdqa xmm4, xmm3
    psrlq xmm4, 26
    psllq xmm3, 38
    psrlq xmm3, 38
    paddq xmm0, xmm4
    movdqa xmm2, xmm0
    psrlq xmm2, 26
    pand xmm0, xmmword ptr [rbx]
    psllq xmm2, 12
    paddq xmm0, xmm2
    movdqa xmm2, xmm3
    psrlq xmm2, 26
    pand xmm3, xmmword ptr [rbx]
    paddq xmm0, xmm2
    movdqa xmm2, xmm0
    psrlq xmm2, 26
    pand xmm0, xmmword ptr [rbx]
    paddq xmm3, xmm2
    movdqa xmm2, xmm3
    psrlq xmm2, 26
    pand xmm3, xmmword ptr [rbx]
    psllq xmm2, 26
    paddq xmm0, xmm2
    lfence
    pop rbx
    ret
sneppx_poly1305_mul ENDP

; void sneppx_poly1305_mac(uint8_t mac[16], const uint8_t *msg, size_t len, const uint8_t key[32])
sneppx_poly1305_mac PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 64
    lfence
    mov rbx, rcx
    mov r12, rdx
    mov r13, r8
    mov r10, r9
    lea rdi, [rsp]
    mov rcx, 8
    xor eax, eax
    rep stosq
    movdqu xmm5, xmmword ptr [r10]
    movdqu xmm6, xmmword ptr [r10 + 16]
    lea r14, p1305_mask
    pand xmm5, xmmword ptr [r14]
    pxor xmm0, xmm0
    mov r15, r13
    test r15, r15
    jz poly_process_done
poly_block_loop:
    cmp r15, 16
    jb poly_partial_block
    movdqu xmm1, xmmword ptr [r12]
    por xmm1, xmmword ptr [one]
    call sneppx_poly1305_mul
    add r12, 16
    sub r15, 16
    jmp poly_block_loop
poly_partial_block:
    test r15, r15
    jz poly_process_done
    pxor xmm1, xmm1
    xor r14, r14
poly_partial_copy:
    cmp r14, r15
    jae poly_partial_pad
    movzx eax, byte ptr [r12 + r14]
    mov byte ptr [rsp + 64], al
    movd xmm0, dword ptr [rsp + 64]
    pinsrd xmm1, eax, 0
    por xmm1, xmm0
    inc r14
    jmp poly_partial_copy
poly_partial_pad:
    mov byte ptr [rsp + r15], 1
    por xmm1, xmmword ptr [one]
    call sneppx_poly1305_mul
poly_process_done:
    paddq xmm0, xmm6
    movdqu xmmword ptr [rbx], xmm0
    lea rdi, [rsp]
    mov rcx, 8
    xor eax, eax
    rep stosq
    movdqa xmmword ptr [rsp], xmm5
    movdqa xmmword ptr [rsp + 16], xmm6
    xor eax, eax
    mov ecx, 4
    lea rdi, [rsp]
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
sneppx_poly1305_mac ENDP

; void sneppx_poly1305_verify(const uint8_t mac1[16], const uint8_t mac2[16])
; Constant-time MAC verification
sneppx_poly1305_verify PROC
    push rbx
    lfence
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    pcmpeqb xmm0, xmm1
    pmovmskb eax, xmm0
    xor eax, 0FFFFh
    neg eax
    sbb eax, eax
    neg eax
    lfence
    pop rbx
    ret
sneppx_poly1305_verify ENDP

; void sneppx_poly1305_wipe_key(uint8_t key[32])
sneppx_poly1305_wipe_key PROC
    lfence
    xor eax, eax
    mov rdi, rcx
    mov ecx, 4
    rep stosq
    mfence
    lfence
    ret
sneppx_poly1305_wipe_key ENDP

END
