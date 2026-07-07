; SSE2-optimized Poly1305 MAC
; MASM x64 syntax

.data
    align 16
    p1305_mask dd 0fffffffh, 0ffffffch, 0fffffffh, 0ffffffch
    align 16
    p1305_rr dq 5, 0
    align 16
    p1305_shift dd 26, 26, 26, 26
    align 16
    p1305_26 dd 67108863, 67108863, 67108863, 67108863

.code

; void poly1305_sse_mac(uint8_t mac[16], const uint8_t *msg, size_t len, const uint8_t key[32])
poly1305_sse_mac PROC
    push rbx
    push r12
    push r13
    mov rbx, rcx
    mov r12, rdx
    mov r13, r8
    mov r10, [rsp + 56]

    ; Load key (r, s)
    movdqu xmm5, xmmword ptr [r10]
    movdqu xmm6, xmmword ptr [r10 + 16]

    ; Clamp r
    pand xmm5, xmmword ptr [p1305_mask]

    ; Initialize accumulator
    pxor xmm0, xmm0

    mov r11, r13
poly_loop:
    cmp r11, 16
    jl partial_block

    movdqu xmm1, xmmword ptr [r12]
    por xmm1, xmmword ptr [one]
    call poly1305_mul
    add r12, 16
    sub r11, 16
    jmp poly_loop

partial_block:
    test r11, r11
    jz poly_done
    movdqu xmm1, xmmword ptr [r12]
    mov r14d, 256
    sub r14d, r11d
    shl r14d, 3
    movd xmm2, r14d
    pslldq xmm2, 12
    por xmm1, xmm2
    por xmm1, xmmword ptr [one]
    call poly1305_mul

poly_done:
    ; Add s
    paddq xmm0, xmm6
    movdqu xmmword ptr [rbx], xmm0

    pop r13
    pop r12
    pop rbx
    ret

poly1305_mul PROC
    ; Multiply and reduce modulo 2^130-5
    movdqa xmm7, xmm0
    movdqa xmm2, xmm5

    ; partial products
    pmuludq xmm0, xmm2
    pshufd xmm3, xmm0, 0b01001110
    pmuludq xmm3, xmm2

    pshufd xmm4, xmm1, 0b01001110
    pmuludq xmm4, xmm2
    pshufd xmm2, xmm4, 0b01001110
    pmuludq xmm2, xmm5

    paddq xmm0, xmm4
    paddq xmm3, xmm2

    ; Reduction
    movdqa xmm4, xmm3
    psrlq xmm4, 26
    psllq xmm3, 38
    psrlq xmm3, 38
    paddq xmm0, xmm4

    movdqa xmm2, xmm0
    psrlq xmm2, 26
    pand xmm0, xmmword ptr [p1305_26]
    psllq xmm2, 12
    paddq xmm0, xmm2

    movdqa xmm2, xmm3
    psrlq xmm2, 26
    pand xmm3, xmmword ptr [p1305_26]
    paddq xmm0, xmm2

    ; Final carry
    movdqa xmm2, xmm0
    psrlq xmm2, 26
    pand xmm0, xmmword ptr [p1305_26]
    paddq xmm3, xmm2

    movdqa xmm2, xmm3
    psrlq xmm2, 26
    pand xmm3, xmmword ptr [p1305_26]
    psllq xmm2, 26
    paddq xmm0, xmm2

    ret
poly1305_mul ENDP

.data
    align 16
    one dq 1, 0

END
