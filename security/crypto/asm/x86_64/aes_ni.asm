; AES-NI accelerated AES-256-GCM operations
; MASM x64 syntax

.data
    align 16
    rcon db 01h, 02h, 04h, 08h, 10h, 20h, 40h, 80h, 1bh, 36h
    align 16
    shuf_mask db 15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0

.code

; void aes_ni_encrypt_block(uint8_t out[16], const uint8_t in[16], const uint8_t rk[240], int rounds)
aes_ni_encrypt_block PROC
    movdqa xmm0, xmmword ptr [rdx]
    movdqu xmm1, xmmword ptr [r8]
    pxor xmm0, xmm1
    mov r10, r9
    sub r10, 1
    xor r11, r11
loop_head:
    lea r11, [r11 + 16]
    movdqu xmm1, xmmword ptr [r8 + r11]
    aesenc xmm0, xmm1
    dec r10
    jnz loop_head
    lea r11, [r11 + 16]
    movdqu xmm1, xmmword ptr [r8 + r11]
    aesenclast xmm0, xmm1
    movdqu xmmword ptr [rcx], xmm0
    ret
aes_ni_encrypt_block ENDP

; void aes_ni_expand_key(uint8_t rk[240], const uint8_t key[32])
aes_ni_expand_key PROC
    push rsi
    push rdi
    mov rsi, rdx
    mov rdi, rcx
    movdqu xmm1, xmmword ptr [rsi]
    movdqu xmm2, xmmword ptr [rsi + 16]
    movdqu xmmword ptr [rdi], xmm1
    movdqu xmmword ptr [rdi + 16], xmm2
    mov r8d, 8
    lea r9, rcon
exp_loop:
    aeskeygenassist xmm3, xmm2, 0
    call shuffle_rcon
    movdqu xmmword ptr [rdi + r8*4], xmm1
    add rdi, 16
    aeskeygenassist xmm2, xmm1, 0
    call shuffle
    movdqu xmmword ptr [rdi + r8*4 - 16], xmm2
    inc r8d
    cmp r8d, 15
    jl exp_loop
    pop rdi
    pop rsi
    ret
shuffle_rcon:
    pshufb xmm3, xmmword ptr [shuf_mask]
    movd eax, xmm3
    movzx ecx, byte ptr [r9]
    xor eax, ecx
    movd xmm3, eax
    pxor xmm1, xmm3
    inc r9
    ret
shuffle:
    pshufb xmm2, xmmword ptr [shuf_mask]
    pxor xmm2, xmm1
    ret
aes_ni_expand_key ENDP

; void aes_ni_gcm_ghash(uint8_t out[16], const uint8_t h[16], const uint8_t *aad, size_t aad_len, const uint8_t *ct, size_t ct_len)
aes_ni_gcm_ghash PROC
    pxor xmm0, xmm0
    movdqu xmm1, xmmword ptr [rdx]
    mov r10, r8
    xor r11, r11
ghash_aad_loop:
    cmp r11, r9
    jge ghash_ct_start
    movdqu xmm2, xmmword ptr [r10 + r11]
    pxor xmm0, xmm2
    call gfmul
    add r11, 16
    jmp ghash_aad_loop
ghash_ct_start:
    mov r10, [rsp + 40]
    mov r11, 0
ghash_ct_loop:
    cmp r11, [rsp + 48]
    jge ghash_done
    movdqu xmm2, xmmword ptr [r10 + r11]
    pxor xmm0, xmm2
    call gfmul
    add r11, 16
    jmp ghash_ct_loop
ghash_done:
    movdqu xmmword ptr [rcx], xmm0
    ret
gfmul:
    pclmulqdq xmm3, xmm1, 0
    pclmulqdq xmm4, xmm1, 1
    pclmulqdq xmm5, xmm1, 16
    pclmulqdq xmm6, xmm1, 17
    pxor xmm5, xmm4
    pxor xmm3, xmm6
    pshufd xmm4, xmm3, 78
    pxor xmm4, xmm3
    pclmulqdq xmm4, xmmword ptr [reduce_poly], 0
    pshufd xmm4, xmm4, 78
    pxor xmm3, xmm4
    movdqa xmm0, xmm3
    ret
aes_ni_gcm_ghash ENDP

.data
    align 16
    reduce_poly dq 0000000000000001h, 0c200000000000000h

.code

; void aes_ni_ctr_encrypt(uint8_t *out, const uint8_t *in, size_t len, const uint8_t rk[240], int rounds, uint8_t iv[16])
aes_ni_ctr_encrypt PROC
    push rbx
    push r12
    mov rbx, rcx
    mov r12, rdx
    movdqu xmm4, xmmword ptr [rsp + 56]
ctr_loop:
    sub r8, 16
    jl ctr_done
    movdqa xmm0, xmm4
    movdqu xmm1, xmmword ptr [r12]
    pxor xmm0, xmm1
    mov r10d, [rsp + 48]
    mov r11, r10
    sub r11, 1
    xor r10, r10
ctr_round:
    lea r10, [r10 + 16]
    movdqu xmm1, xmmword ptr [rsp + 40 + r10]
    aesenc xmm0, xmm1
    dec r11
    jnz ctr_round
    lea r10, [r10 + 16]
    movdqu xmm1, xmmword ptr [rsp + 40 + r10]
    aesenclast xmm0, xmm1
    movdqu xmmword ptr [rbx], xmm0
    add rbx, 16
    add r12, 16
    inc_ctr:
    movd eax, xmm4
    bswap eax
    inc eax
    bswap eax
    movd xmm4, eax
    jmp ctr_loop
ctr_done:
    pop r12
    pop rbx
    ret
aes_ni_ctr_encrypt ENDP

END
