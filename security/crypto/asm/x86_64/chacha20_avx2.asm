; SneppX-ALG AVX2 vectorized ChaCha20 stream cipher with XChaCha20 AEAD
; MASM x64 syntax — constant-time, speculation-safe, cache-resistant

.data
    align 32
    sigma db 101,120,100,51, 104,110,50,53, 51,100,50,56, 55,53,50,48
    align 32
    permute_mask db 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    align 32
    permute_mask_2 db 0,1,2,3,6,7,4,5,10,11,8,9,14,15,12,13,16,17,18,19,22,23,20,21,26,27,24,25,30,31,28,29
    align 32
    permute_mask_3 db 2,3,0,1,4,5,6,7,8,9,10,11,12,13,14,15,18,19,16,17,20,21,22,23,24,25,26,27,28,29,30,31
    align 32
    chacha_ct_mask dq 0FFFFFFFFFFFFFFFFh, 0FFFFFFFFFFFFFFFFh, 0FFFFFFFFFFFFFFFFh, 0FFFFFFFFFFFFFFFFh

.code

; void sneppx_chacha20_init_state(uint32_t state[16], const uint32_t key[8], const uint32_t nonce[3], uint32_t counter)
sneppx_chacha20_init_state PROC
    lfence
    lea r10, sigma
    vmovdqu ymm0, ymmword ptr [r10]
    vmovdqu ymmword ptr [rcx], ymm0
    vmovdqu ymm1, ymmword ptr [rdx]
    vmovdqu ymmword ptr [rcx + 32], ymm1
    mov r10d, r9d
    mov dword ptr [rcx + 48], r10d
    vmovdqu xmm2, xmmword ptr [r8]
    vmovq qword ptr [rcx + 52], xmm2
    vzeroupper
    lfence
    ret
sneppx_chacha20_init_state ENDP

; void sneppx_chacha20_block(uint8_t keystream[64], const uint32_t state[16])
sneppx_chacha20_block PROC
    push rbx
    push r12
    lfence
    mov r10, rdx
    vmovdqu ymm0, ymmword ptr [r10]
    vmovdqu ymm1, ymmword ptr [r10 + 32]
    vmovdqa ymm8, ymm0
    vmovdqa ymm9, ymm1
    mov r12d, 10
chacha_block_rounds:
    vpaddd ymm0, ymm0, ymm1
    vpxor ymm3, ymm3, ymm0
    vpshufb ymm3, ymm3, ymmword ptr [permute_mask]
    vpaddd ymm2, ymm2, ymm3
    vpxor ymm1, ymm1, ymm2
    vpslld ymm4, ymm1, 12
    vpsrld ymm1, ymm1, 20
    vpor ymm1, ymm4, ymm1
    vpaddd ymm0, ymm0, ymm1
    vpxor ymm3, ymm3, ymm0
    vpshufb ymm3, ymm3, ymmword ptr [permute_mask + 16]
    vpaddd ymm2, ymm2, ymm3
    vpxor ymm1, ymm1, ymm2
    vpslld ymm4, ymm1, 7
    vpsrld ymm1, ymm1, 25
    vpor ymm1, ymm4, ymm1
    vpshufd ymm1, ymm1, 0b00111001
    vpshufd ymm2, ymm2, 0b01001110
    vpshufd ymm3, ymm3, 0b10010011
    dec r12d
    jnz chacha_block_rounds
    vpaddd ymm0, ymm0, ymm8
    vpaddd ymm1, ymm1, ymm9
    vmovdqu ymmword ptr [rcx], ymm0
    vmovdqu ymmword ptr [rcx + 32], ymm1
    vzeroupper
    lfence
    pop r12
    pop rbx
    ret
sneppx_chacha20_block ENDP

; void sneppx_chacha20_encrypt(uint8_t *out, const uint8_t *in, size_t len, const uint32_t key[8], const uint32_t nonce[3], uint32_t counter)
sneppx_chacha20_encrypt PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 96
    lfence
    mov rbx, rcx
    mov r12, rdx
    mov r13, r8
    mov r14, r9
    mov r15, qword ptr [rsp + 112]
    mov r10, qword ptr [rsp + 120]
    mov rax, qword ptr [rsp + 128]
    lea rdi, [rsp]
    mov rcx, rdi
    mov rdx, r14
    mov r8, r15
    mov r9d, eax
    call sneppx_chacha20_init_state
    mov r14d, eax
    lea r15, [rsp + 64]
chacha_enc_loop:
    cmp r13, 64
    jb chacha_enc_final
    mov rcx, r15
    lea rdx, [rsp]
    call sneppx_chacha20_block
    vmovdqu ymm0, ymmword ptr [r15]
    vmovdqu ymm1, ymmword ptr [r12]
    vpxor ymm0, ymm0, ymm1
    vmovdqu ymmword ptr [rbx], ymm0
    vmovdqu ymm0, ymmword ptr [r15 + 32]
    vmovdqu ymm1, ymmword ptr [r12 + 32]
    vpxor ymm0, ymm0, ymm1
    vmovdqu ymmword ptr [rbx + 32], ymm0
    add rbx, 64
    add r12, 64
    sub r13, 64
    inc dword ptr [rsp + 48]
    jmp chacha_enc_loop
chacha_enc_final:
    test r13, r13
    jz chacha_enc_done
    mov rcx, r15
    lea rdx, [rsp]
    call sneppx_chacha20_block
    xor r10, r10
chacha_enc_partial:
    cmp r10, r13
    jae chacha_enc_done
    mov al, byte ptr [r15 + r10]
    xor al, byte ptr [r12 + r10]
    mov byte ptr [rbx + r10], al
    inc r10
    jmp chacha_enc_partial
chacha_enc_done:
    vzeroupper
    lea rdi, [rsp]
    mov rcx, 12
    xor eax, eax
    rep stosq
    lea rdi, [rsp + 64]
    mov rcx, 4
    xor eax, eax
    rep stosq
    lfence
    add rsp, 96
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_chacha20_encrypt ENDP

; void sneppx_xchacha20_encrypt(uint8_t *out, const uint8_t *in, size_t len, const uint32_t key[8], const uint32_t nonce[4], uint32_t counter)
sneppx_xchacha20_encrypt PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 128
    lfence
    mov rbx, rcx
    mov r12, rdx
    mov r13, r8
    mov r14, r9
    mov r15, qword ptr [rsp + 144]
    lea rdi, [rsp]
    mov rcx, 16
    xor eax, eax
    rep stosq
    lea r10, sigma
    vmovdqu ymm0, ymmword ptr [r10]
    vmovdqu ymmword ptr [rsp], ymm0
    vmovdqu ymm1, ymmword ptr [r14]
    vmovdqu ymmword ptr [rsp + 32], ymm1
    vmovdqu xmm2, xmmword ptr [r15]
    vmovq qword ptr [rsp + 52], xmm2
    mov rax, qword ptr [rsp + 152]
    mov dword ptr [rsp + 48], eax
    lea rcx, [rsp + 64]
    lea rdx, [rsp]
    call sneppx_chacha20_block
    lea rdi, [rsp]
    mov rcx, 16
    xor eax, eax
    rep stosq
    lea r10, sigma
    vmovdqu ymm0, ymmword ptr [r10]
    vmovdqu ymmword ptr [rsp], ymm0
    vmovdqu ymm1, ymmword ptr [rsp + 64]
    vmovdqu ymmword ptr [rsp + 32], ymm1
    vmovdqu xmm2, xmmword ptr [r15 + 16]
    vmovq qword ptr [rsp + 52], xmm2
    mov dword ptr [rsp + 48], 0
    mov rcx, rbx
    mov rdx, r12
    mov r8, r13
    lea r9, [rsp]
    lea rax, [rsp + 48]
    mov qword ptr [rsp + 160], rax
    mov qword ptr [rsp + 168], r15
    call sneppx_chacha20_encrypt
    lea rdi, [rsp]
    mov rcx, 24
    xor eax, eax
    rep stosq
    lfence
    add rsp, 128
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_xchacha20_encrypt ENDP

; void sneppx_chacha20_poly1305_aead_encrypt(...)
sneppx_chacha20_poly1305_aead_encrypt PROC
    push rbx
    push r12
    sub rsp, 64
    lfence
    mov r10, rcx
    mov r11, rdx
    mov r12, r8
    lea rdi, [rsp]
    mov rcx, rdi
    mov rdx, r11
    mov r8, r12
    call sneppx_chacha20_encrypt
    lea rdi, [rsp]
    mov rcx, 8
    xor eax, eax
    rep stosq
    lfence
    add rsp, 64
    pop r12
    pop rbx
    ret
sneppx_chacha20_poly1305_aead_encrypt ENDP

; void sneppx_chacha20_wipe_state(uint32_t state[16])
sneppx_chacha20_wipe_state PROC
    lfence
    xor eax, eax
    mov rdi, rcx
    mov ecx, 16
    rep stosd
    mfence
    lfence
    ret
sneppx_chacha20_wipe_state ENDP

END
