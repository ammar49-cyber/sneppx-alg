; AVX2 vectorized ChaCha20 stream cipher
; MASM x64 syntax

.data
    align 32
    sigma dd 1634760805, 857760878, 2036477234, 1797285236
    align 32
    permute_mask db 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

.code

; void chacha20_avx2_encrypt(uint8_t *out, const uint8_t *in, size_t len, const uint32_t key[8], const uint32_t nonce[3], uint32_t counter)
chacha20_avx2_encrypt PROC
    push rbx
    push r12
    push r13
    mov rbx, rcx
    mov r12, rdx
    mov r13, r8

    vzeroupper
    mov r10, [rsp + 56]
    mov r11, [rsp + 64]

chacha_loop:
    cmp r13, 64
    jl chacha_done

    vmovdqu ymm0, ymmword ptr [sigma]
    vmovdqu ymm1, ymmword ptr [r10]
    vmovd ymm2, dword ptr [r11]
    vmovdqu ymm3, ymmword ptr [r11 + 4]

    vmovdqa ymm8, ymm0
    vmovdqa ymm9, ymm1
    vmovdqa ymm10, ymm2
    vmovdqa ymm11, ymm3

    mov r14d, 10
quarter_round_loop:
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

    vpshufd ymm1, ymm1, 0b10010011
    vpshufd ymm2, ymm2, 0b01001110
    vpshufd ymm3, ymm3, 0b00111001

    dec r14d
    jnz quarter_round_loop

    vpaddd ymm0, ymm0, ymm8
    vpaddd ymm1, ymm1, ymm9
    vpaddd ymm2, ymm2, ymm10
    vpaddd ymm3, ymm3, ymm11

    vmovdqu ymm4, ymmword ptr [r12]
    vpxor ymm0, ymm0, ymm4
    vmovdqu ymmword ptr [rbx], ymm0

    vmovdqu ymm4, ymmword ptr [r12 + 32]
    vpxor ymm1, ymm1, ymm4
    vmovdqu ymmword ptr [rbx + 32], ymm1

    add rbx, 64
    add r12, 64
    sub r13, 64

    inc dword ptr [r11]
    jmp chacha_loop

chacha_done:
    vzeroupper
    pop r13
    pop r12
    pop rbx
    ret
chacha20_avx2_encrypt ENDP

.data
    align 32
    permute_mask_2 db 0,1,2,3,6,7,4,5,10,11,8,9,14,15,12,13,16,17,18,19,22,23,20,21,26,27,24,25,30,31,28,29
    align 32
    permute_mask_3 db 2,3,0,1,4,5,6,7,8,9,10,11,12,13,14,15,18,19,16,17,20,21,22,23,24,25,26,27,28,29,30,31

END
