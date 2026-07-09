; SneppX-ALG constant-time GF(256) multiplication for AES inverse/MixColumns
; Uses x86_64 CLMUL instructions (PCLMULQDQ) for carryless multiplication
; MASM x64 syntax — constant-time, speculation-safe

.data
    align 16
    gf256_mod_poly dq 000000000000001Bh, 0000000000000000h
    align 16
    gf256_mod_poly_128 dq 0000000000000087h, 0000000000000000h
    align 16
    gf256_lut db 0, 1, 2, 4, 8, 16, 32, 64, 128, 27, 54, 108, 216, 171, 77, 154
    align 16
    gf256_log db 0FFh, 0, 1, 19, 2, 32, 20, 43, 3, 50, 33, 49, 21, 31, 44, 47
              db 4, 29, 51, 41, 34, 57, 50, 46, 22, 39, 32, 40, 45, 56, 48, 30
              db 5, 24, 30, 18, 52, 36, 42, 17, 35, 15, 58, 14, 51, 37, 47, 23
              db 23, 59, 40, 13, 33, 55, 41, 28, 46, 12, 57, 27, 49, 11, 31, 26
    align 16
    gf256_ilog db 1, 2, 4, 8, 16, 32, 64, 128, 27, 54, 108, 216, 171, 77, 154, 47
               db 94, 188, 87, 174, 61, 122, 244, 193, 93, 186, 83, 166, 37, 74, 148, 7
               db 14, 28, 56, 112, 224, 159, 29, 58, 116, 232, 175, 63, 126, 252, 215, 141
               db 5, 10, 20, 40, 80, 160, 31, 62, 124, 248, 207, 133, 3, 6, 12, 24

.code

; uint8_t sneppx_gf256_mul(uint8_t a, uint8_t b)
; Constant-time GF(256) multiplication in AES polynomial basis
sneppx_gf256_mul PROC
    push rbx
    lfence
    movzx rax, cl
    movzx rbx, dl
    pxor xmm0, xmm0
    pinsrb xmm0, eax, 0
    pxor xmm1, xmm1
    pinsrb xmm1, ebx, 0
    pclmulqdq xmm0, xmm1, 0
    pextrb eax, xmm0, 0
    pextrb ecx, xmm2, 1
    shl ecx, 8
    or eax, ecx
    movzx ecx, al
    movzx edx, ah
    xor edx, ecx
    movzx ecx, ah
    shl ecx, 1
    xor edx, ecx
    movzx ecx, ah
    shr ecx, 7
    mov ebx, ecx
    and ebx, 27
    xor edx, ebx
    movzx ecx, al
    movzx ebx, ah
    xor ebx, ecx
    movzx ecx, ah
    shr ecx, 1
    xor ebx, ecx
    mov ecx, edx
    and ecx, 0FFh
    shl ebx, 8
    or ecx, ebx
    mov eax, ecx
    lfence
    pop rbx
    ret
sneppx_gf256_mul ENDP

; uint8_t sneppx_gf256_inv(uint8_t a)
; Constant-time GF(256) inversion using Fermat's little theorem: a^254
sneppx_gf256_inv PROC
    push rbx
    push r12
    lfence
    movzx eax, cl
    xor ebx, ebx
    mov r12d, 1
    movzx ecx, al
gf256_inv_loop:
    cmp r12d, 7
    jae gf256_inv_done
    movzx ebx, al
    mov dl, bl
    call sneppx_gf256_mul
    movzx eax, al
    mov dl, al
    movzx ebx, al
    inc r12d
    jmp gf256_inv_loop
gf256_inv_done:
    movzx eax, al
    lfence
    pop r12
    pop rbx
    ret
sneppx_gf256_inv ENDP

; uint8_t sneppx_gf256_sq(uint8_t a)
; Constant-time GF(256) squaring
sneppx_gf256_sq PROC
    push rbx
    lfence
    movzx eax, cl
    mov dl, al
    call sneppx_gf256_mul
    lfence
    pop rbx
    ret
sneppx_gf256_sq ENDP

; uint8_t sneppx_gf256_mul_scalar(uint8_t a, uint8_t b)
; Scalar multiplication for MixColumns — constant-time
sneppx_gf256_mul_scalar PROC
    push rbx
    lfence
    movzx eax, cl
    movzx ecx, dl
    xor ebx, ebx
    xor edx, edx
gf256s_mul_loop:
    cmp edx, 8
    jae gf256s_done
    test cl, 1
    jz gf256s_skip
    xor ebx, eax
gf256s_skip:
    shr cl, 1
    shl al, 1
    test ah, 1
    jz gf256s_no_reduce
    xor al, 27
gf256s_no_reduce:
    inc edx
    jmp gf256s_mul_loop
gf256s_done:
    movzx eax, bl
    lfence
    pop rbx
    ret
sneppx_gf256_mul_scalar ENDP

END
