; SHA-NI accelerated SHA-256 operations
; MASM x64 syntax

.data
    align 16
    K256 dd 0428a2f98h, 071374491h, 0b5c0fbcfh, 0e9b5dba5h
         dd 03956c25bh, 059f111f1h, 0923f82a4h, 0ab1c5ed5h
         dd 0d807aa98h, 012835b01h, 02431855eh, 0550c7dc3h
         dd 072be5d74h, 080deb1feh, 09bdc06a7h, 0c19bf174h
         dd 0e49b69c1h, 0efbe4786h, 00fc19dc6h, 0240ca1cch
         dd 02de92c6fh, 04a7484aah, 05cb0a9dch, 076f988dah
         dd 0983e5152h, 0a831c66dh, 0b00327c8h, 0bf597fc7h
         dd 0c6e00bf3h, 0d5a79147h, 006ca6351h, 014292967h
         dd 027b70a85h, 02e1b2138h, 04d2c6dfch, 053380d13h
         dd 0650a7354h, 0766a0abbh, 081c2c92eh, 092722c85h
         dd 0a2bfe8a1h, 0a81a664bh, 0c24b8b70h, 0c76c51a3h
         dd 0d192e819h, 0d6990624h, 0f40e3585h, 0106aa070h
         dd 019a4c116h, 01e376c08h, 02748774ch, 034b0bcb5h
         dd 0391c0cb3h, 04ed8aa4ah, 05b9cca4fh, 0682e6ff3h
         dd 0748f82eeh, 078a5636fh, 084c87814h, 08cc70208h
         dd 090befffah, 0a4506cebh, 0bef9a3f7h, 0c67178f2h

    align 16
    PSHUFFLE_BYTE_FLIP_MASK db 3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12

.code

; void sha_ni_transform(uint32_t state[8], const uint8_t block[64])
sha_ni_transform PROC
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rcx + 16]
    movdqu xmm2, xmmword ptr [rcx + 32]
    movdqu xmm3, xmmword ptr [rcx + 48]

    movdqu xmm4, xmmword ptr [rdx]
    movdqu xmm5, xmmword ptr [rdx + 16]
    movdqu xmm6, xmmword ptr [rdx + 32]
    movdqu xmm7, xmmword ptr [rdx + 48]

    pshufb xmm4, xmmword ptr [PSHUFFLE_BYTE_FLIP_MASK]
    pshufb xmm5, xmmword ptr [PSHUFFLE_BYTE_FLIP_MASK]
    pshufb xmm6, xmmword ptr [PSHUFFLE_BYTE_FLIP_MASK]
    pshufb xmm7, xmmword ptr [PSHUFFLE_BYTE_FLIP_MASK]

    lea r8, K256
    mov r9d, 4
round_loop:
    sha256rnds2 xmm0, xmm1, xmmword ptr [r8]
    sha256rnds2 xmm1, xmm0, xmmword ptr [r8 + 16]
    add r8, 32
    dec r9d
    jnz round_loop

    movdqu xmmword ptr [rcx], xmm0
    movdqu xmmword ptr [rcx + 16], xmm1
    ret
sha_ni_transform ENDP

; void sha_ni_hash(uint8_t out[32], const uint8_t *in, size_t len)
sha_ni_hash PROC
    push rbx
    mov rbx, rcx
    mov r10, rdx
    mov r11, r8

    mov dword ptr [rcx], 06a09e667h
    mov dword ptr [rcx + 4], 0bb67ae85h
    mov dword ptr [rcx + 8], 03c6ef372h
    mov dword ptr [rcx + 12], 0a54ff53ah
    mov dword ptr [rcx + 16], 0510e527fh
    mov dword ptr [rcx + 20], 09b05688ch
    mov dword ptr [rcx + 24], 01f83d9abh
    mov dword ptr [rcx + 28], 05be0cd19h

hash_loop:
    cmp r11, 64
    jl hash_done
    sub r11, 64
    call sha_ni_transform
    add r10, 64
    jmp hash_loop
hash_done:
    pop rbx
    ret
sha_ni_hash ENDP

END
