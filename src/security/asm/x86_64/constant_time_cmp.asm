; ARIX-Algo constant-time comparison — x86-64
; Compares two byte arrays, returns 0 if equal, non-zero if different
; Same number of instructions regardless of match position
; rcx = a, rdx = b, r8 = len

.code
arix_ct_compare_asm PROC
    xor     rax, rax
    xor     r9, r9
    xor     r10, r10
loop_start:
    cmp     r9, r8
    jae     loop_end
    movzx   r10d, BYTE PTR [rcx + r9]
    xor     r10b, BYTE PTR [rdx + r9]
    or      eax, r10d
    inc     r9
    jmp     loop_start
loop_end:
    ret
arix_ct_compare_asm ENDP
END
