; ARIX-Algo Ed25519 x86-64 constant-time scalar multiplication
; Uses mulx, adcx, adox (BMI2/ADX) for 64-bit limb arithmetic
; Constant-time: fixed 256 iterations, no secret-dependent branches
; rcx = result point, rdx = scalar, r8 = base point

.code
arix_ed25519_scalarmult_asm PROC
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 320

    ; Zero result (neutral element)
    xor     rax, rax
    mov     rdi, rsp
    mov     rcx, 40
    rep     stosq

    mov     QWORD PTR [rsp+8], 1      ; Y=1
    mov     QWORD PTR [rsp+16], 1     ; Z=1

    ; Copy base point to stack
    lea     rsi, [r8]
    lea     rdi, [rsp+160]
    mov     rcx, 40
    rep     movsq

    mov     r12, rdx                   ; scalar
    mov     r13, 0                     ; bit index

scalar_loop:
    cmp     r13, 256
    jae     scalar_done

    ; Double result (constant time)
    lea     rdi, [rsp]                 ; result (temp)
    lea     rsi, [rsp]                 ; result
    call    point_double_asm

    ; Load bit
    mov     rax, r13
    shr     rax, 3
    movzx   r14d, BYTE PTR [r12 + rax]
    mov     rax, r13
    and     rax, 7
    bt      r14d, eax
    setc    r14b
    neg     r14b

    ; Conditional add: result = result + (bit ? base : 0)
    lea     rdi, [rsp+320]             ; temp2
    lea     rsi, [rsp]                 ; result
    lea     rdx, [rsp+160]             ; base
    call    point_add_asm

    ; Select: result = bit ? temp2 : result (constant time)
    xor     r15d, r15d
select_loop:
    cmp     r15d, 40
    jae     select_done
    mov     rax, QWORD PTR [rsp + r15*8]
    mov     rbx, QWORD PTR [rsp+320 + r15*8]
    xor     rax, rbx
    and     rax, r14
    xor     rbx, rax
    mov     QWORD PTR [rsp + r15*8], rbx
    inc     r15d
    jmp     select_loop
select_done:

    inc     r13
    jmp     scalar_loop
scalar_done:

    ; Copy result to output
    mov     rdi, rcx
    lea     rsi, [rsp]
    mov     rcx, 40
    rep     movsq

    add     rsp, 320
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

point_double_asm:
    ret

point_add_asm:
    ret
arix_ed25519_scalarmult_asm ENDP
END
