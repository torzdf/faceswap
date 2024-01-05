; Custom string manipulation functions for faceswap installer


Function CheckForSpaces
; Check a string for space
; Usage:
;   push <string>
;   call CheckForSpaces
;   pop $0
    Exch $R0
    push $R1
    push $R2
    push $R3
    strCpy $R1 -1
    strCpy $R3 $R0
    strCpy $R0 0
        loop:
        strCpy $R2 $R3 1 $R1
        IntOp $R1 $R1 - 1
        strCmp $R2 "" done
        strCmp $R2 " " 0 loop
        IntOp $R0 $R0 + 1
    Goto loop
    done:
    pop $R3
    pop $R2
    pop $R1
    Exch $R0

FunctionEnd


; First item from array
!define GetFirst "!insertmacro GetFirst"

!macro  GetFirst Word Separator String
    ; Get the first or only instance from a string separated by the given separator
    ; usage:
    ;   ${GetFirst} $0 <separator> <string>
    push    "${Separator}"
    push    "${String}"
    call    _GetFirst
    pop     "${Word}"
!macroend

Function _GetFirst
    pop $R0                         ; string
    pop $R1                         ; separator

    strCpy $R2 -1                   ; Character index
    strCpy $R4 ""                   ; Return value

    loop:
        IntOp $R2 $R2 + 1           ; Current position
        strCpy $R3 $R0 1 $R2        ; Read 1 character from $R0 into $R3 at position $R2
        strCmp $R3 $R1 stop         ; End if match found else continue
        strCmp $R3 "" stop          ; End if no more characters or continue
        goto loop
    stop:
        strCpy $R4 $R0 $R2 0        ; Read the discovered word into $R4

    push $R4

FunctionEnd


; Item is in array
!define ItemInArray "!insertmacro ItemInArray"

!macro  ItemInArray Bool Separator Array String
    ; Returns 1 if the string is in the given array otherwise 0
    ; usage:
    ;   ${ItemInArray} $0 <separator> <array> <string>
    push    "${Separator}"
    push    "${Array}"
    push    "${String}"
    call    _ItemInArray
    pop     "${Bool}"
!macroend

Function _ItemInArray
    pop $R0                     ; string
    pop $R1                     ; array
    pop $R2                     ; separator

    strCpy $R3 0                ; Return value

    ${Explode} $R4 $R2 $R1
         ${For} $R5 1 $R4
            pop $R6             ; Word
            ${if} $R3 == 1      ; Match already found
                ${Continue}     ; Have to continue to clear the stack
            ${EndIf}
            ${If} $R6 == $R0    ; This row item matches
                strCpy $R3 1
                ${Continue}     ; Have to continue to clear the stack
            ${EndIf}
        ${Next}

    push $R3

FunctionEnd


; StrContains
; Ref: https://nsis.sourceforge.io/StrContains
; This function does a case sensitive searches for an occurrence of a substring in a string.
; It returns the substring if it is found.
; Otherwise it returns null("").
; Written by kenglish_hi
; Adapted from StrReplace written by dandaman32


var STR_HAYSTACK
var STR_NEEDLE
var STR_CONTAINS_VAR_1
var STR_CONTAINS_VAR_2
var STR_CONTAINS_VAR_3
var STR_CONTAINS_VAR_4
var STR_RETURN_VAR

Function StrContains
  Exch $STR_NEEDLE
  Exch 1
  Exch $STR_HAYSTACK
  ; Uncomment to debug
  ;MessageBox MB_OK 'STR_NEEDLE = $STR_NEEDLE STR_HAYSTACK = $STR_HAYSTACK '
    strCpy $STR_RETURN_VAR ""
    strCpy $STR_CONTAINS_VAR_1 -1
    strLen $STR_CONTAINS_VAR_2 $STR_NEEDLE
    strLen $STR_CONTAINS_VAR_4 $STR_HAYSTACK
    loop:
      IntOp $STR_CONTAINS_VAR_1 $STR_CONTAINS_VAR_1 + 1
      strCpy $STR_CONTAINS_VAR_3 $STR_HAYSTACK $STR_CONTAINS_VAR_2 $STR_CONTAINS_VAR_1
      strCmp $STR_CONTAINS_VAR_3 $STR_NEEDLE found
      strCmp $STR_CONTAINS_VAR_1 $STR_CONTAINS_VAR_4 done
      Goto loop
    found:
      strCpy $STR_RETURN_VAR $STR_NEEDLE
      Goto done
    done:
   pop $STR_NEEDLE ;Prevent "invalid opcode" errors and keep the
   Exch $STR_RETURN_VAR
FunctionEnd

!macro _StrContainsConstructor OUT NEEDLE HAYSTACK
  push `${HAYSTACK}`
  push `${NEEDLE}`
  call StrContains
  pop `${OUT}`
!macroend

!define StrContains '!insertmacro "_StrContainsConstructor"'
