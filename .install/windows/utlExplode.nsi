; String exploding. Source: https://nsis.sourceforge.io/Explode

!include LogicLib.nsh

!define Explode "!insertmacro Explode"

!macro  Explode Length  Separator   String
    push    `${Separator}`
    push    `${String}`
    call    Explode
    pop     `${Length}`
!macroend

Function Explode
  ; Initialize variables
  var /GLOBAL explString
  var /GLOBAL explSeparator
  var /GLOBAL explStrLen
  var /GLOBAL explSepLen
  var /GLOBAL explOffset
  var /GLOBAL explTmp
  var /GLOBAL explTmp2
  var /GLOBAL explTmp3
  var /GLOBAL explArrCount

  ; Get input from user
  pop $explString
  pop $explSeparator

  ; Calculates initial values
  strLen $explStrLen $explString
  strLen $explSepLen $explSeparator
  strCpy $explArrCount 1

  ${If}   $explStrLen <= 1          ;   If we got a single character
  ${OrIf} $explSepLen > $explStrLen ;   or separator is larger than the string,
    push    $explString             ;   then we return initial string with no change
    push    1                       ;   and set array's length to 1
    return
  ${EndIf}

  ; Set offset to the last symbol of the string
  strCpy $explOffset $explStrLen
  IntOp  $explOffset $explOffset - 1

  ; Clear temp string to exclude the possibility of appearance of occasional data
  strCpy $explTmp   ""
  strCpy $explTmp2  ""
  strCpy $explTmp3  ""
 
  ; Loop until the offset becomes negative
  ${Do}
    ;   If offset becomes negative, it is time to leave the function
    ${IfThen} $explOffset == -1 ${|} ${ExitDo} ${|}

    ;   Remove everything before and after the searched part ("TempStr")
    strCpy $explTmp $explString $explSepLen $explOffset

    ${If} $explTmp == $explSeparator
        ;   Calculating offset to start copy from
        IntOp   $explTmp2 $explOffset + $explSepLen ;   Offset equals to the current offset plus length of separator
        strCpy  $explTmp3 $explString "" $explTmp2

        push    $explTmp3                           ;   Throwing array item to the stack
        IntOp   $explArrCount $explArrCount + 1     ;   Increasing array's counter

        strCpy  $explString $explString $explOffset 0   ;   Cutting all characters beginning with the separator entry
        strLen  $explStrLen $explString
    ${EndIf}

    ${If} $explOffset = 0                       ;   If the beginning of the line met and there is no separator,
                                                ;   copying the rest of the string
        ${If} $explSeparator == ""              ;   Fix for the empty separator
            IntOp   $explArrCount   $explArrCount - 1
        ${Else}
            push    $explString
        ${EndIf}
    ${EndIf}

    IntOp   $explOffset $explOffset - 1
  ${Loop}

  push $explArrCount
FunctionEnd