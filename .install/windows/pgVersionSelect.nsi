; Faceswap Windows Installer - [Input] Version Select page
!include "MUI.nsh"
!include nsDialogs.nsh


Function pgVersionSelect
  ; === Version Select Dialog ===

    strCmp $FlgIsResuming 1 0 +3            ; Skip for resuming WSL2 installs
        strCpy $FlgIsResuming 0             ; Reset flag so user can 'Back' to this page
        abort

    nsDialogs::Create 1018
    pop $0
    strCmp $0 error 0 +2
        abort

    ; Disable Next button
    GetDlgItem $0 $HWNDPARENT 1
    EnableWindow $0 0

    !insertmacro MUI_HEADER_TEXT "faceswap Installer" "Select Version"

    ; Intro
    ${NSD_CreateLabel} 8u 2u 292u 20u "faceswap is powered by Tensorflow. Please choose the correct version to install based on your hardware configuration:"

    ; WSL2
    ${NSD_CreateLabel} 103u 31u 186u 20u "Select this option if you have an Nvidia GPU."
    ${NSD_CreateButton} 8u 31u 84u 22u "Nvidia (WSL2)"
        pop $0
        nsDialogs::SetUserData $0 "wsl2"
        ${NSD_OnClick} $0 fnVersionSelectButtonClick

    ; DirectML
    ${NSD_CreateLabel} 103u 57u 186u 20u "Select this option if you have a DirectX 12 supported AMD or Intel GPU."
    ${NSD_CreateButton} 8u 57u 84u 22u "DirectML"
        pop $0
        nsDialogs::SetUserData $0 "directml"
        ${NSD_OnClick} $0 fnVersionSelectButtonClick

    ; CPU
    ${NSD_CreateLabel} 103u 83u 186u 20u "Select this option if you do not have a GPU [Not recommended]."
    ${NSD_CreateButton} 8u 83u 84u 22u "CPU"
        pop $0
        nsDialogs::SetUserData $0 "cpu"
        ${NSD_OnClick} $0 fnVersionSelectButtonClick

    ; Native
    ${NSD_CreateLabel} 103u 109u 186u 20u "[Deprecated] This version will be removed from a future update. Nvidia users should choose Nvidia (WSL2)"
    ${NSD_CreateButton} 8u 109u 84u 22u "Nvidia (Native)"
        pop $0
        nsDialogs::SetUserData $0 "native"
        ${NSD_OnClick} $0 fnVersionSelectButtonClick

    nsDialogs::Show
FunctionEnd


Function fnCheckConda
    ; Look for existing Conda install in common locations
    nsExec::ExecToStack "$\"$PROFILE\Miniconda3\Scripts\conda.exe$\" -V"    ; miniconda default
    pop $0
    pop $1
    strCmp $0 0 0 +4
        strCpy $CondaDir "$PROFILE\Miniconda3"
        strCpy $Log "$Log[INFO] MiniConda installed: $1$\n"
        goto doLeave

    nsExec::ExecToStack "$\"$APPDATA\Miniconda3\Scripts\conda.exe$\" -V"    ; miniconda program files
    pop $0
    pop $1
    strCmp $0 0 0 +4
        strCpy $CondaDir "$APPDATA\Miniconda3"
        strCpy $Log "$Log[INFO] MiniConda installed: $1$\n"
        goto doLeave

    nsExec::ExecToStack "$\"C:\Miniconda3\Scripts\conda.exe$\" -V"          ; miniconda C:
    pop $0
    pop $1
    strCmp $0 0 0 +4
        strCpy $CondaDir "C:\Miniconda3"
        strCpy $Log "$Log[INFO] MiniConda installed: $1$\n"
        goto doLeave

    nsExec::ExecToStack "$\"$PROFILE\Anaconda3\Scripts\conda.exe$\" -V"     ; anaconda default
    pop $0
    pop $1
    strCmp $0 0 0 +4
        strCpy $CondaDir "$PROFILE\Anaconda3"
        strCpy $Log "$Log[INFO] AnaConda installed: $1$\n"
        goto doLeave

    nsExec::ExecToStack "$\"$APPDATA\Anaconda3\Scripts\conda.exe$\" -V"     ; anaconda program files
    pop $0
    pop $1
    strCmp $0 0 0 +4
        strCpy $CondaDir "$APPDATA\Anaconda3"
        strCpy $Log "$Log[INFO] AnaConda installed: $1$\n"
        goto doLeave

    nsExec::ExecToStack "$\"C:\Anaconda3\Scripts\conda.exe$\" -V"           ; anaconda C:
    pop $0
    pop $1
    strCmp $0 0 0 +4
        strCpy $CondaDir "C:\Anaconda3"
        strCpy $Log "$Log[INFO] AnaConda installed: $1$\n"
        goto doLeave

    doLeave:
        return
FunctionEnd


Function fnVersionSelectButtonClick
    pop $R0
	nsDialogs::GetUserData $R0
    pop $SetupType                                              ; Set the global SetupType variable

    strCpy $Log "$Log[INFO] Setting up for: $SetupType$\n"      ; Log the setup type

    strCmp $SetupType "wsl2" continue 0
        strCpy $FlgInstallWSL 0                                 ; Clear the flag to install WSL2 for non WSL2 installs (user may have gone back to this page)
        call fnCheckConda                                       ; Check for existing Conda install for non WSL2 installs
        strCmp $CondaDir "" 0 continue
            strCpy $FlgInstallConda 1                           ; Conda not found, then flag for install

    continue:
        GetDlgItem $0 $HWNDPARENT 1                             ; Re-enable Next
        EnableWindow $0 1
        GetDlgItem $0 $HWNDPARENT 1
        SendMessage $HWNDPARENT ${WM_COMMAND} 1 $0              ; Move to next page

FunctionEnd
