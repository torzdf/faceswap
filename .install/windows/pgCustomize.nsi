; Faceswap Windows Installer -  [Input] Customize and Finalize Install pages

!include "MUI.nsh"
!include nsDialogs.nsh
!include "${NSISDIR}\Include\WinMessages.nsh"

var cusCtlEnvName                                                                   ; Text box for Conda environment name
var cusCtlCondaText                                                                 ; Text box for Conda location
var cusCtlCondaButton                                                               ; Button for Conda file browser


Function fnCtlCondaButtonClick
    ; Launch file browser on Conda browse button click and set returned value to Conda text box
	pop $R0
	${If} $R0 == $cusCtlCondaButton
		${NSD_GetText} $cusCtlCondaText $R0
		nsDialogs::SelectFolderDialog /NOUNLOAD "" "$R0"
		pop $R0
		${If} "$R0" != "error"
			${NSD_SetText} $cusCtlCondaText "$R0"
		${EndIf}
	${EndIf}
FunctionEnd


; === INSTALL DETAILS AND CUSTOMIZE PAGE
Function pgCustomize

    strCmp $FlgInstallWSL 1 0 +2                                                    ; skip page if we need to install WSL2
        abort

    !insertmacro  MUI_HEADER_TEXT "faceswap Installer" "Customize Install"

    nsDialogs::Create 1018
    pop $0

    ${If} $0 == error
        abort
    ${EndIf}

    ; Info Text
    ${NSD_CreateLabel} 8u 2u 292u 10u "Finally we need to collect some information about the environment that will run faceswap."

    ; Environment Name
    ${NSD_CreateLabel} 8u 27u 292u 30u "Faceswap runs in an isolated space called an 'Environment'. The default name \
        will be fine for almost all use cases, but if you need to select a custom name, you can do so here. \
        (Note: Existing environments with this name will be deleted)"
    ${NSD_CreateLabel} 12u 57u 74u 10u "Environment name:"                          ; label
        pop $0
        SendMessage $0 ${WM_SETFONT} $FntBold 0                                     ; set label bold
    ${NSD_CreateText} 87u 55u 203u 13u "$EnvName"                                   ; text box
        pop $cusCtlEnvName

    ; Conda location
    strCmp $WSLInstallDistro "" 0 continue                                          ; If we're installing a distro, skip Conda request
    strCmp $FlgInstallConda 1 0 continue
        strCpy $0 "Faceswap uses Conda to manage its environment, but it could not be detected"
        strCmp $SetupType "wsl2" 0 +2
            strCpy $0 "$0 within the Distro '$WSLDistro'"
        strCpy $0 "$0. If Conda is already installed, please specify the location below, otherwise leave blank."
        strCmp $SetupType "wsl2" 0 +2
            strCpy $0 "$0 $\r$\n(Note: The path specified should be within the WSL2 distro)"

        strCpy $1 "94u"                                                             ; info label y position
        strCpy $2 "30u"                                                             ; info label height
        strCpy $3 "151u"                                                            ; text box width
        strCmp $SetupType "wsl2" 0 +4
            strCpy $1 "84u"                                                         ; info label y position (wsl2 text)
            strCpy $2 "40u"                                                         ; info label height (wsl2 text)
            strCpy $3 "203u"                                                        ; text box width (No button for wsl2 version)

        ${NSD_CreateLabel} 8u $1 292u $2 $0                                         ; Conda information label
        ${NSD_CreateLabel} 12u 123u 74u 10u "Conda location:"                       ; header label
            pop $0
            SendMessage $0 ${WM_SETFONT} $FntBold 0                                 ; set label bold
        ${NSD_CreateText} 87u 121u $3 13u ""                                        ; text box
            pop $cusCtlCondaText                                                    ; store conda path for checking
        strCmp $SetupType "wsl2" continue 0
            ${NSD_CreateButton} 238u 121u 50u 13u "Browse ..."                      ; file browser button for Windows only
                pop $cusCtlCondaButton
                ${NSD_OnClick} $cusCtlCondaButton fnCtlCondaButtonClick             ; callback to open file browser
    continue:
        nsDialogs::Show
FunctionEnd


Function cusCheckCondaPath
    ; Check Conda path is correct. If so store $CondaDir otherwise flag that Conda must be installed
    ${NSD_GetText} $cusCtlCondaText $R0                                             ; Get the specified path to conda folder
    strCmp $R0 "" 0 +2                                                              ; No value, just exit
        return

    strCpy $R1 '"$R0\Scripts\conda.exe" -V'                                         ; Command for Windows Conda check
    strCmp $SetupType "wsl2" 0 +2
        strCpy $R1 '$WSLExeUser "$R0/bin/conda" -V'                                 ; If WSL2 then update command accordingly

    nsExec::Exec $R1                                                                ; Test executable
    pop $0

    strCmp $0 0 0 onError                                                           ; Check return code
        strCpy $CondaDir "$R0"                                                      ; set $condaDir
        strCpy $FlgInstallConda 0                                                   ; UnSet install conda flag
        strCpy $Log "$Log[INFO] Custom Conda found: '$R0'$\n"                       ; Log
        return
    onError:                                                                        ; If conda not found at given location: log warning message
        strCpy $Log "$Log[WARNING] Custom Conda not found at: '$R0'. Installing MiniConda$\n"
FunctionEnd


Function pgCustomizeLeave
    ; On leave Ensure that Conda path is correct and set the environment name
    call cusCheckCondaPath                                                          ; Check Conda path for Windows installs (WSL2 will do this as part of installation)
    ${NSD_GetText} $cusCtlEnvName $EnvName                                          ; Store the environment name
FunctionEnd


; === INSTALL DETAILS AND CONFIRM PAGE
Function pgFinalize

    strCmp $FlgInstallWSL 1 0 +2                                                    ; skip page if we need to install WSL2
        abort

    !insertmacro  MUI_HEADER_TEXT "faceswap Installer" "Finalize Install"

    nsDialogs::Create 1018
    pop $0

    strCmp $0 error 0 +2
        abort

    ; Get list of required install actions and display
    strCpy $1 ""
    strCmp $WSLInstallDistro "" +4 0
        strCpy $1 "$1 • [INSTALL] WSL2 Distribution: '$WSLInstallDistro'$\r$\n$\r$\n"
        strCpy $1 "$1 • [SETUP] WSL2 User: '$WSLUser'$\r$\n$\r$\n"
        goto conda

    strCmp $SetupType "wsl2" 0 conda
        strCpy $1 "$1 • [USE] WSL2 Distribution: '$WSLDistro' with User '$WSLUser'$\r$\n$\r$\n"

    conda:
        ${If} $FlgInstallConda == 1                                                 ; Conda to install?
            strCpy $1 "$1 • [INSTALL] MiniConda3"
            ${If} $SetupType == "wsl2"
                strCpy $1 "$1 to WSL2 Distro: '$WSLDistro'"
            ${EndIf}
        ${Else}
            strCpy $1 "$1 • [USE] Existing Conda: '$CondaDir'"
        ${EndIf}
        strCpy $1 "$1$\r$\n$\r$\n"

        strCpy $1 "$1 • [CREATE] Conda environment: '$EnvName'$\r$\n$\r$\n"         ; Conda env name
        strCpy $1 "$1 • [INSTALL] faceswap to: '$INSTDIR'$\r$\n$\r$\n"              ; Faceswap install location
        strCpy $1 "$1 • [CONFIGURE] environment for '$SetupType'$\r$\n$\r$\n"
        strCpy $1 "$1 • [ADD] a Desktop shortcut"

     strCpy $Log "$Log[INFO] Install actions to perform:$\r$\n$1"

     ${NSD_CreateLabel} 8u 0u 292u 10u "The following actions will be performed:"   ; Install actions header
     ${NSD_CreateLabel} 12u 12u 288u 110u $1                                        ; Install actions details
        pop $0
        SetCtlColors $0 0x000000 0xffffff

    ${NSD_CreateLabel} 8u 130u 292u 10u "Please take a minute to review your install options and then \
        press 'Install' to proceed."

    nsDialogs::Show
FunctionEnd
