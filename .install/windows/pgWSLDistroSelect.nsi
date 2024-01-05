; Pages for selecting an existing WSL2 and for selecting and installing a new WSL2 distro

!include LogicLib.nsh
!include MUI2.nsh

var wslDistros                                                                  ; Installed distros
var wslDistroLB                                                                 ; Installed distros listbox
var wslDistroNewLB                                                              ; Available distros listbox
var wslUserTB                                                                   ; Linux username textbox
var wslPasswordTB                                                               ; Linux password textbox
var wslPasswordConfirmTB                                                        ; Linux Password confirmation textbox

!define wslTxtInstallNew "Install new Distro..."                                ; Set default installer distro text
!define wslAvailableDistros "Debian|Ubuntu|openSUSE-Tumbleweed"                 ; Available WSL2 distros


; ===================================
; === SELECT EXISTING DISTRO PAGE ===
; ===================================

Function pgWSLSelect
    ; Selection options for WSL2 installs
    strCmp $SetupType "wsl2" +2 0                                               ; skip page for Non WSL2 installs
        abort

    nsExec::ExecToStack '"$WINDIR\Sysnative\wsl.exe" -l -q'                     ; Get installed distros
        pop $0                                                                  ; ret code
        pop $1                                                                  ; ret value. Distro list newline separated
        ${If} $0 == 0
            strCpy $wslDistros $1                                               ; Set distro list to $wslDistros
        ${Else}
            strCpy $Log "[INFO] WSL2 could not be found. WSL2 distro needs installing."
            strCpy $FlgInstallWSL 1                                             ; Set install WSL flag to 1
            abort                                                               ; Exit
        ${EndIf}

    nsDialogs::Create 1018                                                      ; create dialog
    pop $0
    strCmp $0 error 0 +2
        abort

    ; Header + intro
    !insertmacro MUI_HEADER_TEXT "faceswap Installer" "WSL2 Select Distribution"
    ${NSD_CreateLabel} 8u 2u 292u 50u "Faceswap can be installed into an existing WSL2 Distro, or a new Distro can be downloaded and installed.$\r$\n$\r$\n\
        If any installed WSL2 Distros have been found, they wiil be shown here.$\r$\n$\r$\n\
        Please select the WSL2 Distro to install faceswap to:"

    ${NSD_CreateListBox} 8u 60u 292u 70u ""                                     ; List box for selecting exsting WSL2 distro
        pop $wslDistroLB

    strCpy $4 ""                                                                ; default value

    ${Explode} $0 "$\r$\n" $wslDistros                                          ; Iterate through found installed distros
        ${For} $1 1 $0
            pop $2

            ${if} $2 == ""                                                      ; skip empty rows
                ${continue}
            ${EndIf}

            ${NSD_LB_AddString} $wslDistroLB $2                                 ; Add distro to listbox

            ${if} $4 == ""                                                      ; set the default distro (first found, or Debian if it exists)
            ${orif} $2 == "Debian"
                strCpy $4 $2
            ${EndIf}

        ${Next}

    ${NSD_LB_AddString} $wslDistroLB "${wslTxtInstallNew}"                      ; Add new distro option

    strCmp $4 "" 0 +2                                                           ; Set the default if no distros installed
        strCpy $4 "${wslTxtInstallNew}"

    ${NSD_LB_SelectString} $wslDistroLB $4                                      ; Highlight default distro
    nsDialogs::Show
FunctionEnd


Function pgWSLSelectLeave
    ; Get selected distro on page exit
    ${NSD_LB_GetSelection} $wslDistroLB $WSLDistro
    ${if} $WSLDistro == "${wslTxtInstallNew}"                                   ; Clear selected distro when selecting new distro
        strCpy $WSLDistro ""
    ${EndIf}
FunctionEnd


; ==============================
; === SELECT NEW DISTRO PAGE ===
; ==============================

Function pgWSLDistroInstall
    ; Select a new WSL2 distro to install

    ; Note: At time of writing there is no mechanism for using the 'wsl.exe' command to install a distro without
    ; requiring user input. This means that we need to manually install any selected distro, and cannot just list
    ; available distros using the 'wsl.exe --list --online' as we would need to keep track of the actual distro
    ; download locations. With this in mind, we provide a limited selection of available distros based on known
    ; (and hopefully non-changing) download locations

    strCmp $SetupType "wsl2" +2 0                                               ; skip page for Non WSL2 installs
        abort
    strCmp $FlgInstallWSL 1 0 +2                                                ; skip page if we need to install WSL2
        abort
    strCmp $WSLDistro "" +2 0                                                   ; skip page when distro already selected
        abort

    nsDialogs::Create 1018                                                      ; create dialog
    pop $0
    strCmp $0 error 0 +2
        abort

    ; Header + intro
    !insertmacro MUI_HEADER_TEXT "faceswap Installer" "WSL2 Install Distribution"
    ${NSD_CreateLabel} 8u 2u 292u 26u "Faceswap can help set up a new WSL2 Distro for you.$\r$\n$\r$\n\
        Select the WSL2 Distro that you wish to install:"

    ; == Distro List ==
    ${NSD_CreateListBox} 8u 28u 292u 50u ""                                     ; Distro listbox
        pop $wslDistroNewLB

    strCpy $1 "${wslAvailableDistros}"                                          ; Valid distros for install
    strCpy $0 ""                                                                ; default value
    ${Explode} $2 "|" $1                                                        ; split available distros by "|"
        ${For} $3 1 $2                                                          ; iterate rows
            pop $4                                                              ; row value

            ${ItemInArray} $5 "$\r$\n" $wslDistros "$4-faceswap"
            ${If} $5 == 1                                                       ; Skip if Distro already installed
                ${continue}
            ${EndIf}

            ${NSD_LB_AddString} $wslDistroNewLB $4                              ; Add distro to listbox

            ${if} $0 == ""                                                      ; Set the default
            ${orif} $4 == "Debian"
                strCpy $0 $4
            ${EndIf}
        ${Next}

    ${NSD_LB_SelectString} $wslDistroNewLB $0                                   ; Highlight default

    ; == User Information ==
    ${NSD_CreateLabel} 8u 90u 292u 22u "The WSL2 Distro requires a Username and Password. \
        These do not need to be the same as your Windows credentials:"

    System::Call "advapi32::GetUserName(t .r0, *i ${NSIS_MAX_STRLEN} r1) i.r2"  ; Get current username
    ${GetFirst} $1 " " $0                                                       ; Get name before first space

    ${NSD_CreateLabel} 8u 112u 40u 10u "Username:"                              ; linux username
        pop $0
        SendMessage $0 ${WM_SETFONT} $FntBold 0                                 ; set label bold
    ${NSD_CreateText} 50u 111u 96u 11u "$1"                                     ; Set default username to current Windows user
        pop $wslUserTB                                                          ; store username handle for handling on leave

    ${NSD_CreateLabel} 154u 112u 40u 10u "Password:"                            ; linux password
        pop $0
        SendMessage $0 ${WM_SETFONT} $FntBold 0                                 ; set label bold
        ${NSD_AddStyle} $0 ${SS_RIGHT}                                          ; align right
    ${NSD_CreatePassword} 198u 111u 102u 11u ""
        pop $wslPasswordTB                                                      ; store password handle for handling on leave

    ${NSD_CreateLabel} 124u 129u 70u 10u "Confirm Password:"                    ; linux password confirm
        pop $0
        SendMessage $0 ${WM_SETFONT} $FntBold 0                                 ; set label bold
        ${NSD_AddStyle} $0 ${SS_RIGHT}                                          ; align right
    ${NSD_CreatePassword} 198u 128u 102u 11u ""
        pop $wslPasswordConfirmTB                                               ; store password confirmation handle for handling on leave

    nsDialogs::Show
FunctionEnd


Function pgWSLDistroInstallLeave
    ; Validate and collect information on page exit

    ; == Validate ==
    ${NSD_GetText} $wslUserTB $R0                                               ; Get username
    strCmp $R0 "" 0 +3                                                          ; Go back if no username provided
        MessageBox MB_ICONEXCLAMATION "A new username is required for WSL2"
        abort

    push $R0
    call CheckForSpaces                                                         ; Check the username for spaces
    pop $0
    strCmp $0 0 +3 0                                                            ; Go back if spaces in username
        MessageBox MB_ICONEXCLAMATION "The WSL2 username must not contain spaces"
        abort

    ${NSD_GetText} $wslPasswordTB $R1                                           ; Get password
    strCmp $R1 "" 0 +3                                                          ; Go back if no password provided
        MessageBox MB_ICONEXCLAMATION "A new password is required for WSL2"
        abort

    ${NSD_GetText} $wslPasswordConfirmTB $R2                                    ; Get password confirmation
    strCmp $R1 $R2 +3 0                                                         ; Go back if passwords don't match
        MessageBox MB_ICONEXCLAMATION "Passwords don't match"
        abort

    ; == Store Information ==
    strCpy $WSLUser $R0                                                         ; Set the WSLUser var
    strCpy $WSLPassword $R1                                                     ; Set the WSLPassword var

    ${NSD_LB_GetSelection} $wslDistroNewLB $WSLDistro                           ; Get selected Distro
    strCpy $WSLInstallDistro $WSLDistro                                         ; Put distro to install into $WSLInstallDistro
    strCpy $FlgInstallConda 1                                                   ; This is a new Distro so Conda will be required
FunctionEnd
