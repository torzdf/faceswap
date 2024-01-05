; Faceswap Windows Installer - [Input] WSL2 Selection pages
; This page collects information from the chosen WSL2 distro, to populate the $WSLUser
; variable and to locate Conda (if it exists).

!include LogicLib.nsh
!include MUI.nsh
!include nsDialogs.nsh
!include WinMessages.nsh

!define PBS_MARQUEE 0x08

var wslScanProgBar                                                                                  ; Marquee progress bar
var wslScanLblInfo                                                                                  ; The information label at the top of the page
var wslScanLB                                                                                       ; List box for outputting progress
var wslScanUserComboBox                                                                             ; Combo to hold discovered users
var wslScanCondaText                                                                                ; Textbox to hold location of Conda install
var wslScanUsers                                                                                    ; Holds all users found for the distro
var wslScanUserCount                                                                                ; Count of users found
var wslScanFlgInterrupted                                                                           ; Flag to indicate that scan was interrupted for user selection

!define wslScanTxtIntro "Please wait whilst we collect some information about your WSL2 install..." ; Default Info text
!define wslScanTxtUser "Please select the WSL2 User Account that you wish to install faceswap to:"  ; Select new user info text


Function wslScanToggleButtons
    ; Toggles the Back and Next buttons. push 1 for enable or 0 to diable
    pop $R0                                                                                         ; 1 for enable 0 for disable
    GetDlgItem $0 $HWNDPARENT 1                                                                     ; Toggle Next
    EnableWindow $0 $R0
    GetDlgItem $0 $HWNDPARENT 3                                                                     ; Toggle Back
    EnableWindow $0 $R0
FunctionEnd


Function wslScanParseUsers
    ; Callback function for ExecDos valid user query. Parses a line of output containing a
    ; valid user for the distro
    pop $0
    IntOp $wslScanUserCount $wslScanUserCount + 1                                                   ; Increment found user count
    strCmp $WSLUser "" 0 +2                                                                         ; If default user not yet set:
        strCpy $WSLUser $0                                                                          ; Set default user to first found (lowest UID)
    strCpy $wslScanUsers "$wslScanUsers|$0"                                                         ; Append this user to the list of distro users
FunctionEnd


Function wslScanUserComboBoxChange
    ; Callback for when the username combobox value is changes. Note: Value of Combo hasn't
    ; updated when this callback triggers, so hack in timer to let value update first
    ${NSD_CreateTimer} wslScanUserComboBoxChangeTimer 1                                             ; Timer for 1ms to let combo value update
FunctionEnd
Function wslScanUserComboBoxChangeTimer
    ; On user combo value change trigger the next scan step (simulate hitting next button)
    ${NSD_KillTimer} wslScanUserComboBoxChangeTimer                                                 ; Kill timer
    call pgWSLScanLeave                                                                             ; Call page leave function
FunctionEnd


Function wslScanChangeUser
    ; Handles when multiple users have been discovered and user wishes to go back
    ; to the page to select a different WSL2 user
    MessageBox MB_YESNO "Multiple users were discovered in Distro '$WSLDistro'.$\r$\n$\r$\n\
            Press 'Yes' to install faceswap for the user '$WSLUser'.$\r$\n$\r$\n\
            Press 'No' to select a different user." IDYES continue IDNO changeUser
        changeUser:
            strCpy $wslScanFlgInterrupted 1                                                         ; Set the page interrupted flag
            strCpy $WSLUser ""                                                                      ; Clear selected WSL User
            push 1
            call wslScanToggleButtons                                                               ; Enable Next/Back buttons
            ShowWindow $wslScanProgBar ${SW_HIDE}                                                   ; Hide the progress bar
            EnableWindow $wslScanUserComboBox 1                                                     ; Enable ComboBox on multiple users if different user is required
            SendMessage $wslScanLblInfo ${WM_SETTEXT} 0 "STR:${wslScanTxtUser}"                     ; Sent info text to 'select user'
            strCpy $Log "$Log${wslScanTxtUser}$\n"                                                  ; Add text to log
            ${NSD_LB_AddString} $wslScanLB "${wslScanTxtUser}"                                      ; Set info text to 'select user' in details box
            abort                                                                                   ; Return to the Page
        continue:
            return                                                                                  ; Carry on to next scan item
FunctionEnd


Function wslScanGotoNextPage
    ; Re-enable back + next button and go to next page
    push 1
    call wslScanToggleButtons                                                                       ; Enable Next/Back buttons
    GetDlgItem $0 $HWNDPARENT 1
    SendMessage $HWNDPARENT ${WM_COMMAND} 1 $0                                                      ; Go to next page
FunctionEnd


Function wslScanGetCondaInfo
    ; Display information about found users and call the Parse Conda functions
    strCpy $0 "[INFO] Setting default user to '$WSLUser'"
    strCpy $Log "$Log$0$\n"                                                                         ; Output default user to log
    strCpy $WSLExeUser '"wsl.exe" -d $WSLDistro -u $WSLUser --'                                     ; Store command for executing on WSL distro as user.

    ${NSD_LB_AddString} $wslScanLB $0                                                               ; Output default user to details box
    strCpy $0  "[INFO] Getting Conda information from WSL2 distro '$WSLDistro'..."
    strCpy $Log "$Log$0$\n"                                                                         ; Output status to log
    ${NSD_LB_AddString} $wslScanLB $0                                                               ; Output status to details box

    File wslCondaSearch.sh                                                                          ; Include the conda search script in installer
    strCpy $0 '$WSLExeUser $$(wslpath -a "$DirTemp\wslCondaSearch.sh")'                             ; Script translated to wsl location

    ExecDos::Exec /NOUNLOAD /TOSTACK $0 "" ""
    pop $0
    strCmp $0 0 +5 0                                                                               ; On error, just default to Conda not found
        strCpy $1 "[WARNING] Error '$0' searching for Conda. We will continue, but this may mean a failure later"
        strCpy $Log "$Log$1"
        ${NSD_LB_AddString} $wslScanLB $1                                                           ; Output status to details box
        push ""                                                                                     ; Push dummy "Not found"

    pop $0                                                                                          ; Path to Conda Folder

    ${if} $0 == ""
        strCpy $0 "Not Found"                                                                       ; Set text to "Not found" if conda not located
        strCpy $1 "[INFO] Conda could not be located so will be installed"                          ; Set detailed text to "Not Found" if conda not located
        strCpy $FlgInstallConda 1                                                                   ; Set the install conda flag
    ${Else}
        strCpy $1 "[INFO] Conda located at $0"                                                      ; Details message to Conda folder
        strCpy $CondaDir $0                                                                         ; Set Conda folder to $CondaDir
    ${EndIf}
    strCpy $Log "$Log$1$\n"                                                                         ; Output Conda details to log
    ${NSD_LB_AddString} $wslScanLB $1                                                               ; Output Conda details to details box
    ${NSD_SetText} $wslScanCondaText $0                                                             ; Store conda location in text box

    call wslScanGotoNextPage                                                                        ; Go to the next page
FunctionEnd


Function wslScanParseUsersComplete
    ; Display information about found users and call the Parse Conda functions
    strCmp $wslScanUserCount 0 0 +4                                                                 ; If no users go back to select distro page
        MessageBox MB_ICONEXCLAMATION "No users found in in Distro '$WSLDistro'.$\r$\n$\r$\nPlease select a different Distro."
        SendMessage $HWNDPARENT "0x408" "-1" ""
        return

    strCpy $0 "[INFO] Found $wslScanUserCount user(s)"
    strCpy $Log "$Log$0$\n"                                                                         ; Add info to log
    ${NSD_LB_AddString} $wslScanLB $0                                                               ; Output found user count

    SendMessage $wslScanUserComboBox ${CB_RESETCONTENT} 0 0                                         ; Reset the ComboBox
    strCpy $wslScanUsers $wslScanUsers "" 1                                                         ; Remove leading '|' from $wslScanUsers

    ${Explode} $0 "|" $wslScanUsers                                                                 ; Iterate through found users
        ${For} $1 1 $0
            pop $2
            ${if} $2 == ""                                                                          ; skip empty rows
                ${continue}
            ${EndIf}
            ${NSD_CB_AddString} $wslScanUserComboBox $2                                             ; Add user to ComboBox
        ${Next}

    ${NSD_CB_SelectString} $wslScanUserComboBox $WSLUser                                            ; Select default user
    ${NSD_OnChange} $wslScanUserComboBox wslScanUserComboBoxChange                                  ; Callback for when the combo box value is changed

    strCmp $wslScanUserCount 1 continue 0                                                           ; Multiple users handling
        call wslScanChangeUser                                                                      ; Call function to ask if the selected user should be changed

    continue:
        call wslScanGetCondaInfo                                                                    ; Call next function to gather install Conda info
FunctionEnd


Function wslScanProgBarAnimate
    ; Callback to animate the maquee progress bar once dialog is drawn
    ${NSD_KillTimer} wslScanProgBarAnimate                                                          ; Kill the callback timer
    SendMessage $wslScanProgBar ${PBM_SETMARQUEE} 1 50                                              ; Marquee animation @50ms
FunctionEnd


; =======================================
; Distro Scan page + page leave functions
; =======================================

Function pgWSLScan
    ; Scan the current WSL install

    strCmp $SetupType "wsl2" +2 0                                                                   ; skip page for Non WSL2 installs
        abort
    strCmp $FlgInstallWSL 1 0 +2                                                                    ; skip page if we need to install WSL2
        abort
    strCmp $WSLInstallDistro "" +2 0                                                                ; skip page when WSL2 Distro needs installing
        abort

    nsDialogs::Create 1018
    pop $0
    strCmp $0 error 0 +2
        abort

    push 0
    call wslScanToggleButtons                                                                       ; Disable Next/Back buttons

    strCpy $wslScanUserCount 0                                                                      ; Set initial user count
    strCpy $wslScanUsers ""                                                                         ; Clear scanned user list (in case we are returning to this page)
    strCpy $FlgInstallConda 0                                                                       ; reset the install conda flag (in case we are returning to this page)
    strCpy $WSLExeUser ""                                                                           ; reset the command to run as specific user on wsl2 (in case we are returning to this page)
    strCpy $CondaDir ""                                                                             ; reset $CondaDir (in case we are returning to this page)

    !insertmacro MUI_HEADER_TEXT "faceswap Installer" "WSL2 Setup"
    ${NSD_CreateLabel} 8u 2u 292u 10u "${wslScanTxtIntro}"                                          ; Dynamic Intro text
    pop $wslScanLblInfo

    ${NSD_CreateProgressBar} 8u 20u 292u 10u "WSL2 Progress"                                        ; Progress bar
        pop $wslScanProgBar
        ${NSD_AddStyle} $wslScanProgBar ${PBS_MARQUEE}                                              ; Make progress bar marquee style
        ${NSD_CreateTimer} wslScanProgBarAnimate 10                                                 ; Trigger animation on timer callback

    ${NSD_CreateListBox} 8u 40u 292u 70u ""                                                         ; Details listbox
        pop $wslScanLB
        strCpy $0 "[INFO] Getting users from WSL2 distro '$WSLDistro'..."
        strCpy $Log "$Log$0$\n"                                                                     ; Output details to log
        ${NSD_LB_AddString} $wslScanLB $0                                                           ; Output details to details box

    ${NSD_CreateLabel} 8u 120u 40u 10u "Username:"                                                  ; linux username
        pop $0
        SendMessage $0 ${WM_SETFONT} $FntBold 0                                                     ; set label bold
    ${NSD_CreateComboBox} 50u 119u 96u 11u ""                                                       ; ComboBox for selecting user if multiple found
        pop $wslScanUserComboBox
        ${NSD_CB_AddString} $wslScanUserComboBox "Scanning..."                                      ; Default Combotext when scanning distro
        ${NSD_CB_SelectString} $wslScanUserComboBox "Scanning..."                                   ; Display scanning...
        EnableWindow $wslScanUserComboBox 0                                                         ; Disable ComboBox until required

    ${NSD_CreateLabel} 154u 120u 30u 10u "Conda:"                                                   ; Conda location
        pop $0
        SendMessage $0 ${WM_SETFONT} $FntBold 0                                                     ; set label bold
        ${NSD_AddStyle} $0 ${SS_RIGHT}                                                              ; align right
    ${NSD_CreateText} 188u 119u 112u 11u "Scanning..."                                              ; Textbox to display conda location (if found)
        pop $wslScanCondaText
        EnableWindow $wslScanCondaText 0                                                            ; Disable textbox

    strCpy $0 "cut -d: -f1,3 /etc/passwd | egrep ':[0-9]{4}$$' | cut -d: -f1"                       ; Bash command for listing users
    strCpy $0 '"wsl.exe" -d $WSLDistro -u root -- /bin/bash -c "$0"'                                ; Run on correct distro as root

    GetFunctionAddress $R0 wslScanParseUsers
    GetFunctionAddress $R1 wslScanParseUsersComplete
    ExecDos::Exec /NOUNLOAD /ASYNC /TOFUNC /ENDFUNC=$R1 $0 "" $R0
    pop $0

    nsDialogs::Show
    ExecDos::wait $0
FunctionEnd


Function pgWSLScanLeave
    ; Actions to perform when leaving the Distro scan page
    ; Code only executes if scan has been interrupted because multiple usernames were found
    ; and the chosen username is not the preferred option (when the Next button is enabled).
    strCmp $wslScanFlgInterrupted 1 continue finally
    continue:                                                                                       ; Continue with Conda scan after a manual user selection
        push 0
        call wslScanToggleButtons                                                                   ; Disable Next/Back buttons
        strCpy $wslScanFlgInterrupted 0                                                             ; Clear interrupted flag
        ShowWindow $wslScanProgBar ${SW_SHOW}                                                       ; Show the progress bar
        SendMessage $wslScanLblInfo ${WM_SETTEXT} 0 "STR:${wslScanTxtIntro}"                        ; Reset the info text
        ${NSD_GetText} $wslScanUserComboBox $WSLUser                                                ; Get selected user
        EnableWindow $wslScanUserComboBox 0                                                         ; Disable user ComboBox
        call wslScanGetCondaInfo                                                                    ; Call get conda info function
    finally:
        return                                                                                      ; Scan completed. Just leave

FunctionEnd
