; NSIS script for confirming WSL2 application install + performing the installation

!include MUI2.nsh

!define wslExeDISM "$WINDIR\Sysnative\dism.exe"                             ; Deployment Image Servicing + Management EXE


; === INSTALL WSL PAGE ===
Function pgWSLInstallConfirm
    ; Comfirm installation of WSL2

    strCmp $SetupType "wsl2" +2 0                                           ; skip page for Non WSL2 installs
        abort
    strCmp $FlgInstallWSL 1 +2 0                                            ; skip page if we don't need to install WSL2
        abort

    nsDialogs::Create 1018                                                  ; create dialog
    pop $0
    strCmp $0 error 0 +2
        abort

    ; Page header
    !insertmacro MUI_HEADER_TEXT "faceswap Installer" "WSL2 Install"

    ; Intro
    ${NSD_CreateLabel} 8u 10u 292u 90u "WSL2 (Windows Subsystem for Linux) is required but was \
        not found on your system.$\r$\n$\r$\n$\r$\n\
        Press 'Next' to install WSL2. Note: You will need to reboot your PC to continue the installation.$\r$\n$\r$\n$\r$\n\
        Press 'Back' to choose a different faceswap version.$\r$\n$\r$\n$\r$\n\
        Press 'Cancel' to quit the installer."

    nsDialogs::Show

FunctionEnd


Function fnWSLInstall
    ; Enable WSL2 Windows Features and prompt for reboot or quit

    strCmp $FlgInstallWSL 1 0 end                                           ; skip page for Non WSL2 application installs

    SetDetailsPrint both
    DetailPrint "[INFO] Installing Microsoft Windows Subsystem Linux..."    ; WSL2
    SetDetailsPrint listonly
    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED '"${wslExeDISM}" /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart'
    pop $R0
    ExecDos::wait $R0
    pop $R0
    strCmp $R0 3010 0 doFail                                                ;  3010 = Reboot required exit code

    SetDetailsPrint both
    DetailPrint "[INFO] Installing Microsoft Virtual Machine Platform..."   ; Virtual Machine Platform
    SetDetailsPrint listonly
    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED '"${wslExeDISM}" /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart'
    pop $R0
    ExecDos::wait $R0
    pop $R0
    strCmp $R0 3010 0 doFail                                                ;  3010 = Reboot required exit code

    ; set the reboot resume flag
    DetailPrint "[INFO] Setting resume flag..."
    WriteRegStr HKCU "Software\Microsoft\Windows\Currentversion\Runonce" "InstallResume" '"$EXEPATH" /resume'
    SetDetailsPrint both
    DetailPrint "[INFO] WSL2 as been installed."
    DetailPrint "[INFO] The system must be rebooted before installation can proceed."
    DetailPrint "[INFO] Press 'Cancel' to quit the installer."

    MessageBox MB_OKCANCEL "Windows Subsystem for Linux (WSL2) has been installed.$\r$\n$\r$\n\
        You must reboot your system before continuing with the installation.$\r$\n$\r$\n\
        Press 'Ok' to reboot now, or 'Cancel' to exit the installer."  IDOK doReboot IDCANCEL doLeave

    doReboot:
        Reboot
    doLeave:
        abort
    doFail:
        MessageBox MB_ICONEXCLAMATION "WSL2 Could not be installed. Please install this software manually or choose a different version"
        Quit
    end:

FunctionEnd
