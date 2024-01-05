; Faceswap Windows Installer - Final actions to perform on install

!define FnlLauncherPath "$INSTDIR\faceswap_win_launcher.bat"                                ; Full path to the .bat launcher

Function fnFnlAddGuiLauncher
    ; Add a .bat file to the faceswap folder that launches straight into the faceswap GUI
    SetDetailsPrint both
    DetailPrint "[INFO] Creating GUI Launcher"
    SetOutPath "$INSTDIR"

    strCpy $0 'conda activate "$EnvName" && python'                                         ; activate environment and introduce python
    strCpy $1 '"$CondaDir\scripts\activate.bat" && $0 "$INSTDIR\faceswap.py" gui$\r$\n"'    ; Windows launcher command
    strCmp $SetupType "wsl2" 0 +3
        strCpy $1 '$0 $$(wslpath -a "$INSTDIR\faceswap.py") gui'
        strCpy $1 '$WSLExeUser bash -ic "$1"$\r$\n'                                         ; WSL2 launcher command

    SetDetailsPrint listonly
    DetailPrint "[INFO] Creating launcher '${FnlLauncherPath}' with command '$1'"

    FileOpen $9 "${FnlLauncherPath}" w                                                      ; Open launcher .bat file for writing
    FileWrite $9 "$1"                                                                       ; Write command
    FileClose $9                                                                            ; Close
FunctionEnd

Function fnFnlDesktopShortcut
    ; Create a desktop shortcut to the newly created launcher .bat file
    SetDetailsPrint both
    DetailPrint "[INFO] Creating Desktop Shortcut"
    strCpy $0 '$DESKTOP\faceswap.lnk'                                                       ; Desktop file
    strCpy $1 '${FnlLauncherPath}'                                                          ; Launcher .bat file
    strCpy $2 '$INSTDIR\.install\windows\fs_logo.ico'                                       ; Desktop Icon

    SetDetailsPrint listonly
    DetailPrint "[INFO] Desktop file: '$0'. Launcher path: '$1'. Icon path: '$2'"
    CreateShortCut '$0' '$1' '' '$2'
FunctionEnd
