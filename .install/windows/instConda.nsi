; Faceswap Windows Installer - Download and install Conda functions

var cndInstallerExe                                                                                 ; Filename of downloaded miniconda3 installer
var cndEnvFolder                                                                                    ; Full path to the conda environment folder

!define cndURLWin "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"        ; Windows Conda installer
!define cndURLWSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"           ; Linux Conda installer
!define cndInstFlagsWin "/S /RegisterPython=0 /AddToPath=0 /D=$PROFILE\MiniConda3"                  ; MiniConda installation flags for Windows
!define cndInstFlagsWSL "-b -p"                                                                     ; MiniConda installation flags for WSL
!define cndFlagsEnv "-y python=3.10"                                                                ; Conda Python version


Function fnCondaDownload
    ; Download the MiniConda3 installer to temp folder
    strCpy $0 "${cndURLWin}"                                                                        ; Windows download URL
    strCpy $cndInstallerExe "miniconda.exe"                                                         ; Windows installer executable name
    strCmp $SetupType "wsl2" 0 +3
        strCpy $0 "${cndURLWSL}"                                                                    ; WSL2 download URL
        strCpy $cndInstallerExe "miniconda.sh"                                                      ; WSL2 installer executable name

    SetDetailsPrint both
    DetailPrint "[INFO] Downloading Miniconda3..."
    SetDetailsPrint listonly
    DetailPrint "[INFO] Download URL: '$0'"
    inetc::get /caption "Downloading Miniconda3..." /canceltext "Cancel" $0 $cndInstallerExe /end   ; Download installer
    pop $0

    SetDetailsPrint both
    strCmp $0 "OK" +3 0
        DetailPrint "[ERROR] Miniconda3 download failed"                                            ; Notify Download failed
        strCpy $FlgInstallFailed 1                                                                  ; Set Install Failed flag
    return
FunctionEnd


Function fnCondaPostInstallWSL
    ; Add conda to path and disable base activation for WSL2 installs
    SetDetailsPrint both
    DetailPrint "[INFO] Configuring Miniconda3..."

    strCpy $0 '$WSLExeUser "$CondaDir/bin/conda" init'
    SetDetailsPrint listonly
    DetailPrint "[INFO] Executing: '$0'"

    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED '$0'                                                   ; Initialise Conda for the user
    pop $0
    ExecDos::wait $0
    pop $0
    strCmp $0 0 +3 0                                                                                ; On Error
        push $0                                                                                     ; Push return code
        return                                                                                      ; Leave function

    strCpy $0 '$WSLExeUser "$CondaDir/bin/conda" config --set auto_activate_base false'
    SetDetailsPrint listonly
    DetailPrint "[INFO] Executing: '$0'"

    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED '$0'                                                  ; Disable base auto-activation
    pop $0
    ExecDos::wait $0
    pop $0
    strCmp $0 0 0 +3                                                                                ; On Success
        SetDetailsPrint listonly
        DetailPrint "[INFO] Configured Miniconda3"

    push $0                                                                                         ; Push return code
FunctionEnd


Function fnCondaInstall
    ; Install Miniconda3 from downloaded installer
    SetDetailsPrint both
    DetailPrint "[INFO] Installing Miniconda3. This may take a few minutes..."

    strCpy $0 '"$DirTemp\$cndInstallerExe"'                                                         ; Full path to installer
    strCpy $1 "${cndInstFlagsWin}"                                                                  ; Windows installation flags
    strCpy $2 "$PROFILE\MiniConda3"                                                                 ; Windows MiniConda3 install location
    strCmp $SetupType "wsl2" 0 install
        strCpy $0 '$WSLExeUser $$(wslpath -a $0)'                                                   ; Installer translated to wsl location
        strCpy $2 '"/home/$WSLUser/miniconda3"'                                                     ; WSL2 MiniConda3 install location
        strCpy $1 "${cndInstFlagsWSL} $2"                                                           ; WLS installation flags + installation location

    install:
        SetDetailsPrint listonly
        DetailPrint "[INFO] Executing: '$0 $1'"
        ExecDos::exec /NOUNLOAD /ASYNC /DETAILED '$0 $1'                                            ; Execute installer
        pop $0
        ExecDos::wait $0
        pop $0

    strCmp $0 0 0 onError
        strCpy $CondaDir $2                                                                         ; Add conda location to $CondaDir
        DetailPrint "[INFO] Miniconda3 installed"

    strCmp $SetupType "wsl2" 0 onSuccess
        call fnCondaPostInstallWSL                                                                  ; Post install actions for WSL2 installs
        pop $0
        strCmp $0 0 onSuccess onError                                                               ; Handle errors

    onSuccess:
        return                                                                                      ; Exit before onError triggers

    onError:
        SetDetailsPrint both
        DetailPrint "[ERROR] Miniconda3 installation failed"                                        ; Notify installation failed
        strCpy $FlgInstallFailed 1                                                                  ; Set Install Failed flag
        return
FunctionEnd


Function fnCondaUpdate
    ; Update the base Conda environmnet
    SetDetailsPrint both
    DetailPrint "[INFO] Updating Conda..."

    strCpy $0 "update -y -n base -c defaults conda"                                                 ; Conda update command
    strCmp $SetupType "wsl2" 0 +3
        strCpy $0 '$WSLExeUser "$CondaBin" $0'                                                      ; WSL2 update command
        goto update
    strCpy $0 '$CondaBin $0'                                                                        ; Windows update command

    update:
        SetDetailsPrint listonly
        DetailPrint "[INFO] Executing: '$0'"

        ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$0"
        pop $0
        ExecDos::wait $0
        pop $0
        strCmp $0 0 +3 0
            SetDetailsPrint both
            DetailPrint "[ERROR] Miniconda3 update failed"                                          ; Notify installation failed

        push $0                                                                                     ; Push return code
FunctionEnd


Function fnCondaEnvFolderExists
    ; Tests if a Conda env folder exits. Pushes 1 for exists, 0 for doesn't exist
    strCmp $SetupType "wsl2" testWSL testWindows

    testWSL:
        strCpy $0 '$WSLExeUser ls "$cndEnvFolder"'
        SetDetailsPrint listonly
        DetailPrint "[INFO] Executing: '$0'"
        ExecDos::exec /NOUNLOAD "$0"                                                                ; Attempt to ls the env folder
        pop $0
        strCmp $0 0 pushExists pushNoExists

    testWindows:
        IfFileExists "$cndEnvFolder" pushExists pushNoExists

    pushExists:
        push 1                                                                                      ; Push 1 for folder exists
        return

    pushNoExists:
        push 0                                                                                      ; Push 0 for folder does not exist
FunctionEnd


Function fnCondaEnvFolderRemove
    ; Often Conda won't actually remove the folder and some of it's contents which leads to permission problems later
    call fnCondaEnvFolderExists                                                                     ; Check if Env folder exists
    pop $0
    strCmp $0 1 delFolder 0
        push 0                                                                                      ; Push success if no folder to delete
        return                                                                                      ; exit

    delFolder:
        SetDetailsPrint listonly
        DetailPrint "[INFO] Deleting stale Conda Virtual Environment files"
        strCmp $SetupType "wsl2" delFolderWSL delFolderWin

    delFolderWSL:
        strCpy $0 '$WSLExeUser rm -rf "$cndEnvFolder"'
        SetDetailsPrint listonly
        DetailPrint "[INFO] Executing: '$0'"
        ExecDos::exec /NOUNLOAD "$0"                                                                ; Remove folder in WSL environ
        goto finally

    delFolderWin:
        RMDir /r "$cndEnvFolder"

    finally:
        pop $0
        strCmp $0 0 +3 0
            SetDetailsPrint both
            DetailPrint "[ERROR] Unable to delete Conda Virtual Environment Folder"

        push $0                                                                                     ; push return code
FunctionEnd


Function fnCondaEnvDelete
    ; Delete any pre-existing Conda env with our required name
    call fnCondaEnvFolderExists                                                                     ; Check if Env folder exists
    pop $0
    strCmp $0 1 delEnv 0
        push 0                                                                                      ; Push success if no folder to delete
        return                                                                                      ; exit

    delEnv:
        SetDetailsPrint both
        DetailPrint "[INFO] Removing existing Conda Virtual Environment '$EnvName'..."

        strCpy $0 'env remove -y -n "$EnvName"'
        strCpy $1 '$CondaBin $0'                                                                     ; Windows remove Env command
        strCmp $SetupType "wsl2" 0 +2
            strCpy $1 '$WSLExeUser "$CondaBin" $0'                                                   ; WSL2 Remove Env command

        SetDetailsPrint listonly
        DetailPrint "[INFO] Executing: '$1'"

        ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$1"
        pop $0
        ExecDos::wait $0
        pop $0
        strCmp $0 0 delFolder 0
            SetDetailsPrint both
            DetailPrint "[ERROR] Deleting pre-existing environment '$EnvName' failed"
            push $0                                                                                 ; Push return code
            return                                                                                  ; exit function

    delFolder:
        call fnCondaEnvFolderRemove                                                                 ; Remove the folder if it survived removal
        pop $0                                                                                      ; Get return code
        Push $0                                                                                     ; Push return code
FunctionEnd


Function fnCondaEnvCreate
    ; Create the faceswap Conda environment
    SetDetailsPrint both
    DetailPrint "[INFO] Creating Conda Virtual Environment '$EnvName'..."

    strCpy $0 'create ${cndFlagsEnv} -n "$EnvName"'                                                 ; Conda create env command
    strCmp $SetupType "wsl2" 0 +3
        strCpy $0 '$WSLExeUser "$CondaBin" $0'                                                      ; WSL2 create env command
        goto create
    strCpy $0 '$CondaBin $0'                                                                        ; Windows create env command

    create:
        SetDetailsPrint listonly
        DetailPrint "[INFO] Executing: '$0'"
        ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$0"                                               ; Create the environment
        pop $0
        ExecDos::wait $0
        pop $0
        strCmp $0 0 +3 0
            SetDetailsPrint both
            DetailPrint "[ERROR] Creating Conda environment '$EnvName' failed"
        push $0                                                                                     ; Push return code
FunctionEnd


Function fnCondaSetupEnvironment
    ; Update base conda, delete any pre-existing environment and create the faceswap environment
    strCpy $cndEnvFolder "$CondaDir\envs\$EnvName"                                                  ; Set path to env on Windows
    strCpy $CondaBin "$CondaDir\scripts\activate.bat && conda "                                     ; Set the activation path for Windows

    strCmp $SetupType "wsl2" 0 setup
        strCpy $cndEnvFolder "$CondaDir/envs/$EnvName"                                              ; Set path to env on WSL2
        strCpy $CondaBin "$CondaDir/bin/conda"                                                      ; Set path to conda executable for WSL2

    setup:
        call fnCondaUpdate                                                                          ; Update the Conda base environment
        pop $0
        strCmp $0 0 0 onError

        call fnCondaEnvDelete                                                                       ; Delete any pre-existing Conda envs
        pop $0
        strCmp $0 0 0 onError

        call fnCondaEnvCreate                                                                       ; Create the faceswap environment
        pop $0
        strCmp $0 0 0 onError

    return                                                                                          ; Exit before onError triggers

    onError:
        strCpy $FlgInstallFailed 1                                                                  ; Set Install Failed flag

FunctionEnd


Function fnCondaInstallGit
    ; Install Git into the newly created faceswap environment
    SetDetailsPrint both
    DetailPrint "[INFO] Installing Git..."

    strCpy $0 "conda install git -y -q"                                                             ; Conda install Git command
    strCpy $1 '$CondaBin activate "$EnvName" && $0'                                                 ; Windows install Git command
    strCmp $SetupType "wsl2" 0 +3
        strCpy $1 'conda activate "$EnvName" && $0' 
        strCpy $1 '$WSLExeUser /bin/bash -ic "$1"'                                                  ; WSL2 install Git command

    SetDetailsPrint listonly
    DetailPrint "[INFO] Executing: '$1'"
    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$1"                                                   ; Install git
    pop $0
    ExecDos::wait $0
    pop $0

    strCmp $0 0 +4 0
        SetDetailsPrint both
        DetailPrint "[ERROR] Failed to install Git"
        strCpy $FlgInstallFailed 1                                                                  ; Flag failed on error
FunctionEnd
