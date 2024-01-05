; Faceswap Windows Installer - Clone faceswap repo and install environment

!define fsGitCmd "git clone --depth 1 --no-single-branch --progress https://github.com/deepfakes/faceswap.git" ; Git command
!define fsFlgSetup "--installer"                                                                        ; flags for calling faceswap/setup.py



Function fnFsCloneRepoWSL
    ; Git clone the faceswap repo into the selected installation folder for WSL.
    ; As we are using an NTFS drive, git fails due to permissions and the way WSL2 mounts the disks
    ; We get around this by git cloning to a temp folder, and then copying the files over to the
    ; requested install folder
    strCpy $R0 '$WSLExeUser /bin/bash -ic '                                                             ; Common WSL execution prefix
    strCpy $R1 '$$(wslpath -a "$INSTDIR")'                                                              ; Translation from Windows path to WSL2

    strCpy $0 'rm -rf /tmp/faceswap && '                                                                ; Remove temp folder if it exists
    strCpy $0 '$0 conda activate "$EnvName" && ${fsGitCmd} /tmp/faceswap'                               ; WSL2 git clone command inside Conda to temp folder
    strCpy $0 '$R0 "$0"'
    SetDetailsPrint listonly
    DetailPrint "[INFO] Executing: '$0'"
    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$0"
    pop $0
    ExecDos::wait $0
    pop $0
    strCmp $0 0 0 onError

    strCpy $0 '$R0 "cp -r /tmp/faceswap $R1 && rm -r /tmp/faceswap"'                                    ; Copy from temp folder to install folder. Can't preserve ownership on ntfs
    SetDetailsPrint listonly
    DetailPrint "[INFO] Executing: '$0'"
    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$0"
    pop $0
    ExecDos::wait $0
    pop $0
    strCmp $0 0 0 onError

    strCpy $0 'conda activate "$EnvName" && git config --global --add safe.directory $R1'
    strCpy $0 '$R0 "$0"'                                                                                ; Tell git that this folder is safe (due to ownership issues)
    SetDetailsPrint listonly
    DetailPrint "[INFO] Executing: '$0'"
    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$0"
    pop $0
    ExecDos::wait $0
    pop $0

    onError:
        Push $0                                                                                         ; Push return code (This is pushed for success + failure)
FunctionEnd


Function fnFsCloneRepo
    ; Git clone the faceswap repo into the selected installation folder
    SetDetailsPrint both
    DetailPrint "[INFO] Downloading faceswap..."
    strCmp $SetupType "wsl2" 0 +4
        call fnFsCloneRepoWSL                                                                           ; Call WSL install code
        pop $0                                                                                          ; Get return code
        goto finally                                                                                    ; jump to end

    strCpy $0 '$CondaBin activate "$EnvName" && ${fsGitCmd} "$INSTDIR"'                                 ; Windows git clone command inside Conda

    SetDetailsPrint listonly
    DetailPrint "[INFO] Executing: '$0'"
    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$0"                                                       ; Clone repo
    pop $0
    ExecDos::wait $0
    pop $0

    finally:
        strCmp $0 0 +4 0
            SetDetailsPrint both
            DetailPrint "[ERROR] Failed to Download faceswap"
            strCpy $FlgInstallFailed 1
FunctionEnd


Function fnFsConfigure
    ; Call the faceswap setup script for the requested backend to set up the environment
    SetDetailsPrint both
    DetailPrint "Setting up faceswap Environment... This may take a while"

    strCpy $0 '$CondaBin activate "$EnvName" && python "$INSTDIR\setup.py" ${fsFlgSetup} --$SetupType'  ; Windows installer command
    strCmp $SetupType "wsl2" 0 +4                                                                       ; Build WSL2 command
        strCpy $0 'conda activate "$EnvName" &&'
        strCpy $0 '$0 python $$(wslpath -a "$INSTDIR\setup.py") ${fsFlgSetup} --nvidia'                 ; SetupType wsl2 = nvidia
        strCpy $0 '$WSLExeUser /bin/bash -ic "$0"'

    SetDetailsPrint listonly
    DetailPrint "[INFO] Executing: '$0'"
    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$0"                                                       ; Setup the faceswap environment
    pop $0
    ExecDos::wait $0
    pop $0

    strCmp $0 0 +4 0
        SetDetailsPrint both
        DetailPrint "[ERROR] Failed to setup faceswap environment"
        strCpy $FlgInstallFailed 1
FunctionEnd
