; Functions to install a new WSL2 distro

!include LogicLib.nsh

!define wslZipPS "$\"powershell.exe $\" Add-Type -assembly 'system.io.compression.filesystem'"      ; Enable zip utils in powershell

!define wslUbuntuWWW "https://aka.ms/wslubuntu"                                                     ; Ubuntu download URL
!define wslDebianWWW "https://aka.ms/wsl-debian-gnulinux"                                           ; Debian download URL
!define wslOpenSuseWWW "https://aka.ms/wsl-opensuse-tumbleweed"                                     ; openSUSE download URL
!define wslBundleZip "bundle.zip"                                                                   ; The name that the parent AppX bundle will be named to on download
!define wslDistroZip "distro.zip"                                                                   ; The name that the child AppX distro will be named to on extraction
!define wslImageArchive "install.tar.gz"                                                            ; The name of the WSL2 image within the child AppX distro
!define wslPkgsDeb "libx11-6 libxft2 libgl1-mesa-dev"                                               ; Debian/Ubuntu required packages
!define wslPkgsSUSE "libX11-6 libXft2 xorg-x11-fonts Mesa-libGL1"                                   ; OpenSUSE required packages
!define wslPkgsOracle "libX11 libXft mesa-libGL"                                                    ; OracleLinux required packages

var wslExe                                                                                          ; The wsl command executable to be run for executing linux commands. Set when we know the distro


; ========
; WSL Core
; ========
Function fnWSLKernelUpdate
    ; Update the kernel to latest version

    SetDetailsPrint both
    DetailPrint "[INFO] Updating WSL2 Kernel. This may take a few minutes..."
    ; Note: We don't output to log because, for some reason, WSL commands only outputs the first character from STDOUT :/

    strCpy $0 '"$WINDIR\Sysnative\wsl.exe" --update'
    SetDetailsPrint listonly
    DetailPrint "[INFO] Executing: '$0'"
    ExecDos::exec /NOUNLOAD /ASYNC '$0'                                                             ; Update Kernel to latest
    pop $0
    ExecDos::wait $0
    pop $0
    strCmp $0 0 0 doExit                                                                            ; Go to exit on failure

    strCpy $0 '"$WINDIR\Sysnative\wsl.exe" --set-default-version 2'
    SetDetailsPrint listonly
    DetailPrint "[INFO] Executing: '$0'"
    ExecDos::exec /NOUNLOAD /ASYNC '$0'                                                             ; Set default to WSL2
    pop $0
    ExecDos::wait $0
    pop $0

    doExit:
        push $0                                                                                     ; Push return code
FunctionEnd


; ==================
; WSL Distro Install
; ==================
Function fnWSLDistroDownload
    ; Download the selected WSL Distro to temp folder and rename to zip file

    strCmp $WSLInstallDistro "Ubuntu" 0 +2                                                          ; Get download URL
        strCpy $0 "${wslUbuntuWWW}"
    strCmp $WSLInstallDistro "Debian" 0 +2
        strCpy $0 "${wslDebianWWW}"
    strCmp $WSLInstallDistro "openSUSE-Tumbleweed" 0 +2
        strCpy $0 "${wslOpenSuseWWW}"

    SetDetailsPrint both
    DetailPrint "[INFO] Downloading WSL2 distro '$WSLInstallDistro'..."
    inetc::get /caption "Downloading WSL2 distro '$WSLInstallDistro'..." $0 "$DirTemp\${wslBundleZip}" /end
FunctionEnd

Function fnWSLDistroExtract
    ; Extract the install.tar.gz image from the downloaded appx distro file and cleanup unneeded files

    SetDetailsPrint both
    DetailPrint "[INFO] Extracting WSL2 distro '$WSLInstallDistro'..."
    SetDetailsPrint listonly

    strCpy $0 "${wslZipPS} ; [io.compression.zipfile]::OpenRead('$DirTemp\${wslBundleZip}').Entries.Name"
    push "ExecDos::End"

    SetDetailsPrint listonly
    DetailPrint "[INFO] Executing: '$0'"
    ExecDos::exec /NOUNLOAD /TOSTACK $0 "" ""                                                       ; List contents of archive
    pop $0
    strCmp $0 0 0 doFailure

    Loop:
        pop $0
        strCmp $0 "ExecDos::End" notFound
        ${StrContains} $1 "x64.appx" $0                                                             ; Locate the x64 Distro version
        strCmp $1 "" Loop
        Goto exitLoop

    notFound:
        DetailPrint "[INFO] Distro bundle not found in archive. Searching in parent"
        Rename "$DirTemp\${wslBundleZip}" "$DirTemp\${wslDistroZip}"                                ; Rename bundle.zip to distro.zip to look for install.tar.gz in parent
        strCpy $R0 "${wslBundleZip}"
        goto extractImage

    exitLoop:
    strCpy $R0 $0
    DetailPrint "Located Distro bundle: '$R0'"
    DetailPrint "Extracting '$R0' from '${wslBundleZip}'"

    strCpy $0 "$$zip = [IO.Compression.ZipFile]::OpenRead('$DirTemp\${wslBundleZip}')"
    strCpy $0 "$0 ; $$exfile = $$zip.Entries.Where({ $$_.FullName -eq '$R0' }, 'First')"
    strCpy $0 "$0 ; [IO.Compression.ZipFileExtensions]::ExtractToFile( $$exfile[ 0 ], '$DirTemp\${wslDistroZip}' )"
    strCpy $0 "${wslZipPS} ; $0"

    SetDetailsPrint listonly
    DetailPrint "[INFO] Executing: '$0'"
    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED $0 "" ""                                               ; Extract x64 distro
    pop $0
    ExecDos::wait $0
    pop $0
    strCmp $0 0 +3 0
        DetailPrint "[ERROR] Failed to extract '$R0' from archive '${wslBundleZip}'."
        goto doFailure

    Delete '$DirTemp\${wslBundleZip}'                                                               ; Delete bundle.zip

    extractImage:
    DetailPrint "Extracting '${wslImageArchive}' from '$R0'"
    strCpy $0 "$$zip = [IO.Compression.ZipFile]::OpenRead('$DirTemp\${wslDistroZip}')"
    strCpy $0 "$0 ; $$exfile = $$zip.Entries.Where({ $$_.FullName -eq '${wslImageArchive}' }, 'First')"
    strCpy $0 "$0 ; [IO.Compression.ZipFileExtensions]::ExtractToFile( $$exfile[ 0 ], '$DirTemp\${wslImageArchive}' )"
    strCpy $0 "${wslZipPS} ; $0"

    SetDetailsPrint listonly
    DetailPrint "[INFO] Executing: '$0'"
    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED $0 "" ""                                               ; Extract compressed install file
    pop $0
    ExecDos::wait $0
    pop $0
    strCmp $0 0 +3 0
        DetailPrint "[ERROR] Failed to extract '${wslImageArchive}' from archive '${wslDistroZip}'."
        goto doFailure

    Delete '$DirTemp\${wslDistroZip}'                                                               ; Delete distro.zip
    push $0
    return

    doFailure:
        push 1
FunctionEnd

Function fnWSLDistroImport
    ; Import the downloaded install.tar.gz file into WSL2

    SetDetailsPrint both
    DetailPrint "[INFO] Importing WSL2 distro '$WSLInstallDistro'. This may take a while..."

    strCpy $0 '"wsl.exe" --import "$WSLDistro" "$LocalAppdata\Packages\$WSLDistro" "$DirTemp\${wslImageArchive}"'
    SetDetailsPrint listonly
    DetailPrint "[INFO] Executing: '$0'"
    ExecDos::exec /NOUNLOAD /ASYNC $0 "" ""                                                         ; Import distro to WSL2
    pop $0
    ExecDos::wait $0
    pop $0

    strCmp $0 0 0 +2
        Delete '$DirTemp\${wslImageArchive}'                                                        ; Delete install.tar.gz
    push $0                                                                                         ; push WSL.exe return code
FunctionEnd

; ====================
; WSL Distro Configure
; ====================
Function fnWSLDistroSetupUser
    ; Setup the user for the distro
    SetDetailsPrint both
    DetailPrint "[INFO] Creating user '$WSLUser' for WSL2 distro '$WSLInstallDistro'..."

    strCpy $0 'useradd -m -s /bin/bash $WSLUser &&'                                                 ; Add user
    strCpy $0 '$0 echo $\'$WSLUser:$WSLPassword$\' | chpasswd'                                      ; Set password
    strCmp $WSLInstallDistro "openSUSE-Tumbleweed" +2 0                                             ; openSUSE is not default configured for sudo. Skip as beyond scope to set it up here
        strCpy $0 ' && $0 usermod -aG sudo $WSLUser'                                                ; Add to correct superuser group
    strCpy $0 '$wslExe /bin/bash -c "$0"'                                                           ; Execute on WSL

    ; Don't log execution here as password is passed
    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED $0 "" ""                                               ; Create user with password and add to superusers
    pop $0
    ExecDos::wait $0                                                                                ; Return code popped in parent
FunctionEnd

Function fnWSLDistroUpdate
    ; Update the distro's packages

    SetDetailsPrint both
    DetailPrint "[INFO] Updating WSL2 distro '$WSLInstallDistro'. This may take a while..."

    strCpy $0 '$wslExe apt-get update && apt-get upgrade -y && apt-get autoremove --purge -y'       ; Debian/Ubuntu update command
    strCmp $WSLInstallDistro "openSUSE-Tumbleweed" 0 +2
        strCpy $0 '$wslExe zypper --gpg-auto-import-keys refresh && zypper -n up && zypper -n dup'  ; openSUSE update command

    SetDetailsPrint listonly
    DetailPrint "[INFO] Executing: '$0'"
    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED $0 "" ""                                               ; Update distro packages
    pop $0
    ExecDos::wait $0
FunctionEnd

Function fnWSLDistroShutdown
    ; Shutdown the distro. Should be done after updata and new user

    SetDetailsPrint both
    DetailPrint "[INFO] Finalizing WSL2 distro '$WSLInstallDistro'..."

    strCpy $0 '"wsl.exe" -t $WSLDistro'

    SetDetailsPrint listonly
    DetailPrint "[INFO] Executing: '$0'"
    ExecDos::exec /NOUNLOAD /ASYNC $0 "" ""                                                         ; Terminate distro
    pop $0
    ExecDos::wait $0
FunctionEnd


; ============
; Entry Points
; ============
Function fnWSLDistroInstall
    ; Download and install the requested WSL2 distro

    ; Note: It is better to manage this directly with wsl.exe. However, wsl.exe does not currently give a method
    ; to install a distro without prompting the user for a default username and password, so we have to do a more
    ; convoluted manual install

    strCmp $WSLInstallDistro "" 0 +2                                                                ; Skip if WSL2 Distro does not need installing
        return

    strCpy $WSLDistro "$WSLInstallDistro-faceswap"                                                  ; Set the name of the distro to global $WSLDistro
    strCpy $WSLExeUser '"wsl.exe" -d $WSLDistro -u $WSLUser --'                                     ; Store command for executing on WSL distro as user.
    strCpy $wslExe '"wsl.exe" -d $WSLDistro -u root --'                                             ; Set the wsl executable prefix

    call fnWSLKernelUpdate                                                                          ; Update WSL2 Kernel to latest
    pop $0
    strCmp $0 0 0 onError

    call fnWSLDistroDownload                                                                        ; Download distro
    pop $0
    strCmp $0 "OK" 0 onError

    call fnWSLDistroExtract                                                                         ; Extract install.tar.gz file from downloaded bundle
    pop $0
    strCmp $0 0 0 onError

    call fnWSLDistroImport                                                                          ; Import the install.tar.gz file into WSL2
    pop $0
    strCmp $0 0 0 onError

    call fnWSLDistroSetupUser                                                                       ; Setup the distro's user
    pop $0
    strCmp $0 0 0 onError

    call fnWSLDistroUpdate                                                                          ; Update the distro's packages
    pop $0
    strCmp $0 0 +2 0
        DetailPrint "[WARNING] WSL2 Distro: '$WSLInstallDistro' could not be updated"

    call fnWSLDistroShutdown                                                                        ; Shutdown Distro once configured
    pop $0
    strCmp $0 0 +2 0
        DetailPrint "[WARNING] WSL2 Distro: '$WSLInstallDistro' could not be terminated"

    DetailPrint "[INFO] Installed WSL2 Distro: '$WSLInstallDistro'"
    return

    onError:
        DetailPrint "[ERROR] WSL2 Distro: '$WSLInstallDistro' could not be installed"
        strCpy $FlgInstallFailed 1
FunctionEnd

Function fnWSLDistroPkgInstall
    ; Install required system packages for running faceswap. These are required for
    ; the GUI and for OpenCV
    SetDetailsPrint both
    DetailPrint "[INFO] Installing required packages to WSL2 distro '$WSLDistro'..."

    strCpy $R0 '"wsl.exe" -d $WSLDistro -u root --'                                                 ; Prefix to execute WSL2 command in distro as root

    ${StrContains} $0 "ubuntu" $WSLDistro                                                           ; Test for Ubuntu distro
        strCmp $0 "" 0 installDeb
    ${StrContains} $0 "debian" $WSLDistro                                                           ; Test for Debian distro
        strCmp $0 "" 0 installDeb
    ${StrContains} $0 "kali" $WSLDistro                                                             ; Test for kali distro
        strCmp $0 "" 0 installDeb
    ${StrContains} $0 "suse" $WSLDistro                                                             ; Test for openSUSE distro
        strCmp $0 "" 0 installSUSE
    ${StrContains} $0 "oracle" $WSLDistro                                                           ; Test for OracleLinux distro
        strCmp $0 "" 0 installOracle

    SetDetailsPrint listonly                                                                        ; If couldn't match distro: post warning to log and exit
    DetailPrint "[WARNING] Could not detect Distro to install required packages"
    DetailPrint "[WARNING] Some system packages may need to be installed manually"
    return

    installDeb:
        strCpy $1 '${wslPkgsDeb}'
        strCpy $0 '$R0 apt-get update && apt-get install --no-install-recommends -y $1'             ; Install command for debian based distros
        goto runCmd

    installSUSE:
        strCpy $1 '${wslPkgsSUSE}'
        strCpy $0 '$R0 zypper --gpg-auto-import-keys refresh && zypper -n in $1'                    ; Install command for SUSE based distros
        goto runCmd

    installOracle:
        strCpy $1 '${wslPkgsOracle}'
        strCpy $0 '$R0 yum install -y $1'                                                           ; Install command for Oracle based distros
        goto runCmd

    runCmd:
        SetDetailsPrint listonly
        DetailPrint '[INFO] Executing "$0"'
        ExecDos::exec /NOUNLOAD /ASYNC /DETAILED $0 "" ""                                           ; Install required packages (if needed)
        pop $0
        ExecDos::wait $0
        pop $0

    strCmp $0 0 +4                                                                                  ; Output error on failure, but don't abort
        SetDetailsPrint both
        DetailPrint "[ERROR] Could not install '$1' to WSL2 Distro"
        DetailPrint "[ERROR] You may need to install these packages manually"

FunctionEnd

