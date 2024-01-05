; Faceswap NSIS Windows Installer

RequestExecutionLevel Admin                             ; Needed for potential WSL2 install


;=========================
; Definitions and includes
;=========================
; system includes
!include MUI2.nsh
!include nsDialogs.nsh
!include winmessages.nsh
!include LogicLib.nsh
!include CPUFeatures.nsh
!include FileFunc.nsh

; Util includes
!include utlMultiDetailPrint.nsi
!include utlExplode.nsi
!include utlString.nsi

; Global variables required in local includes
var FlgIsResuming                                       ; Flag to indicate that the install is resuming after enabling WSL2
var FlgInstallWSL                                       ; Flag to indicate if WSL2 Windows features need to be enabled
var FlgInstallConda                                     ; Flag for whether to install conda
var FlgInstallFailed                                    ; Flag to indicate if a component has failed to install

var Log                                                 ; Logging for detail print
var FntBold                                             ; Config for bold font
var DirTemp                                             ; Temporary working folder

var SetupType                                           ; Version of fs to install
var EnvName                                             ; Name of conda env
var WSLUser                                             ; The name of the linux user for WSL2
var WSLPassword                                         ; The linux password for WSL2
var WSLInstallDistro                                    ; WSL2 Distro to install. Empty if none to install
var WSLDistro                                           ; The name of the WSL2 Distro that will be used
var WSLExeUser                                          ; Command to run WSL.EXE as a specific user
var CondaDir                                            ; Location of conda install
var CondaBin                                            ; Location of conda executable/activation script

; Local includes
!include pgVersionSelect.nsi                            ; Custom fs version selection page
!include wslApp.nsi                                     ; Selecting and installing WSL2 application (Windows features)
!include pgWSLDistroSelect.nsi                          ; Selecting an existing or new WSL2 distro
!include pgWSLDistroInfo.nsi                            ; Collects information about the selected installed WSL2 distro
!include pgCustomize.nsi                                ; Custom customize install page

!include instWSLDistro.nsi                              ; Install a new WSL2 Distro
!include instConda.nsi                                  ; Install and configure Conda
!include instFaceswap.nsi                               ; Install and configure Faceswap
!include instFinalize.nsi                               ; Set launcher and desktop shortcut

; Installer names and locations
OutFile "faceswap_setup_x64.exe"
Name "faceswap"
InstallDir $PROFILE\faceswap

; global variables only used in this file
!define wwwFaceswap "https://www.faceswap.dev"          ; Final website url
var hasAVX                                              ; CPU Check AVX
var hasSSE4                                             ; CPU Check SSE4

; ==========
; Modern UI2
; ==========
; General options
!define MUI_COMPONENTSPAGE_NODESC
!define MUI_ABORTWARNING                                ; Prompt for confirmation on cancel
!define MUI_ICON "fs_logo.ico"

; Welcome page
!define MUI_WELCOMEPAGE_TITLE "faceswap"
!define MUI_WELCOMEPAGE_TEXT "Welcome to the faceswap installer.$\r$\n$\r$\n\
    This installer will guide you through the process of installing faceswap on your Windows machine.$\r$\n$\r$\n\
    The installer needs to download resources from the internet, so please ensure that you have a stable internet connection.$\r$\n$\r$\n\
    faceswap can be run on Nvidia, AMD and Intel GPUs as well as on CPU.$\r$\n$\r$\n\
    Please ensure you choose the correct options for your setup."
!insertmacro MUI_PAGE_WELCOME

; Version select
page custom pgVersionSelect

; WSL2
page custom pgWSLSelect pgWSLSelectLeave
page custom pgWSLInstallConfirm
page custom pgWSLDistroInstall pgWSLDistroInstallLeave
page custom pgWSLScan pgWSLScanLeave

; Install Location Page
!define MUI_PAGE_HEADER_TEXT "faceswap Installer"
!define MUI_PAGE_HEADER_SUBTEXT "Install Location"
!define MUI_PAGE_CUSTOMFUNCTION_PRE onSelectDestination
!define MUI_DIRECTORYPAGE_TEXT_DESTINATION "Select Destination Folder:"
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE VerifyInstallDir
!insertmacro MUI_PAGE_DIRECTORY

; Install Customize and Finalize pages
page custom pgCustomize pgCustomizeLeave
page custom pgFinalize

; Install faceswap Page
!define MUI_PAGE_CUSTOMFUNCTION_SHOW InstFilesShow      ; Enable Cancel button
!define MUI_PAGE_HEADER_SUBTEXT "Installing faceswap..."
!insertmacro MUI_PAGE_INSTFILES

; Set language (or Modern UI doesn't work)
!insertmacro MUI_LANGUAGE "English"


; =====================
; Installation sections
; =====================
Section -StartInstallation
    ; Output logging information so far
    push $Log                                           ; Output logging info so far
    call MultiDetailPrint                               ; Print log to details window
SectionEnd

Section InstallWSL2
    ; Install WSL2 and/or intall a WSL2 distro. Install system packages to requested distro
    strCmp $SetupType "wsl2" +2 0                       ; Skip for non-wsl2
        return

    call fnWSLInstall                                   ; Enable Windows WSL2 features (process will end here and reboot if required)
    call fnWSLDistroInstall                             ; Install a WSL2 distro
    strCmp $FlgInstallFailed 1 0 +2
        call Abort                                      ; Abort on failure

    call fnWSLDistroPkgInstall                          ; Install required system packages
SectionEnd

Section InstallConda
    ; Install Conda to correct location
    strCmp $FlgInstallConda 1 +2 0                      ; Skip if Conda does not need installing
        return

    call fnCondaDownload                                ; Download Conda
    strCmp $FlgInstallFailed 1 0 +2
        call Abort                                      ; Abort on failure

    call fnCondaInstall                                 ; Install Conda
    strCmp $FlgInstallFailed 1 0 +2
        call Abort                                      ; Abort on failure
SectionEnd

Section ConfigureConda
    ; Set up the Conda virtual environment + install Git
    call fnCondaSetupEnvironment                        ; Delete any pre-existing environment and setup a new one for faceswap
    strCmp $FlgInstallFailed 1 0 +2
        call Abort                                      ; Abort on failure

    call fnCondaInstallGit                              ; Install git into faceswap's Conda environment
    strCmp $FlgInstallFailed 1 0 +2
        call Abort                                      ; Abort on failure
SectionEnd

Section InstallFaceswap
    ; Clone faceswap repo and run setup
    call fnFsCloneRepo                                  ; Clone repo into install folder
    strCmp $FlgInstallFailed 1 0 +2
        call Abort                                      ; Abort on failure

    call fnFsConfigure                                  ; Set up the Faceswap environment
    strCmp $FlgInstallFailed 1 0 +2
        call Abort                                      ; Abort on failure
SectionEnd

Section Finalize
    ; Add a launch for the Faceswap GUI, a desktop shortcut and pop open the webpage
    call fnFnlAddGuiLauncher                            ; Add a .bat file for launching straight into the GUI
    call fnFnlDesktopShortcut                           ; Add a desktop shortcut for the launcher
    ExecShell "open" "${wwwFaceswap}"                   ; Open web page
    DetailPrint "Visit ${wwwFaceswap} for help and support."
SectionEnd


; ==============
; Initialization
; ==============
Function InstFilesShow
    ; Enable the cancel button during installation
    GetDlgItem $0 $HWNDPARENT 2
    EnableWindow $0 1
FunctionEnd

Function CheckCPU
    ; Basic system checks
    push $PROFILE                                       ; Check for spaces in paths. Conda doesn't like them
        call CheckForSpaces  ; TODO warn user if this triggers
    pop $R0

    ; CPU Capabilities
    ${If} ${CPUSupports} "AVX2"
    ${OrIf} ${CPUSupports} "AVX1"
        strCpy $Log "$Log[INFO] CPU Supports AVX Instructions$\n"
        strCpy $hasAVX 1
    ${EndIf}
    ${If} ${CPUSupports} "SSE4.2"
    ${OrIf} ${CPUSupports} "SSE4"
        strCpy $Log "$Log[INFO] CPU Supports SSE4 Instructions$\n"
        strCpy $hasSSE4 1
    ${EndIf}
FunctionEnd

Function .onInit
    SetShellVarContext current
    InitPluginsDir                                      ; It's better to put stuff in $pluginsdir, $temp is shared
    strCpy $DirTemp "$pluginsdir\faceswap\temp"
    strCpy $EnvName "faceswap"
    SetOutPath "$DirTemp"
    CreateFont $FntBold "MS Shell Dlg" "8.25" "700"     ; custom font for bold headers

    ; == On Resume from reboot ==
    ClearErrors
    ${GetParameters} $0                                 ; Get passed in switches
    ${GetOptions} $0 "/resume" $1                       ; Check if /resume exists
    IfErrors +3 0
        strCpy $FlgIsResuming 1                         ; Set the resuming flag
        strCpy $SetupType "wsl2"                        ; Set backend type to WSL2

    call CheckCPU                                       ; Standard CPU capability check
FunctionEnd


; ============================
; Destination folder selection
; ============================
Function onSelectDestination
    ; Don't enable selection of install folder when installing WSL2
    strCmp $FlgInstallWSL 1 0 +2
        abort
FunctionEnd

Function VerifyInstallDir
    ; Check install folder does not already exist
    IfFileExists  $INSTDIR 0 +3
    MessageBox MB_OK "Destination directory exists. Please select an alternative location"
    abort
FunctionEnd


; ==========
; On Failure
; ==========
Function Abort
    ; Raise message if there was an install failure
    MessageBox MB_OK "Some applications failed to install. Process Aborted. Check Details."
    abort "Install Aborted"
FunctionEnd
