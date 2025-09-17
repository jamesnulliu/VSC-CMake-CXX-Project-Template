@echo off

@REM [NOTE] Change the path to your Visual Studio installation directory
set "VS_INSTALL_DIR=%ProgramFiles%\Microsoft Visual Studio"

if not exist "%VS_INSTALL_DIR%" (
    echo [msvc-setup.bat][Error] Visual Studio not found at: "%VS_INSTALL_DIR%"
    echo [msvc-setup.bat][Note] Please modify the path in this script to match your Visual Studio installation.
    exit /b 1
)


set "VS2026_INSIDER_DIR=%ProgramFiles%\Microsoft Visual Studio\18\Insiders"
set "VS2022_DIR=%ProgramFiles%\Microsoft Visual Studio\2022"

@REM Check for Visual Studio 2026 Insider
if exist "%VS2026_INSIDER_DIR%" (
    set "VS_BASE_DIR=%VS2026_INSIDER_DIR%"
    goto END
)
echo [msvc-setup.bat] Visual Studio 2026 Insider not found at: "%VS2026_INSIDER_DIR%"

@REM Check for Visual Studio 2022 Enterprise
if exist "%VS2022_DIR%\Enterprise" (
    set "VS_BASE_DIR=%VS2022_DIR%\Enterprise"
    goto END
)
echo [msvc-setup.bat] Visual Studio 2022 Enterprise not found at: "%VS2022_DIR%\Enterprise"

@REM Check for Visual Studio 2022 Professional
if exist "%VS2022_DIR%\Professional" (
    set "VS_BASE_DIR=%VS2022_DIR%\Professional"
    goto END
)
echo [msvc-setup.bat] Visual Studio 2022 Professional not found at: "%VS2022_DIR%\Professional"

@REM Check for Visual Studio 2022 Community
if exist "%VS2022_DIR%\Community" (
    set "VS_BASE_DIR=%VS2022_DIR%\Community"
    goto END
)
echo [msvc-setup.bat] Visual Studio 2022 Community not found at: "%VS2022_DIR%\Community"

echo [msvc-setup.bat][Error] No valid Visual Studio installation found.
exit /b 1

:END

echo [msvc-setup.bat] Using Visual Studio installation at: "%VS_BASE_DIR%"
set "ARCH=x86_amd64"
set "VCVARSALL=%VS_BASE_DIR%\VC\Auxiliary\Build\vcvarsall.bat"
echo [msvc-setup.bat] Setting up Visual Studio environment for %ARCH%...
call "%VCVARSALL%" %ARCH%