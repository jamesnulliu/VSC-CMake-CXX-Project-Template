@echo off

@REM [NOTE] Change the path to your Visual Studio 2022 installation
set "VS2022_PATH=%ProgramFiles%\Microsoft Visual Studio\2022\Community"
if not exist "%VS2022_PATH%" (
    echo Visual Studio 2022 not found at: %VS2022_PATH%
    exit /b 1
)

set "ARCH=x86_amd64"

set "VCVARSALL=%VS2022_PATH%\VC\Auxiliary\Build\vcvarsall.bat"
echo Setting up Visual Studio environment for %ARCH%...
call "%VCVARSALL%" %ARCH%
