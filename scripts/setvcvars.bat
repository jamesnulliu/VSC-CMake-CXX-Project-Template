@echo off

set "VS2022_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
if not exist "%VS2022_PATH%" (
    echo Visual Studio 2022 not found at: %VS2022_PATH%
    exit /b 1
)

set "VCVARSALL=%VS2022_PATH%\VC\Auxiliary\Build\vcvarsall.bat"

set "ARCH=x86_amd64"
if not "%1" == "" (
    set "ARCH=%1"
)

echo Setting up Visual Studio environment for %ARCH%...
call "%VCVARSALL%" %ARCH%

call "bash"