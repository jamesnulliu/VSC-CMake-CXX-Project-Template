@echo off

set "ARCH=x86_amd64"
set "BUILD_TYPE=%1"

set "VS2022_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
if not exist "%VS2022_PATH%" (
    echo Visual Studio 2022 not found at: %VS2022_PATH%
    exit /b 1
)
set "VCVARSALL=%VS2022_PATH%\VC\Auxiliary\Build\vcvarsall.bat"
echo Setting up Visual Studio environment for %ARCH%...
call "%VCVARSALL%" %ARCH%


call cmake -G Ninja -S . -B ./build -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
call cmake --build ./build -j %NUMBER_OF_PROCESSORS%