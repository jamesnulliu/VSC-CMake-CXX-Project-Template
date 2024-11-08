@echo off

set "BUILD_TYPE=%1" 
set "CXX_STANDARD=20"
set "CUDA_STANDARD=20"
set "BUILD_SHARED_LIBS=OFF"
set "BUILD_CUDA_EXAMPLES=OFF"

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

call cmake -G Ninja -S . -B ./build  ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE%   ^
    -DCMAKE_CXX_STANDARD=%CXX_STANDARD%  ^
    -DCMAKE_CUDA_STANDARD=%CUDA_STANDARD%  ^
    -DBUILD_CUDA_EXAMPLES=%BUILD_CUDA_EXAMPLES%  ^
    -DBUILD_SHARED_LIBS=%BUILD_SHARED_LIBS%

set "NUMBER_OF_PROCESSORS=8"

call cmake --build ./build -j %NUMBER_OF_PROCESSORS%

call cmake --install ./build --prefix ./build/install
