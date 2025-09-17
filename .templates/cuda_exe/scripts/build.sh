#!/bin/bash
# Env Variables: CC, CXX, NVCC_CCBIN
set -e

SOURCE_DIR=.
BUILD_DIR=./build
BUILD_TYPE=Release
CXX_STANDARD=20
CUDA_STANDARD=20
GENERATOR=Ninja

function print_help() {
    echo "Usage: build.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help"
    echo "    Show this help message"
    echo "  -S, --source-dir <dir-path>"
    echo "    [optional] Specify the source directory. Default: \"$SOURCE_DIR\""
    echo "  -B, --build-dir <dir-path>"
    echo "    [optional] Specify the build directory. Default: \"$BUILD_DIR\""
    echo "  -G, --generator <generator-name>"
    echo "    [optional] Specify the CMake generator. Default: \"$GENERATOR\""
    echo "  Release|Debug|RelWithDebInfo|RD  (RD=RelWithDebInfo)"
    echo "    [optional] Specify the build type. Default: \"$BUILD_TYPE\""
    echo "  --stdc++=<version>"
    echo "    [optional] Specify the C++ standard to use. Default: \"$CXX_STANDARD\""
    echo "  --stdcuda=<version>"
    echo "    [optional] Specify the CUDA standard to use. Default: \"$CUDA_STANDARD\""
    echo "  --rm-build-dir"
    echo "    [optional] Remove the build directory before building"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -S|--source-dir)
            SOURCE_DIR=$2; shift ;;
        -B|--build-dir)
            BUILD_DIR=$2; shift ;;
        -G|--generator)
            GENERATOR=$2; shift ;;
        Release|Debug|RelWithDebInfo|RD)
            BUILD_TYPE=${1/RD/RelWithDebInfo} ;;
        --stdc++=*)
            CXX_STANDARD="${1#*=}" ;;
        --stdcuda=*)
            CUDA_STANDARD="${1#*=}" ;;
        --rm-build-dir)
            rm -rf $BUILD_DIR ;;
        -h|--help)
            print_help; exit 0 ;;
        *)
            echo "[build.sh][ERROR] Unknown argument: $1"
            print_help; exit 1 ;;
    esac
    shift
done

# Prune PATH for Windows
if [[ "$OSTYPE" == "msys" ]]; then source ./scripts/windows-prune-PATH.sh; fi

# Check if stdout is terminal -> enable/disable colored output
if [ -t 1 ]; then 
    STDOUT_IS_TERMINAL=ON; export GTEST_COLOR=yes
else
    STDOUT_IS_TERMINAL=OFF; export GTEST_COLOR=no
fi

CMAKE_TOOL_CHAIN_FILE=""
# Check and set CMAKE_TOOL_CHAIN_FILE to vcpkg.cmake
if [ -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ]; then
    CMAKE_TOOL_CHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
elif [ -f "$VCPKG_HOME/scripts/buildsystems/vcpkg.cmake" ]; then
    CMAKE_TOOL_CHAIN_FILE="$VCPKG_HOME/scripts/buildsystems/vcpkg.cmake"
else
    echo "[build.sh][ERROR] ENV:VCPKG_ROOT or ENV:VCPKG_HOME is not set or " \
        "vcpkg.cmake is not found. Please install vcpkg and set VCPKG_ROOT " \
        "or VCPKG_HOME to the vcpkg root directory"
    exit 1
fi

cmake -G $GENERATOR -S $SOURCE_DIR -B $BUILD_DIR \
    -DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOL_CHAIN_FILE" \
    -DSTDOUT_IS_TERMINAL=$STDOUT_IS_TERMINAL \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CXX_STANDARD=$CXX_STANDARD \
    -DCMAKE_CUDA_STANDARD=$CUDA_STANDARD

cmake --build $BUILD_DIR -j $(nproc)