# ==================================================================================================
# @file compiler-configs-cpp.cmake
# @brief Compiler configurations for the host.
#
# @note Several parameters SHOULD be set BEFORE including this file:
#         - `ENV:CXX`: C++ Compiler. Default: auto-detected.
#         - `CMAKE_CXX_STANDARD`: C++ Standard. Default: 20.
#         - `STACK_SIZE`: Stack size for the executable. Default: 1048576 (1MB).
# ==================================================================================================

include(${CMAKE_CURRENT_LIST_DIR}/logging.cmake)

enable_language(CXX)

# Generate compile_commands.json in build directory
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
log_info("CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")

# Set stack size
if(NOT DEFINED STACK_SIZE)
    set(STACK_SIZE 1048576)  # 1MB by default
endif()

# Compiler flags for MSVC
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    string(APPEND CMAKE_CXX_FLAGS "/permissive- /Zc:forScope /openmp /Zc:__cplusplus")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " /O2")
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " /Zi")
    # Set stack size
    string(APPEND CMAKE_EXE_LINKER_FLAGS " /STACK:${STACK_SIZE}")
# Compiler flags for Clang
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    string(APPEND CMAKE_CXX_FLAGS "-fopenmp -Wall -Wextra -Werror")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g")
    # Set stack size
    string(APPEND CMAKE_EXE_LINKER_FLAGS 
        $<$<PLATFORM_ID:Windows>:-Wl,--stack,${STACK_SIZE}> 
        $<$<NOT:$<PLATFORM_ID:Windows>>:-Wl,-zstack-size=${STACK_SIZE}>)
# Compiler flags for GNU
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    string(APPEND CMAKE_CXX_FLAGS "-fopenmp -Wall -Wextra -Werror")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g")
    # Set stack size
    string(APPEND CMAKE_EXE_LINKER_FLAGS 
        $<$<PLATFORM_ID:Windows>:-Wl,--stack,${STACK_SIZE}> 
        $<$<NOT:$<PLATFORM_ID:Windows>>:-Wl,-zstack-size=${STACK_SIZE}>)
# [TODO] @jamesnulliu
# |- Add compiler flags for other compilers
else()
    log_fatal("Unsupported compiler")
endif()
