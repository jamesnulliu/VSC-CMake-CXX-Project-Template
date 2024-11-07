# ==================================================================================================
# @file compiler-configs.cmake
# @brief Compiler configurations
#
# @note Several parameters should be set before including this file:
#       1. `STACK_SIZE`: Stack size for the executable. The default value is 1048576 (1MB).
#       2. `CUDA_CC`: `CUDA_CC` should be set to the path of the host compiler. This is useful when 
#          your default `CXX` compiler is too new for the CUDA toolkit.
# ==================================================================================================
include(${CMAKE_CURRENT_LIST_DIR}/logging.cmake)

# Generate compile_commands.json in build directory
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
log_info("CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")

# Set stack size
if(NOT DEFINED STACK_SIZE)
    set(STACK_SIZE 1048576)  # 1MB by default
endif()

# Compiler flags for MSVC
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/permissive- /Zc:forScope /openmp)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi")
    # Set stack size
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:${STACK_SIZE}")
# Compiler flags for Clang
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-fopenmp)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    # Set stack size
    if(WIN32)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,/STACK:${STACK_SIZE}")
    else()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-zstack-size=${STACK_SIZE}")
    endif()
# Compiler flags for GNU
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fopenmp)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    # Set stack size
    if(WIN32)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--stack,${STACK_SIZE}")
    else()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-zstack-size=${STACK_SIZE}")
    endif()
# @todo jamesnulliu
# |- Add compiler flags for other compilers
else()
    log_fatal("Unsupported compiler")
endif()


if(NOT DEFINED CUDA_CC)
    set(CUDA_CC ${CMAKE_CXX_COMPILER})
    log_warning("CUDA_CC is not set; Using default CXX compiler: ${CUDA_CC}")
endif()
log_info("CUDA_CC: ${CUDA_CC}")

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
log_info("CMAKE_CUDA_STANDARD: ${CMAKE_CUDA_STANDARD}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CUDA_CC}")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -lineinfo")