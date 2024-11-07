# ==================================================================================================
# @file compiler-configs-cuda.cmake
# @brief Compiler configurations for cuda.
#
# @note Several parameters should be set BEFORE including this file:
#       1. `CUDA_CC`: `CUDA_CC` should be set to the path of the host cpp compiler. This is useful 
#          when the default cpp compiler is too new/old for the CUDA toolkit.
# ==================================================================================================

include(${CMAKE_CURRENT_LIST_DIR}/logging.cmake)

enable_language(CUDA)

if(WIN32)
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        log_fatal("You have to use MSVC for CUDA on Windows")
    endif()
endif()

if(NOT DEFINED CUDA_CC)
    set(CUDA_CC ${CMAKE_CXX_COMPILER})
    log_info("CUDA_CC is not set; Using default CXX compiler: ${CUDA_CC}")
endif()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
log_info("CMAKE_CUDA_STANDARD: ${CMAKE_CUDA_STANDARD}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin \"${CUDA_CC}\" --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -lineinfo")