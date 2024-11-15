# Minimum required CMake version
cmake_minimum_required(VERSION 3.30)

# Project Name and version
project(VSC-CMAKE-CXX-TEMPLATE VERSION 1.0.0)

####################################################################################################
# [!] All CMake Options and Variables MAY Affect the Build Process
# 
# In "cmake/utils/logging.cmake":
#   - `LOG_PREFIX`: The prefix for logging messages
# 
# In "cmake/compilers/cxx-compiler-configs.cmake":
#   - `ENV{CC}`: The compiler for C code
#   - `ENV{CXX}`: The compiler for C++ code
#   - `CMAKE_CXX_STANDARD`: The standard for C++ code
#   - `STACK_SIZE`: The stack size for the executable
#
# In "cmake/compilers/cuda-compiler-configs.cmake":
#   - `ENV{NVCC_CCBIN}`: The CUDA compiler bindir
#   - `CMAKE_CUDA_STANDARD`: The standard for CUDA code
# 
# In "cmake/utils/run-python.cmake":
#   - `ENV{PYTHON_EXECUTABLE}`: The path to the Python executable
# 
# In "cmake/libraries/cuda.cmake":
#   - `ENV{CUDA_HOME}`: The path to the CUDA toolkit
#
# Others:
#   - `BUILD_CUDA_EXAMPLES`: Whether to build CUDA examples
#   - `PROJECT_NAMESPACE`: The namespace for the project
#   - `BUILD_SHARED_LIBS`: Whether to build shared libraries
####################################################################################################

# Common utility functions:
include(${PROJECT_SOURCE_DIR}/cmake/utils/common.cmake)

# @see "./cmake/utils/variables.cmake"
set_default_values(
    # Whether to build CUDA examples; 
    # [NOTE] Set to OFF if you don't have CUDA installed
    BUILD_CUDA_EXAMPLES OFF

    # CMake project namespace:
    # All the libs should have a alias with this namespace
    # [NOTE] Change this to your own namespace
    PROJECT_NAMESPACE "cmake-project-namespace::"
)

# Compiler configurations for cpp, e.g., g++, clang++, msvc(cl), etc.
include(${PROJECT_SOURCE_DIR}/cmake/compilers/cxx-compiler-configs.cmake)

# Utility functions for installing libraries:
include(${PROJECT_SOURCE_DIR}/cmake/utils/lib-install.cmake)

if (BUILD_CUDA_EXAMPLES)
    # Compiler configurations for CUDA 
    include(${PROJECT_SOURCE_DIR}/cmake/compilers/cuda-compiler-configs.cmake)
    # Some preparational steps for linking CUDA libraries:
    include(${PROJECT_SOURCE_DIR}/cmake/libraries/libcuda.cmake)
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/lib)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)