# Minimum required CMake version
cmake_minimum_required(VERSION 3.30)

# Project Name and version
project(VSC-CMAKE-CXX-TEMPLATE VERSION 1.0.0)

####################################################################################################
# All CMake Options and Variables MAY Affect the Build Process
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
# In "cmake/libraries/cuda.cmake":
#   - `ENV{CUDA_HOME}`: The path to the CUDA toolkit
#
# Others:
#   - `PROJECT_NAMESPACE`: The namespace for the project
#   - `BUILD_SHARED_LIBS`: Whether to build shared libraries
####################################################################################################

# Common utility functions:
include(${PROJECT_SOURCE_DIR}/cmake/utils/common.cmake)

# [SEE] "./cmake/utils/variables.cmake"
set_default_values(
    # CMake project namespace:
    # All the libs should have a alias with this namespace
    # [NOTE] Change this to your own namespace
    PROJECT_NAMESPACE "_template_project_name_::"
)

# Utility functions for installing libraries:
include(${PROJECT_SOURCE_DIR}/cmake/utils/lib-install.cmake)

# Compiler configurations for cpp, e.g., g++, clang++, msvc(cl), etc.
include(${PROJECT_SOURCE_DIR}/cmake/compilers/cxx-compiler-configs.cmake)
# Compiler configurations for CUDA 
include(${PROJECT_SOURCE_DIR}/cmake/compilers/cuda-compiler-configs.cmake)
# Some preparational steps for linking CUDA libraries:
include(${PROJECT_SOURCE_DIR}/cmake/libraries/libcuda.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/lib)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)