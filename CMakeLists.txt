# Minimum required CMake version
cmake_minimum_required(VERSION 3.30)

# Project Name and version
project(VSC-CPP-TEMPLATE VERSION 0.1.0)

####################################################################################################
# [!] All CMake Options and Variables MAY Affect the Build Process
# 
# In "cmake/utils/logging.cmake":
#   - `LOG_PREFIX`: The prefix for logging messages
# 
# In "cmake/utils/compiler-configs-cpp.cmake":
#   - `ENV{CXX}`: The compiler for C++ code
#   - `CMAKE_CXX_STANDARD`: The standard for C++ code
#   - `STACK_SIZE`: The stack size for the executable
#
# In "cmake/utils/compiler-configs-cuda.cmake":
#   - `ENV{NVCC_CCBIN}`: The CDUA compiler bindir
#   - `CMAKE_CUDA_STANDARD`: The standard for CUDA code
# 
# In "cmake/utils/run-python.cmake":
#   - `ENV{PYTHON_EXECUTABLE}`: The path to the Python executable
# 
# In "cmake/libraries/cuda.cmake":
#  - `ENV{CUDA_HOME}`: The path to the CUDA toolkit
#
# Others:
#  - `BUILD_CUDA_EXAMPLES`: Whether to build CUDA examples
#  - `PROJECT_NAMESPACE`: The namespace for the project
#  - `BUILD_SHARED_LIBS`: Whether to build shared libraries
####################################################################################################

# Utility functions for setting values:
include(${PROJECT_SOURCE_DIR}/cmake/utils/variables.cmake)

# @see "./cmake/utils/variables.cmake"
set_default_values(
    # Whether to build CUDA examples; 
    # Set to OFF if you don't have CUDA installed
    BUILD_CUDA_EXAMPLES ON

    # Namespace for the project;
    # All the libs should have a alias with this namespace
    PROJECT_NAMESPACE "vsc-cpp-template::"
)

# Compiler configurations for cpp, e.g., g++, clang++, msvc(cl), etc.
include(${PROJECT_SOURCE_DIR}/cmake/utils/compiler-configs-cpp.cmake)

# Utility functions for installing libraries:
include(${PROJECT_SOURCE_DIR}/cmake/utils/lib-install.cmake)

if (BUILD_CUDA_EXAMPLES)
    # Compiler configurations for CUDA 
    include(${PROJECT_SOURCE_DIR}/cmake/utils/compiler-configs-cuda.cmake)
    # Some preparational steps for linking CUDA libraries:
    include(${PROJECT_SOURCE_DIR}/cmake/libraries/libcuda.cmake)
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/lib)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)