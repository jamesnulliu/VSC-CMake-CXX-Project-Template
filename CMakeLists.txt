# Minimum required CMake version
cmake_minimum_required(VERSION 3.30)

# Project Name
project(VSC-CPP-TEMPLATE VERSION 0.1.0 LANGUAGES CXX CUDA)

# Compiler configurations for host, e.g., gcc, clang, msvc:
include(${PROJECT_SOURCE_DIR}/cmake/utils/compiler-configs-host.cmake)

# Compiler configurations for CUDA; 
# [NOTE] Remove this if not developing CUDA code:
include(${PROJECT_SOURCE_DIR}/cmake/utils/compiler-configs-cuda.cmake)

# Utility functions for installing libraries:
include(${PROJECT_SOURCE_DIR}/cmake/utils/lib-install.cmake)

# Some preparational steps for linking CUDA libraries:
include(${PROJECT_SOURCE_DIR}/cmake/libraries/libcuda.cmake)

set(PROJECT_NAMESPACE "vsc-cpp-template")

add_subdirectory(${PROJECT_SOURCE_DIR}/lib)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)