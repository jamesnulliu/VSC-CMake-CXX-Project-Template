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
# Others:
#   - `PROJECT_NAMESPACE`: The namespace for the project
####################################################################################################

# Common utility functions:
include(${PROJECT_SOURCE_DIR}/cmake/utils/common.cmake)

# [SEE] "./cmake/utils/common.cmake"
set_default_values(
    # CMake project namespace:
    # All the libs should have a alias with this namespace
    # [NOTE] Change this to your own namespace
    PROJECT_NAMESPACE "_template_project_name_::"
)

# Compiler configurations for cpp, e.g., g++, clang++, msvc(cl), etc.
include(${PROJECT_SOURCE_DIR}/cmake/compilers/cxx-compiler-configs.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)