# Minimum required CMake version
cmake_minimum_required(VERSION 3.30)

# Project Name
project(VSC-CPP-TEMPLATE VERSION 0.1.0 LANGUAGES CXX CUDA)

include(${PROJECT_SOURCE_DIR}/cmake/utils/compiler-configs.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/utils/lib-install.cmake)

include(${PROJECT_SOURCE_DIR}/cmake/libraries/libcuda.cmake)

set(PROJECT_NAMESPACE "vsc-cpp-template")

add_subdirectory(${PROJECT_SOURCE_DIR}/lib)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)