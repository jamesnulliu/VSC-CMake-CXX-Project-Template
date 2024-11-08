set(TARGET_NAME "test")

file(GLOB_RECURSE CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

add_executable(${TARGET_NAME} ${CXX_SOURCES} ${CUDA_SOURCES})

target_link_libraries(
    ${TARGET_NAME} 
    PRIVATE 
    # @see "lib/cpu/CMakeLists.txt"
    ${PROJECT_NAMESPACE}templateCpuLib
    # @see "lib/cuda/CMakeLists.txt"
    $<$<BOOL:${BUILD_CUDA_EXAMPLES}>:${PROJECT_NAMESPACE}templateCudaLib>
)

target_compile_definitions(
    ${TARGET_NAME} 
    PRIVATE 
    $<$<BOOL:${BUILD_CUDA_EXAMPLES}>:BUILD_CUDA_EXAMPLES>
)