set(TARGET_NAME "main")

file(GLOB_RECURSE CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${TARGET_NAME} ${CXX_SOURCES} ${CUDA_SOURCES})

target_include_directories(
    ${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(
    ${TARGET_NAME} PROPERTIES
    # Force the output name to end with ".exe"
    EXECUTABLE_SUFFIX ".exe"
)