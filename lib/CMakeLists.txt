set(LIB_NAME "templatelib")

file(GLOB_RECURSE CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

add_library(${LIB_NAME} ${CXX_SOURCES} ${CUDA_SOURCES})
add_library(${PROJECT_NAMESPACE}::${LIB_NAME} ALIAS ${LIB_NAME})

target_include_directories(
    ${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    ${LIB_NAME} 
    PUBLIC 
    CUDA::cudart 
    CUDA::cublas
)

# Sperate compilation for CUDA files
# set_target_properties(${LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install_libraries(${LIB_NAME} ${PROJECT_NAMESPACE} ${LIB_NAME})