# This example lib of cpu is always built
add_subdirectory(cpu)

# Build example lib of cuda only if `BUILD_CUDA_EXAMPLES` is ON
if (BUILD_CUDA_EXAMPLES)
    add_subdirectory(cuda)
endif()
