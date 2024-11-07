#include <cstdint>

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>

#include "vsc-cpp-template/vec_add.hpp"

#include "vsc-cpp-template/utils/cuda/address.cuh"
#include "vsc-cpp-template/utils/cuda/logging.cuh"

namespace vsc_cpp_template
{

__global__ void vec_add(float* a, float* b, float* c, int n)
{
    const int threadIndex = threadIdx.x;
    std::uint32_t streamingMultiprocessorId;
    asm("mov.u32 %0, %smid;" : "=r"(streamingMultiprocessorId));
    std::uint32_t warpId;
    asm volatile("mov.u32 %0, %warpid;" : "=r"(warpId));
    std::uint32_t laneId;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(laneId));

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        log("Thread %d: %f + %f = %f\n", i, a[i], b[i], a[i] + b[i]);
        c[i] = a[i] + b[i];
        auto offset = computeOffset(1, 2, 3, 4, 5, 6);
        log("Offset: %d\n", offset);
        log("SM: %d | Warp: %d | Lane: %d | Thread %d - Here!\n",
            streamingMultiprocessorId, warpId, laneId, threadIndex);
    }
}

void launch_vec_add(float* a, float* b, float* c, int n)
{
    printf("Hello World from CUDA!\n");
    printf("Vector size: %d\n", n);
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    // Apply Device Memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**) &d_a, n * sizeof(float));
    cudaMalloc((void**) &d_b, n * sizeof(float));
    cudaMalloc((void**) &d_c, n * sizeof(float));

    // Copy Host Memory to Device Memory
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch Kernel
    vec_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

    // Copy Device Memory to Host Memory
    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free Device Memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

}  // namespace vsc_cpp_template