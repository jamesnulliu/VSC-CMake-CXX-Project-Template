#include <cuda_runtime.h>
#include "vsc-cpp-template/vec_add.hpp"

namespace vsc_cpp_template
{

__global__ void vec_add(float* a, float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void launch_vec_add(float* a, float* b, float* c, int n)
{
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    vec_add<<<grid_size, block_size>>>(a, b, c, n);
}

}  // namespace vsc_cpp_template