#pragma once

#include <cstdio>
#include <cuda.h>

namespace vsc_cpp_template
{

template <typename... Args>
__host__ __device__ void log(const char* fmt, Args... args)
{
    printf(fmt, args...);
}

}  // namespace vsc_cpp_template