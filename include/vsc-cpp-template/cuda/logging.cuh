#pragma once

#include <cstdio>
#include <cuda_runtime.h>

namespace vsc_cpp_template::cuda
{

/**
 * @brief Device log.
 */
template <typename... Args>
__device__ void dLog(const char* fmt, Args... args)
{
    ::printf(fmt, args...);
}

}  // namespace vsc_cpp_template::cuda