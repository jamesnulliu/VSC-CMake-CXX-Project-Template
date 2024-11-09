#pragma once

#include <cstdio>
#include <cuda_runtime.h>

namespace project_namespace::cuda
{

/**
 * @brief Device log.
 */
template <typename... Args>
__device__ void dLog(const char* fmt, Args... args)
{
    ::printf(fmt, args...);
}

}  // namespace project_namespace::cuda