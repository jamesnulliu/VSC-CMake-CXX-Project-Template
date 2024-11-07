#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include <array>
#include <tuple>

namespace vsc_cpp_template
{

/**
 * @brief Compute the offset of a multi-dimensional array.
 *
 * @param args First half is the indexes, second half is the size of each
 *             dimension.
 * @return std::uint32_t The offset of the multi-dimensional array.
 *
 * @example computeOffset(1, 2, 3, 4, 5, 6) -> 3*1 + 2*6 + 1*6*5 = 45
 */
template <typename... ArgsT>
__host__ __device__ constexpr std::uint32_t computeOffset(ArgsT... args)
{
    constexpr size_t total_args = sizeof...(ArgsT);
    constexpr size_t dims = total_args / 2;

    auto params = std::make_tuple(static_cast<std::uint32_t>(args)...);

    std::uint32_t offset = 0;
    std::uint32_t stride = 1;

    [&]<size_t... I>(std::index_sequence<I...>) {
        ((I < dims ? (offset += std::get<dims - 1 - I>(params) * stride,
                      stride *= std::get<total_args - 1 - I>(params))
                   : 0),
         ...);
    }(std::make_index_sequence<dims>{});

    return offset;
}

}  // namespace vsc_cpp_template
