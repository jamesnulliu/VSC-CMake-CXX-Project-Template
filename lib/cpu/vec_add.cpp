#include <cstdio>

#include "vsc-cpp-template/math/vec_add.hpp"
#include "vsc-cpp-template/utils/address.hpp"

namespace vsc_cpp_template::cpu
{

void launch_vec_add(const float* const a, const float* const b, float* const c,
                    const int n)
{
    ::printf("Hello World from CPU!\n");
    ::printf("Vector size: %d\n", n);

    for (int i = 0; i < n; ++i) {
        auto offset = computeOffset<std::uint32_t>(1, 2, 3, 4, 5, 6);
        ::printf("Offset: %d\n", offset);
        c[i] = a[i] + b[i];
    }
}

}  // namespace vsc_cpp_template::cpu
