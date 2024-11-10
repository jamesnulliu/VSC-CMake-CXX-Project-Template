#include <cstdio>

#include "project-name/math/vec_add.hpp"
#include "project-name/utils/address.hpp"

namespace project_namespace::cpu
{

void launch_vec_add(const float* const a, const float* const b, float* const c,
                    const int n)
{
    ::printf("Hello World from CPU!\n");
    ::printf("Vector size: %d\n", n);

    for (int i = 0; i < n; ++i) {
        auto offset = computeOffset<std::size_t>(1, 2, 3, 4, 5, 6);
        ::printf("Offset: %d\n", std::uint32_t(offset));
        c[i] = a[i] + b[i];
    }
}

}  // namespace project_namespace::cpu
