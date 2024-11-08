#include "vsc-cpp-template/system.hpp"

namespace vsc_cpp_template::cuda
{
VSC_CPP_TEMPLATE_API void launch_vec_add(float* a, float* b, float* c, int n);
}

namespace vsc_cpp_template::cpu
{
VSC_CPP_TEMPLATE_API void launch_vec_add(const float* const a,
                                         const float* const b, float* const c,
                                         const int n);
}
