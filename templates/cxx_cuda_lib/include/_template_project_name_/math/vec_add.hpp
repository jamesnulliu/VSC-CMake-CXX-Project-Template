#pragma once

#include "_template_project_name_/_api.hpp"

namespace _template_project_name_::cuda {
VSC_CPP_TEMPLATE_API void launch_vec_add(const float *const a,
                                         const float *const b, float *const c,
                                         const int n);
} // namespace _template_project_name_::cuda

namespace _template_project_name_::cpu {
VSC_CPP_TEMPLATE_API void launch_vec_add(const float *const a,
                                         const float *const b, float *const c,
                                         const int n);
} // namespace _template_project_name_::cpu
