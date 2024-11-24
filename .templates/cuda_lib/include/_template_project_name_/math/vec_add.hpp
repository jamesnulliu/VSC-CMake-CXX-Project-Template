#pragma once

#include "_template_project_name_/_api.hpp"

namespace _template_project_name_::cuda
{
_template_project_name__API void launch_vec_add(const float* const a,
                                                const float* const b,
                                                float* const c, const int n);
}  // namespace _template_project_name_::cuda

namespace _template_project_name_::cpu
{
_template_project_name__API void launch_vec_add(const float* const a,
                                                const float* const b,
                                                float* const c, const int n);
}  // namespace _template_project_name_::cpu
