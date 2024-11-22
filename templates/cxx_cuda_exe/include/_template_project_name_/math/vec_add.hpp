#pragma once

namespace _template_project_name_::cuda
{
void launch_vec_add(const float* const a,
                                         const float* const b, float* const c,
                                         const int n);
}  // namespace _template_project_name_::cuda

namespace _template_project_name_::cpu
{
void launch_vec_add(const float* const a,
                                         const float* const b, float* const c,
                                         const int n);
}  // namespace _template_project_name_::cpu
