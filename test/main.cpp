#include <iostream>
#include <vector>

#include <project-name/math/vec_add.hpp>

int main()
{
    auto printVec = [](const std::vector<float>& vec) {
        for (const auto& v : vec) {
            std::cout << v << ", ";
        }
        std::cout << std::endl;
    };

    std::vector<float> a = {1, 2, 3};
    std::vector<float> b = {4, 5, 6};
    std::vector<float> c = {0, 0, 0};

    printVec(a);
    printVec(b);

    std::cout << __cplusplus << std::endl;

    project_namespace::cpu::launch_vec_add(a.data(), b.data(), c.data(),
                                           int(a.size()));
    printVec(c);

#if defined(BUILD_CUDA_EXAMPLES)
    project_namespace::cuda::launch_vec_add(a.data(), b.data(), c.data(),
                                            int(a.size()));
    printVec(c);
#endif

    return 0;
}