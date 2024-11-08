#include <iostream>
#include <vector>

#include <vsc-cpp-template/math/vec_add.hpp>

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

    vsc_cpp_template::cpu::launch_vec_add(a.data(), b.data(), c.data(),
                                          a.size());
    printVec(c);

#if defined(BUILD_CUDA_EXAMPLES)
    vsc_cpp_template::cuda::launch_vec_add(a.data(), b.data(), c.data(),
                                           a.size());
    printVec(c);
#endif

    return 0;
}