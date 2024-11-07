#include "vsc-cpp-template/vec_add.hpp"
#include <vector>

int main()
{
    std::vector<float> a = {1, 2, 3};
    std::vector<float> b = {4, 5, 6};
    std::vector<float> c = {0, 0, 0};

    vsc_cpp_template::launch_vec_add(a.data(), b.data(), c.data(), a.size());
}