#include <array>
#include <iostream>
#include <vector>

#include <_template_project_name_/math/vec_add.hpp>

auto main() -> int {
  auto printVec = [](const auto &vec) -> void {
    for (const auto &v : vec) {
      std::cout << v << ", ";
    }
    std::cout << std::endl;
  };

  auto a = std::array<float, 3>{1, 2, 3};
  auto b = std::array<float, 3>{4, 5, 6};
  auto c = std::array<float, 3>{0, 0, 0};

  printVec(a);
  printVec(b);

  std::cout << __cplusplus << std::endl;

  _template_project_name_::launch_vec_add(a.data(), b.data(), c.data(),
                                          int(a.size()));
  printVec(c);

  return 0;
}