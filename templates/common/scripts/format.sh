# [NOTE] 
# |- This script is used to format the code using clang-format.
# |- Although CI pipeline is configured to format the code after pushing to main,
# |- it is still recommended to run this script manually before committing the code.

clang-format -i $(find .  \
                -name '*.c' -o -name '*.h'  \
                -o -name '*.cpp' -o -name '*.hpp'  \
                -o -name '*.cu' -o -name '*.cuh')