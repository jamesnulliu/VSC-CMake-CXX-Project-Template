BuildType=$1

cmake -G Ninja -S . -B ./build \
    -DCMAKE_BUILD_TYPE=$BuildType  

cmake --build ./build -j $(nproc)