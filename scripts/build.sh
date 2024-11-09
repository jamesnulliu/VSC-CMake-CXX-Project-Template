# Env Variables: CXX, NVCC_CCBIN

BUILD_TYPE=Release
CXX_STANDARD=20
CUDA_STANDARD=20
BUILD_SHARED_LIBS=OFF
BUILD_CUDA_EXAMPLES=OFF

while [[ $# -gt 0 ]]; do
    case $1 in
        Release|Debug)
            BUILD_TYPE=$1 ;;
        --stdc++=*)
            CXX_STANDARD="${1#*=}" ;;
        --stdcuda=*)
            CUDA_STANDARD="${1#*=}" ;;
        --shared)
            BUILD_SHARED_LIBS=ON ;;
        --cuda-examples)
            BUILD_CUDA_EXAMPLES=ON ;;
        *)
            # [TODO] Add detailed help message
            echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

cmake -G Ninja -S . -B ./build \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CXX_STANDARD=$CXX_STANDARD \
    -DCMAKE_CUDA_STANDARD=$CUDA_STANDARD \
    -DBUILD_SHARED_LIBS=$BUILD_SHARED_LIBS \
    -DBUILD_CUDA_EXAMPLES=$BUILD_CUDA_EXAMPLES

cmake --build ./build -j $(nproc)