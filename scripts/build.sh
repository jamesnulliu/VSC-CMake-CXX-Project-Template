# Env Variables: CC, CXX, NVCC_CCBIN

# @func  windows_prune_env_path
# @brief Prune the PATH environment variable to only include necessary paths.
#        This is necessary because the PATH environment variable on Windows
#        can get very long and cause issues with the CMake generator.
#
# @note  You should modify the function to include the paths that are necessary
#        for your build.
windows_prune_env_path() {
    # Return if not windows
    if [[ "$OSTYPE" != "msys" ]]; then return; fi

    echo "[build.sh] Pruning PATH environment variable..."

    local CMAKE_DIR=$(dirname $(which cmake))
    local NINJA_DIR=$(dirname $(which ninja))

    IFS=':' read -r -a paths <<< "$PATH"
    local new_path=""

    # Include only the necessary paths
    # You may want to add "conda" to the list if you are using Anaconda/Miniconda
    local -a include_paths=("/usr/bin" "cuda" "windows kits" "microsoft visual studio")
    local -a filtered_paths
    for path in "${paths[@]}"; do
        local lowercase_path=$(echo "$path" | tr '[:upper:]' '[:lower:]')
        for keyword in "${include_paths[@]}"; do
            if [[ "$lowercase_path" == *"$keyword"* ]]; then
                filtered_paths+=("$path")
                break
            fi
        done
    done

    # Insert the CMake and Ninja directories at the beginning of PATH
    local filtered_paths=($CMAKE_DIR $NINJA_DIR "${filtered_paths[@]}")

    # Remove duplicates
    local -A unique_paths
    local new_path=""
    for path in "${filtered_paths[@]}"; do
        if [[ -z "${unique_paths[$path]}" ]]; then
            if [[ -n "$new_path" ]]; then
                new_path+=":"
            fi
            new_path+="$path"
            unique_paths[$path]=1
        fi
    done
    
    # Update the PATH environment variable
    PATH=$new_path

    echo "[build.sh] PATH environment variable pruned."
}

windows_prune_env_path

SOURCE_DIR=.
BUILD_DIR=./build
BUILD_TYPE=Release
CXX_STANDARD=20
CUDA_STANDARD=20
BUILD_SHARED_LIBS=OFF
BUILD_CUDA_EXAMPLES=OFF

while [[ $# -gt 0 ]]; do
    case $1 in
        -S|--source-dir)
            SOURCE_DIR=$2; shift ;;
        -B|--build-dir)
            BUILD_DIR=$2; shift ;;
        Release|Debug)
            BUILD_TYPE=$1 ;;
        --stdc++=*)
            CXX_STANDARD="${1#*=}" ;;
        --stdcuda=*)
            CUDA_STANDARD="${1#*=}" ;;
        --shared)
            BUILD_SHARED_LIBS=ON ;;
        --build-cuda-examples)
            BUILD_CUDA_EXAMPLES=ON ;;
        *)
            # [TODO] Add detailed help message
            echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

cmake -G Ninja -S $SOURCE_DIR -B $BUILD_DIR \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CXX_STANDARD=$CXX_STANDARD \
    -DCMAKE_CUDA_STANDARD=$CUDA_STANDARD \
    -DBUILD_SHARED_LIBS=$BUILD_SHARED_LIBS \
    -DBUILD_CUDA_EXAMPLES=$BUILD_CUDA_EXAMPLES

cmake --build $BUILD_DIR -j $(nproc)