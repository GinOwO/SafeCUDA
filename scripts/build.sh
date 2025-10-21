#!/bin/bash
set -e

BUILD_TYPE=${1:-Debug}
BUILD_DIR="cmake-build-$BUILD_TYPE"
N_JOBS=${2:-$(nproc)}

export CUDA_PATH="/usr/local/cuda/bin"
# export CUDA_PATH="/opt/cuda/bin"

export NVCC_CCBIN="$CUDA_PATH/g++"
export NVCC_CBIN="$CUDA_PATH/gcc"
export PATH=$CUDA_PATH:$PATH

if [[ "$BUILD_TYPE" != "Debug" && "$BUILD_TYPE" != "Release" ]]; then
        echo "Error: Build type must be 'Debug' or 'Release'"
        echo "Usage: $0 [Debug|Release]"
        exit 1
fi

echo "Building SafeCUDA in $BUILD_TYPE mode..."

if ! command -v nvcc &>/dev/null; then
        echo "Error: nvcc not found. Please install CUDA development tools."
        exit 1
fi

echo "Using CUDA compiler: $(which nvcc)"
echo "CUDA version: $(nvcc --version | grep release)"
echo "g++ version: $(g++ --version)" 

mkdir -p "$BUILD_DIR"

cmake -G Ninja \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_CUDA_HOST_COMPILER="$NVCC_CCBIN" \
        -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets" \
		    -DCMAKE_CXX_COMPILER="$NVCC_CCBIN" \
		    -DCMAKE_C_COMPILER="$NVCC_CBIN" \
       -B "$BUILD_DIR"

ninja -C "$BUILD_DIR" -j"$N_JOBS"

echo "Build complete! Binaries are in $BUILD_DIR/"
echo "Run tests by running ctest in $BUILD_DIR/ or with the test script."
