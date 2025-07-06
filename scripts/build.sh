#!/bin/bash
set -e

BUILD_TYPE=${1:-Debug}
BUILD_DIR="cmake-build-$BUILD_TYPE"
N_JOBS=${2:-$(nproc)}

export NVCC_CCBIN="/usr/local/cuda/bin/g++"

if [[ "$BUILD_TYPE" != "Debug" && "$BUILD_TYPE" != "Release" ]]; then
        echo "Error: Build type must be 'Debug' or 'Release'"
        echo "Usage: $0 [Debug|Release]"
        exit 1
fi

echo "Building SafeCUDA in $BUILD_TYPE mode..."

# Check if CUDA is available
if ! command -v nvcc &>/dev/null; then
        echo "Error: nvcc not found. Please install CUDA development tools."
        exit 1
fi

echo "Using CUDA compiler: $(which nvcc)"
echo "CUDA version: $(nvcc --version | grep release)"

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake -G Ninja \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_CUDA_HOST_COMPILER="$NVCC_CCBIN" \
        -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets" \
        ..

ninja -j"$N_JOBS"

echo "Build complete! Binaries are in $BUILD_DIR/"
echo "Run tests by running ctest in $BUILD_DIR/ or with the test script."
