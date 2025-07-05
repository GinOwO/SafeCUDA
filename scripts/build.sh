#!/bin/bash
set -e

# Default build type
BUILD_TYPE=${1:-Debug}
BUILD_DIR="build"

export NVCC_CCBIN=/usr/bin/g+±14

# Validate build type
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

# Create build directory if it doesn't exist
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with explicit host compiler settings
cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_CUDA_HOST_COMPILER=gcc \
        -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets -D_GLIBCXX_USE_CXX11_ABI=1" \
        -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" \
        ..

ninja

echo "Build complete! Binaries are in $BUILD_DIR/"
echo "Run the test with: ./$BUILD_DIR/safecuda_test"
