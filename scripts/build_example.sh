#!/bin/bash
set -e

BUILD_TYPE=${1:-Debug}
BUILD_DIR="cmake-build-$BUILD_TYPE"
N_JOBS=${2:-$(nproc)}

export PATH=/usr/local/cuda/bin:$PATH
export NVCC_CCBIN="/usr/local/cuda/bin/g++"

echo "Building SafeCUDA in $BUILD_TYPE mode..."

if ! command -v nvcc &>/dev/null; then
        echo "Error: nvcc not found. Please install CUDA development tools."
		echo "$PATH"
        exit 1
fi

echo "Using CUDA compiler: $(which nvcc)"
echo "CUDA version: $(nvcc --version | grep release)"

mkdir -p "$BUILD_DIR/examples"

cmake -G Ninja \
        -DCMAKE_CUDA_HOST_COMPILER="$NVCC_CCBIN" \
        -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets" \
        -DCUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
      -S examples \
      -B "$BUILD_DIR/examples"

ninja -C "$BUILD_DIR/examples" -j"$N_JOBS"

echo "Examples built! Binaries are in $BUILD_DIR/"
