#!/bin/bash
set -e

BUILD_TYPE=${1:-Release}
BUILD_DIR="cmake-build-$BUILD_TYPE"
SF_NVCC="$BUILD_DIR/bin/sf-nvcc"

echo "-----------------------------------------------"

sh ./scripts/build.sh Release

echo "-----------------------------------------------"
echo ""

if [ -d "/usr/local/cuda/bin" ]; then
    export CUDA_PATH="/usr/local/cuda/bin"
else
    export CUDA_PATH="/opt/cuda/bin"
fi


export NVCC_CCBIN="$CUDA_PATH/g++"
export NVCC_CBIN="$CUDA_PATH/gcc"
export PATH=$CUDA_PATH:$PATH

if [[ "$BUILD_TYPE" != "Debug" && "$BUILD_TYPE" != "Release" ]]; then
        echo "Error: Build type must be 'Debug' or 'Release'"
        echo "Usage: $0 [Debug|Release]"
        exit 1
fi

echo "Building Examples in $BUILD_TYPE mode..."

if ! command -v nvcc &>/dev/null; then
        echo "Error: nvcc not found. Please install CUDA development tools."
        exit 1
fi

echo "Using CUDA compiler: $(which nvcc)"
echo "Using SF_NVCC: $SF_NVCC"
echo "CUDA version: $(nvcc --version | grep release)"
echo "g++ version: $(g++ --version)"

echo "-----------------------------------------------"

echo "Building example_nvcc"
nvcc -O3 -Xcompiler -fPIC --extended-lambda --expt-relaxed-constexpr \
  --generate-code arch=compute_75,code=sm_75 -rdc=true -Wno-deprecated-gpu-targets \
  examples/example.cpp examples/kernel.cu \
  -o example_nvcc

echo "-----------------------------------------------"

echo "Building example (sf-nvcc)"
$SF_NVCC -O3 -Xcompiler -fPIC --extended-lambda --expt-relaxed-constexpr \
  --generate-code arch=compute_75,code=sm_75 -rdc=true -Wno-deprecated-gpu-targets \
  examples/example.cpp examples/kernel.cu -sf-verbose true \
  -o example

echo "-----------------------------------------------"

echo ""
echo "Running NVCC ver (should not catch the OOB):"
time ./example_nvcc

echo "-----------------------------------------------"
echo ""
echo "Running SF-NVCC ver (should throw an exception):"
export LD_PRELOAD=cmake-build-Release/libsafecuda.so:$LD_PRELOAD
export LD_LIBRARY_PATH=cmake-build-Release/:$LD_LIBRARY_PATH
time ./example
