#!/bin/bash
set -e

BUILD_TYPE=${1:-Release}
BUILD_DIR="cmake-build-$BUILD_TYPE"
N_JOBS=${2:-$(nproc)}
SF_NVCC="$BUILD_DIR/bin/sf-nvcc"

echo "-----------------------------------------------"

sh ./scripts/build.sh Release

echo "-----------------------------------------------"
echo ""

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
#nvcc -O3 -Xcompiler -fPIC --extended-lambda --expt-relaxed-constexpr --generate-code arch=compute_60,code=sm_60 --generate-code arch=compute_61,code=sm_61 --generate-code arch=compute_75,code=sm_75 --generate-code arch=compute_86,code=sm_86 examples/example.cpp examples/kernel1.cu examples/kernel2.cu examples/montecarlo.cu examples/out_of_bounds.cu -o example_nvcc -Wno-deprecated-gpu-targets

echo "-----------------------------------------------"

echo "Building example (sf-nvcc)"
$SF_NVCC -O0 -G -g -Xcompiler -fPIC --extended-lambda --expt-relaxed-constexpr --generate-code arch=compute_75,code=sm_75 -rdc=true examples/example.cpp examples/out_of_bounds.cu -o example -Wno-deprecated-gpu-targets -sf-keep-dir "cmake-build-Debug/examples" -sf-debug true

echo "-----------------------------------------------"

echo ""
echo "Running NVCC ver (should not catch the OOB):"
./example_nvcc

echo "-----------------------------------------------"
echo ""
echo "Running SF-NVCC ver (should throw an exception):"
export LD_PRELOAD=cmake-build-Release/libsafecuda.so:$LD_PRELOAD
export LD_LIBRARY_PATH=cmake-build-Release/:$LD_LIBRARY_PATH
export SAFECUDA_THROW_OOB=1
./example
