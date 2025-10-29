#!/bin/bash

BUILD_TYPE="Release"
BUILD_DIR="cmake-build-$BUILD_TYPE"
SF_NVCC="$BUILD_DIR/bin/sf-nvcc"

if [ -d "/usr/local/cuda/bin" ]; then
    export CUDA_PATH="/usr/local/cuda/bin"
else
    export CUDA_PATH="/opt/cuda/bin"
fi

export NVCC_CCBIN="$CUDA_PATH/g++"
export NVCC_CBIN="$CUDA_PATH/gcc"
export PATH=$CUDA_PATH:$PATH

TEST_DIR="perf_tests"
TESTS="perf1 perf2 perf3 perf4 perf5 perf6"

echo "[BUILD] Type: $BUILD_TYPE"
echo "[BUILD] Directory: $BUILD_DIR"
echo "[BUILD] Using CUDA: $CUDA_PATH"
echo ""

for test in $TESTS; do
    echo "--------------------------------------------------------"
    echo "[NVCC] Building $test..."
    time nvcc -O3 -Xcompiler -fPIC --extended-lambda --expt-relaxed-constexpr \
        --generate-code arch=compute_75,code=sm_75 -rdc=true -Wno-deprecated-gpu-targets \
        "$TEST_DIR/$test.cpp" "$TEST_DIR/$test.cu" -o "$TEST_DIR/${test}_nvcc"
    echo ""
    echo "[SF-NVCC] Building $test..."
    time "$SF_NVCC" -O3 -Xcompiler -fPIC --extended-lambda --expt-relaxed-constexpr \
        --generate-code arch=compute_75,code=sm_75 -rdc=true -Wno-deprecated-gpu-targets \
        "$TEST_DIR/$test.cpp" "$TEST_DIR/$test.cu" -o "$TEST_DIR/${test}_sfnvcc"
    echo ""
done
echo "--------------------------------------------------------"

echo "[BUILD] All tests built successfully."

