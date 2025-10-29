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

RUNS=1000
TEST_DIR="perf_tests"
TESTS="perf1 perf2 perf3"

echo "Building tests..."
for test in $TESTS; do
    echo "Building $test with nvcc..."
    time nvcc -O3 -Xcompiler -fPIC --extended-lambda --expt-relaxed-constexpr \
      --generate-code arch=compute_75,code=sm_75 -rdc=true -Wno-deprecated-gpu-targets \
      "${TEST_DIR}/${test}.cpp" "${TEST_DIR}/${test}.cu" -o "${TEST_DIR}/${test}_nvcc"
    echo ""
    echo "Building $test with sf-nvcc..."
    time $SF_NVCC -O3 -Xcompiler -fPIC --extended-lambda --expt-relaxed-constexpr \
      --generate-code arch=compute_75,code=sm_75 -rdc=true -Wno-deprecated-gpu-targets \
      "${TEST_DIR}/${test}.cpp" "${TEST_DIR}/${test}.cu" -o "${TEST_DIR}/${test}_sfnvcc"
    echo ""
done

echo ""
echo "Running performance tests ($RUNS iterations each)..."
echo ""

for test in $TESTS; do
    echo "-------------------------------- Test: $test --------------------------------"

    echo -n "  nvcc:    "
    nvcc_time=$( (time -p bash -c "for i in \$(seq 1 $RUNS); do ${TEST_DIR}/${test}_nvcc > /dev/null; done") 2>&1 | awk '/real/ {print $2}' )
    nvcc_us=$(echo "$nvcc_time * 1000 / $RUNS" | bc)
    printf "%.2f ms (average)\n" $nvcc_us

    echo -n "  sf-nvcc: "
    sfnvcc_time=$( (time -p bash -c "for i in \$(seq 1 $RUNS); do LD_PRELOAD=cmake-build-Release/libsafecuda.so LD_LIBRARY_PATH=cmake-build-Release/ ${TEST_DIR}/${test}_sfnvcc > /dev/null; done") 2>&1 | awk '/real/ {print $2}' )
    sfnvcc_us=$(echo "$sfnvcc_time * 1000 / $RUNS" | bc)
    printf "%.2f ms (average)\n" $sfnvcc_us

    echo ""
    echo ""
done

echo "All tests done."
