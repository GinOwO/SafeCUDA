# SafeCUDA

Software-Only Memory-Safety for GPUs via PTX Guards and Managed Shadow Metadata

## Introduction

`sf-nvcc` is a drop-in replacement for `nvcc` that automatically injects spatial
and temporal memory safety checks into CUDA kernels.

It instruments generated PTX to track and validate global memory accesses at
runtime, catching out-of-bounds and temporal access violations in global memory
in cuda code, killing the kernel and raising a runtime error whenever
`cudaDeviceSynchronize` or `cudaGetLastError` are called.

It wraps some of the main CUDA functions such as `cudaMalloc`,
`cudaMallocManaged`, `cudaFree` etc and generates a table in memory with each
entry taking 32bytes (for alignment reasons) and performs a bounds check during
runtime.

It's split into two parts, a compile time part `sf-nvcc` which is responsible
for instrumenting the ptx with the bounds_check mechanism and a runtime part
`libsafecuda.so` which acts as a drop in solution by letting the user just
overload `LD_PRELOAD` to call the weapper CUDA functions.

## Pre-requisites

Install the following dependencies:

- gcc-14
- g++-14
- cmake
- ninja
- cuda-toolkit-12-9 (https://developer.nvidia.com/cuda-downloads)
- gtest
- gtest-devel
- zlib

Note: Ensure that the CUDA toolkit is installed in `/usr/local/cuda` or
`/opt/cuda/`.

Note: Ensure that `gcc-14` and `g++-14` are symlinked to `gcc` and `g++`
respectively in CUDA's `bin` directory.

```bash
sudo ln -s /usr/bin/gcc-14 /usr/local/cuda/bin/gcc
sudo ln -s /usr/bin/g++-14 /usr/local/cuda/bin/g++
```

Note: If you encounter issues failing to compile due to math function
specifications ([this is a known issue](https://forums.developer.nvidia.com/t/error-exception-specification-is-incompatible-for-cospi-sinpi-cospif-sinpif-with-glibc-2-41/323591)), apply the patch with (run as root):

```bash
sh ./scripts/apply_math_fix.sh
```

## Build Instructions

1. Clone the repository
2. Navigate to the repository directory:
   ```bash
   cd SafeCUDA
   ```
3. Run the build script:
   ```bash
   sh ./scripts/build.sh [Debug|Release]
   ```
   Replace `[Debug|Release]` with your desired build type (default is Debug).
    1. If you want you can test the build using:
       ```bash
       sh ./scripts/test.sh
       ```
       and run a test example using
       ```bash
       sh ./scripts/build_and_run_example.sh
       ```
       Or you can build and run ctest using:

       ```bash
       sh ./scripts/build_and_test.sh [Debug|Release]
       ```

## Usage

1. After building, you can use `sf-nvcc` from either of the build directories (
   examples will use the release build) to build your cuda program.

   ```bash
   cmake-build-Release/sf-nvcc [SafeCUDA options] <standard nvcc args>
   ```
   For example:
   ```bash
   cmake-build-Release/bin/sf-nvcc -Xcompiler -fPIC --extended-lambda --expt-relaxed-constexpr --generate-code arch=compute_75,code=sm_75 -rdc=true examples/example.cpp examples/out_of_bounds.cu -o example -Wno-deprecated-gpu-targets -sf-keep-dir "cmake-build-Debug/examples" -sf-debug true
   ```

2. After successful compilation with `sf-nvcc` you can run your executable after
   exporting these variables in your terminal session:
   ```bash
   export LD_PRELOAD=cmake-build-Release/libsafecuda.so:$LD_PRELOAD
   export LD_LIBRARY_PATH=cmake-build-Release/:$LD_LIBRARY_PATH
   <your-executable>
   ```
   For example (if you followed the previous build example):
   ```bash
   export LD_PRELOAD=cmake-build-Release/libsafecuda.so:$LD_PRELOAD
   export LD_LIBRARY_PATH=cmake-build-Release/:$LD_LIBRARY_PATH
   ./example
   ```

Note: In the current build the following cuda functions are wrapped

1. cudaMalloc
2. cudaManagedMalloc
3. cudaFree
4. cudaDeviceSynchronize
5. cudaLaunchKernel
6. cudaGetLastError

However the real functions can be accessed by simply prepending a `real_` before
their names, for example `real_cudaMalloc`

### SF-NVCC arguments

Here's a cheatsheet for the sf-nvcc specific arguments, a detailed description
can be obtained from `sf-nvcc -sf-help`

| Option                          | Description                                 |
|---------------------------------|---------------------------------------------|
| `-sf-help`                      | Show SafeCUDA help                          |
| `-sf-version`                   | Print version/build info                    |
| `-sf-debug <true \| FALSE>`     | Enable detailed PTX instrumentation logging |
| `-sf-verbose <true \| FALSE>`   | Show full compile output                    |
| `-sf-fail-fast <TRUE \| false>` | Abort on first violation                    |
| `-sf-keep-dir <path>`           | Keep intermediate build files               |

Note: The following nvcc args are stripped out:

```text
-dryrun
--keep
--keep-dir
-lcudart_static
```

## Benchmarking

A test suite with some representative workload kernels are present in the
`perf_tests` folder. You can use the runner script `perf_tests/run_perf.py` to
benchmark and get the results as a csv file in
`perf_tests/results/perf_results_<timestamp>.csv`

| Test  | Description                                         |
|-------|-----------------------------------------------------|
| perf1 | Large vector ops                                    |
| perf2 | Parallel sum reduction                              |
| perf3 | Memory copy + scaling                               |
| perf4 | Realistic HPC / ML compute loop                     |
| perf5 | Synthetic memory stress (~4 GB traffic)             |
| perf6 | Multi-allocation test (1000 buffers, ~1 GB traffic) |

## Contributing

Format the code using clang-format by running:

```bash
sh ./scripts/format.sh
```

Note: Please enable the the pre-commit hook which automatically formats code &
test changes in Release build before committing.
You can enable it by running:

```bash
sh ./scripts/enable_pre_commit_hook.sh
```

## Known Issues

1. Use of curand functions will trigger a SIGSEGV on the device due to how
   nullptrs are used in there
