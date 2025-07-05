# SafeCUDA
Software-Only Memory-Safety for GPUs via PTX Guards and Managed Shadow Metadata

## Pre-requisites
Install the following dependencies:
- gcc-14
- g++-14
- cmake
- ninja
- cuda-toolkit-12-9 (https://developer.nvidia.com/cuda-downloads)

Note: Ensure that the CUDA toolkit is installed in `/usr/local/cuda`.
Note: Ensure that `gcc-14` and `g++-14` are symlinked to `gcc` and `g++` respectively in CUDA's `bin` directory.:
```bash
sudo ln -s /usr/bin/gcc-14 /usr/local/cuda/bin/gcc
sudo ln -s /usr/bin/g++-14 /usr/local/cuda/bin/g++
```

## Build Instructions
1. Clone the repository:
2. Navigate to the repository directory:
   ```bash
   cd SafeCUDA
   ```
3. Run the build script:
   ```bash
   ./scripts/build.sh [Debug|Release]
   ```
   Replace `[Debug|Release]` with your desired build type (default is Debug).
4. If you encounter issues with math functions, apply the patch:
   ```bash
   ./scripts/apply_math_fix.sh
   ```
5. Test the build:
   ```bash
   ./build/safecuda_test
   ```