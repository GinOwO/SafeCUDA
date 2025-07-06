# SafeCUDA
Software-Only Memory-Safety for GPUs via PTX Guards and Managed Shadow Metadata

## Pre-requisites
Install the following dependencies:
- gcc-14
- g++-14
- cmake
- ninja
- cuda-toolkit-12-9 (https://developer.nvidia.com/cuda-downloads)
- gtest

Note: Ensure that the CUDA toolkit is installed in `/usr/local/cuda`.

Note: Ensure that `gcc-14` and `g++-14` are symlinked to `gcc` and `g++` respectively in CUDA's `bin` directory.:
```bash
sudo ln -s /usr/bin/gcc-14 /usr/local/cuda/bin/gcc
sudo ln -s /usr/bin/g++-14 /usr/local/cuda/bin/g++
```

Note: If you encounter issues failing to compile due to math function specifications, apply the patch with (run as root):
```bash
sh ./scripts/apply_math_fix.sh
```


## Build Instructions
1. Clone the repository:
2. Navigate to the repository directory:
   ```bash
   cd SafeCUDA
   ```
3. Run the build script:
   ```bash
   sh ./scripts/build.sh [Debug|Release]
   ```
   Replace `[Debug|Release]` with your desired build type (default is Debug).
4. Test the build:
   ```bash
   sh ./scripts/test.sh
   ```

Or you can build and test using:
```bash
sh ./scripts/build_and_test.sh [Debug|Release]
```

## Formatting
Format the code using clang-format by running:
```bash
sh ./scripts/format.sh
```

Note: Please enable the the pre-commit hook which automatically formats code & test changes in Release build before committing. 
You can enable it by running:
```bash
sh ./scripts/enable_pre_commit_hook.sh
```