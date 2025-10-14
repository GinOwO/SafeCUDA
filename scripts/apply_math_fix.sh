#!/usr/bin/bash
set -e
# This script applies a patch to fix CUDA math functions in the CUDA toolkit.
# Basically some math functions in the CUDA toolkit headers are declared differently
# which causes compilation errors. This patch was taken from:
# https://forums.developer.nvidia.com/t/error-exception-specification-is-incompatible-for-cospi-sinpi-cospif-sinpif-with-glibc-2-41/323591/3
# Run this script as sudo

patch -p1 -d /usr/local/cuda < cuda_math_fix.patch
