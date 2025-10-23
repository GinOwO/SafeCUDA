/**
 * @file test_safecuda_runtime.cu
 * @brief Device-side implementation of SafeCUDA runtime tests
 *
 * Contains CUDA kernels and device functions for testing SafeCUDA's
 * bounds checking system, including valid access, out-of-bounds
 * detection, and freed memory detection.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-10-23
 * @version 1.0.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-10-23: Initial implementation
 */

#include "test_safecuda_runtime.h"
#include "test_safecuda_runtime.cuh"

#include "safecache.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

using namespace safecuda_runtime_tests;

__global__ void valid_access_kernel(float *data, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		data[idx] = static_cast<float>(idx) * 2.0f;
	}
}

__global__ void out_of_bounds_kernel(float *data, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0) {
		data[size + 10] = 42.0f;
	}
}

__global__ void freed_memory_kernel(float *data, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0) {
		data[0] = 42.0f;
	}
}

__global__ void interior_pointer_kernel(float *data, int size, int offset)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size - offset) {
		data[idx + offset] = static_cast<float>(idx);
	}
}

extern "C" void
safecuda_runtime_tests::launch_valid_access_kernel(float *d_data, int size)
{
	if (d_data == nullptr || size <= 0) {
		printf("Invalid parameters to launch_valid_access_kernel: ptr=%p, size=%d\n",
		       d_data, size);
		return;
	}

	int block_size = DEFAULT_BLOCK_SIZE;
	int grid_size = (size + block_size - 1) / block_size;

	if (grid_size > 65535) {
		throw std::runtime_error(
			"Array size too large for kernel launch");
	}

	valid_access_kernel<<<grid_size, block_size>>>(d_data, size);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Valid access kernel launch error: %s\n",
		       cudaGetErrorString(err));
		throw std::runtime_error(
			std::string("Valid access kernel launch failed: ") +
			cudaGetErrorString(err));
	}
}

extern "C" void
safecuda_runtime_tests::launch_out_of_bounds_kernel(float *d_data, int size)
{
	if (d_data == nullptr || size <= 0) {
		printf("Invalid parameters to launch_out_of_bounds_kernel: ptr=%p, size=%d\n",
		       d_data, size);
		return;
	}

	out_of_bounds_kernel<<<1, 1>>>(d_data, size);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Out of bounds kernel launch error: %s\n",
		       cudaGetErrorString(err));
		throw std::runtime_error(
			std::string("Out of bounds kernel launch failed: ") +
			cudaGetErrorString(err));
	}
}

extern "C" void
safecuda_runtime_tests::launch_freed_memory_kernel(float *d_data, int size)
{
	if (d_data == nullptr || size <= 0) {
		printf("Invalid parameters to launch_freed_memory_kernel: ptr=%p, size=%d\n",
		       d_data, size);
		return;
	}

	freed_memory_kernel<<<1, 1>>>(d_data, size);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Freed memory kernel launch error: %s\n",
		       cudaGetErrorString(err));
		throw std::runtime_error(
			std::string("Freed memory kernel launch failed: ") +
			cudaGetErrorString(err));
	}
}

extern "C" void
safecuda_runtime_tests::launch_interior_pointer_kernel(float *d_data, int size,
						       int offset)
{
	if (d_data == nullptr || size <= 0) {
		printf("Invalid parameters to launch_interior_pointer_kernel: ptr=%p, size=%d\n",
		       d_data, size);
		return;
	}

	int block_size = DEFAULT_BLOCK_SIZE;
	int grid_size = ((size - offset) + block_size - 1) / block_size;

	if (grid_size > 65535) {
		throw std::runtime_error(
			"Array size too large for kernel launch");
	}

	interior_pointer_kernel<<<grid_size, block_size>>>(d_data, size,
							   offset);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Interior pointer kernel launch error: %s\n",
		       cudaGetErrorString(err));
		throw std::runtime_error(
			std::string("Interior pointer kernel launch failed: ") +
			cudaGetErrorString(err));
	}
}

__global__ void manual_bounds_check_kernel(float *data, int size,
					   int should_fail)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0) {
		if (should_fail) {
			__bounds_check_safecuda(data + size + 100);
		} else {
			__bounds_check_safecuda(data);
		}
	}
}

extern "C" void
safecuda_runtime_tests::launch_manual_bounds_check(float *d_data, int size,
						   int should_fail)
{
	if (d_data == nullptr || size <= 0) {
		printf("Invalid parameters to launch_manual_bounds_check: ptr=%p, size=%d\n",
		       d_data, size);
		return;
	}

	manual_bounds_check_kernel<<<1, 1>>>(d_data, size, should_fail);
}
