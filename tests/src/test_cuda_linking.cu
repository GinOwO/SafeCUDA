/**
 * @file test_cuda_linking.cu
 * @brief Device-side implementation of CUDA linking tests
 * 
 * Contains CUDA kernels and device functions for testing GPU
 * functionality, compilation pipeline, and execution correctness.
 * 
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-07-06
 * @version 1.0.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 * 
 * Change Log:
 * - 2025-09-23: Rewrote some comments
 * - 2025-07-06: Initial implementation
 */

#include "test_cuda_linking.h"
#include "test_cuda_linking.cuh"

#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace cuda_linking_tests;

/**
 * @brief Simple CUDA kernel that doubles each array element
 * 
 * Basic arithmetic kernel designed to verify CUDA compilation and
 * execution pipeline. Uses straightforward memory access patterns
 * and simple arithmetic to test fundamental GPU execution.
 * 
 * @param data Device pointer to floating-point array
 * @param size Number of elements in the array
 */
__global__ void test_kernel(float *data, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		data[idx] *= 2.0f;
	}
}

/**
 * @brief CUDA kernel for initializing memory with a specific pattern
 * 
 * Pattern-based initialization kernel that creates predictable data
 * for testing memory transfers and detecting corruption issues.
 * 
 * @param data Device pointer to floating-point array
 * @param size Number of elements in the array
 * @param pattern Base pattern value to initialize with
 */
__global__ void memory_pattern_kernel(float *data, int size, float pattern)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		data[idx] = pattern + static_cast<float>(idx);
	}
}

/**
 * @brief Device function demonstrating device-only code compilation
 * 
 * Simple mathematical transformation that can only be called from
 * device code. Tests the CUDA compiler's device function handling.
 * 
 * @param x Input value for computation
 * @return Computed result (x^2 + 1)
 */
__device__ float device_only_function(float x)
{
	return x * x + 1.0f;
}

/**
 * @brief Advanced kernel demonstrating device function usage
 * 
 * More complex kernel that calls device functions and performs
 * conditional operations. Tests sophisticated CUDA compilation
 * features including device function calls and control flow.
 * 
 * @param data Device pointer to data array
 * @param size Number of elements to process
 */
__global__ void advanced_test_kernel(float *data, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		float value = data[idx];

		// Use device function
		value = device_only_function(value);

		// Conditional computation
		if (idx % 2 == 0) {
			value *= 0.5f;
		} else {
			value += 10.0f;
		}

		data[idx] = value;
	}
}

/**
 * @brief Host wrapper for launching the test doubling kernel
 * 
 * Encapsulates kernel launch logic with proper error checking and
 * grid/block dimension calculation. Provides clean interface for
 * host-side test code.
 * 
 * @param d_data Device pointer to data array
 * @param size Number of elements to process
 * @throws std::runtime_error if kernel launch fails
 */
extern "C" void cuda_linking_tests::launch_test_kernel(float *d_data, int size)
{
	if (d_data == nullptr || size <= 0) {
		printf("Invalid parameters to launch_test_kernel: ptr=%p, size=%d\n",
		       d_data, size);
		return;
	}

	int block_size = DEFAULT_BLOCK_SIZE;
	int grid_size = (size + block_size - 1) / block_size;

	if (grid_size > 65535) {
		throw std::runtime_error(
			"Array size too large for single kernel launch");
	}

	test_kernel<<<grid_size, block_size>>>(d_data, size);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Test kernel launch error: %s\n",
		       cudaGetErrorString(err));
		throw std::runtime_error(
			std::string("Test kernel launch failed: ") +
			cudaGetErrorString(err));
	}
}

/**
 * @brief Host wrapper for launching the memory pattern kernel
 * 
 * Launches pattern initialization kernel with comprehensive error
 * checking and optimal launch configuration.
 * 
 * @param d_data Device pointer to data array
 * @param size Number of elements to initialize
 * @param pattern Base pattern value for initialization
 * @throws std::runtime_error if kernel launch fails
 */
extern "C" void cuda_linking_tests::launch_memory_pattern_kernel(float *d_data,
								 int size,
								 float pattern)
{
	if (d_data == nullptr || size <= 0) {
		printf("Invalid parameters to launch_memory_pattern_kernel: ptr=%p, size=%d\n",
		       d_data, size);
		return;
	}

	int block_size = DEFAULT_BLOCK_SIZE;
	int grid_size = (size + block_size - 1) / block_size;

	if (grid_size > 65535) {
		throw std::runtime_error(
			"Array size too large for pattern kernel launch");
	}

	memory_pattern_kernel<<<grid_size, block_size>>>(d_data, size, pattern);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Pattern kernel launch error: %s\n",
		       cudaGetErrorString(err));
		throw std::runtime_error(
			std::string("Pattern kernel launch failed: ") +
			cudaGetErrorString(err));
	}
}

/**
 * @brief Host wrapper for launching the advanced test kernel
 * 
 * Launches complex kernel that tests device function calls and
 * sophisticated control flow patterns.
 * 
 * @param d_data Device pointer to data array
 * @param size Number of elements to process
 * @throws std::runtime_error if kernel launch fails
 */
extern "C" void cuda_linking_tests::launch_advanced_test_kernel(float *d_data,
								int size)
{
	if (d_data == nullptr || size <= 0) {
		printf("Invalid parameters to launch_advanced_test_kernel: ptr=%p, size=%d\n",
		       d_data, size);
		return;
	}

	int block_size = DEFAULT_BLOCK_SIZE;
	int grid_size = (size + block_size - 1) / block_size;

	if (grid_size > 65535) {
		throw std::runtime_error(
			"Array size too large for advanced kernel launch");
	}

	advanced_test_kernel<<<grid_size, block_size>>>(d_data, size);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Advanced kernel launch error: %s\n",
		       cudaGetErrorString(err));
		throw std::runtime_error(
			std::string("Advanced kernel launch failed: ") +
			cudaGetErrorString(err));
	}
}
