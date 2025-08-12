/**
 * @file test_cuda_linking.cpp
 * @brief Host-side implementation of CUDA linking tests
 * 
 * Contains host-side test logic, CUDA API calls, and Google Test
 * integration for verifying CUDA runtime functionality.
 * 
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-07-06
 * @version 0.0.1
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 * 
 * Change Log:
 * - 2025-07-06: Initial implementation with Google Test
 */

#include "test_cuda_linking.h"
#include "test_cuda_linking.cuh"

#include <cmath>
#include <iostream>
#include <memory>

using namespace cuda_linking_tests;

/**
 * @brief Sets up clean CUDA environment for each test
 * 
 * Ensures each test starts with a fresh CUDA context by resetting
 * the device. This prevents state leakage between tests and ensures
 * consistent test behavior.
 * 
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @version 0.0.1
 */
void CudaLinkingTest::SetUp()
{
	cudaError_t err = cudaDeviceReset();
	if (err != cudaSuccess && err != cudaErrorInvalidDevice) {
		std::cerr << "Warning: cudaDeviceReset failed: "
			  << cudaGetErrorString(err) << std::endl;
	}
}

/**
 * @brief Cleans up CUDA environment after each test
 * 
 * Resets the CUDA device to ensure no state leakage affects
 * subsequent tests. Handles potential errors gracefully.
 * 
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @version 0.0.1
 */
void CudaLinkingTest::TearDown()
{
	cudaError_t err = cudaDeviceReset();
	if (err != cudaSuccess && err != cudaErrorInvalidDevice) {
		std::cerr << "Warning: cudaDeviceReset in teardown failed: "
			  << cudaGetErrorString(err) << std::endl;
	}
}

/**
 * @brief Prints detailed device information for debugging
 * 
 * Outputs comprehensive device properties including compute capability,
 * memory size, and architectural features. This helps diagnose
 * device-specific issues during test failures.
 * 
 * @param device_id CUDA device identifier
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @version 0.0.1
 */
void CudaLinkingTest::print_device_info(int device_id)
{
	cudaDeviceProp prop{};
	cudaError_t err = cudaGetDeviceProperties(&prop, device_id);

	if (err == cudaSuccess) {
		std::cout << "  Device " << device_id << ": " << prop.name
			  << " (Compute " << prop.major << "." << prop.minor
			  << ")"
			  << " - " << prop.totalGlobalMem / (1024 * 1024)
			  << " MB" << std::endl;

		if (prop.major < 3) {
			std::cout << "    [WARNING: Compute capability < 3.0]"
				  << std::endl;
		}
	}
}

/**
 * @brief Helper function for CUDA error checking with descriptive messages
 * 
 * Provides centralized error checking that integrates well with Google Test's
 * assertion system. Returns boolean for use with EXPECT_TRUE/ASSERT_TRUE.
 * 
 * @param error CUDA error code to check
 * @param operation Description of the operation that was attempted
 * @return true if no error, false otherwise
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @version 0.0.1
 */
bool CudaLinkingTest::check_cuda_error(cudaError_t error,
				       const std::string &operation)
{
	if (error != cudaSuccess) {
		std::cerr << operation
			  << " failed: " << cudaGetErrorString(error)
			  << std::endl;
		return false;
	}
	return true;
}

/**
 * @brief Tests basic CUDA device enumeration and properties
 * 
 * Verifies that CUDA runtime can detect and enumerate available
 * GPU devices. This test catches driver issues, CUDA installation
 * problems, and basic hardware compatibility issues.
 * 
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @version 0.0.1
 */
TEST_F(CudaLinkingTest, DeviceDetection)
{
	int device_count;
	cudaError_t err = cudaGetDeviceCount(&device_count);

	ASSERT_TRUE(check_cuda_error(err, "cudaGetDeviceCount"));
	ASSERT_GT(device_count, 0)
		<< "No CUDA devices found. Check driver installation.";

	std::cout << "Found " << device_count
		  << " CUDA device(s):" << std::endl;

	for (int i = 0; i < device_count; ++i) {
		print_device_info(i);
	}
}

/**
 * @brief Tests CUDA memory allocation and deallocation operations
 * 
 * Verifies that basic GPU memory management operations work correctly.
 * Tests allocation of various sizes and proper cleanup to catch memory
 * management issues early in development.
 * 
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @version 0.0.1
 */
TEST_F(CudaLinkingTest, MemoryAllocation)
{
	std::vector<size_t> test_sizes = {
		sizeof(float),
		1024 * sizeof(float),
		1024 * 1024 * sizeof(float),
	};

	for (size_t test_size : test_sizes) {
		void *d_ptr = nullptr;

		cudaError_t err = cudaMalloc(&d_ptr, test_size);
		ASSERT_TRUE(check_cuda_error(
			err,
			"cudaMalloc for size " + std::to_string(test_size)));
		ASSERT_NE(d_ptr, nullptr)
			<< "cudaMalloc returned null for size " << test_size;

		err = cudaFree(d_ptr);
		EXPECT_TRUE(check_cuda_error(
			err, "cudaFree for size " + std::to_string(test_size)));
	}
}

/**
 * @brief Tests complete CUDA memory operation pipeline
 * 
 * Performs end-to-end test of GPU memory operations including allocation,
 * host-to-device transfer, device-to-host transfer, and result validation.
 * This test verifies the entire CUDA memory subsystem.
 * 
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @version 0.0.1
 */
TEST_F(CudaLinkingTest, MemoryOperations)
{
	constexpr int size = TEST_ARRAY_SIZE;
	constexpr size_t bytes = size * sizeof(float);

	std::vector<float> h_data(size);
	for (int i = 0; i < size; ++i) {
		h_data[i] = static_cast<float>(i) * 0.5f + 1.0f;
	}

	float *d_data = nullptr;
	cudaError_t err = cudaMalloc(&d_data, bytes);
	ASSERT_TRUE(check_cuda_error(err, "cudaMalloc"));

	err = cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice);
	ASSERT_TRUE(check_cuda_error(err, "cudaMemcpy H2D"));

	err = cudaDeviceSynchronize();
	ASSERT_TRUE(check_cuda_error(err, "cudaDeviceSynchronize after H2D"));

	std::vector<float> h_result(size);
	err = cudaMemcpy(h_result.data(), d_data, bytes,
			 cudaMemcpyDeviceToHost);
	ASSERT_TRUE(check_cuda_error(err, "cudaMemcpy D2H"));

	for (int i = 0; i < size; ++i) {
		EXPECT_NEAR(h_result[i], h_data[i], FLOATING_POINT_TOLERANCE)
			<< "Data corruption at index " << i;
	}

	cudaFree(d_data);
}

/**
 * @brief Tests CUDA kernel execution and host-device interaction
 * 
 * Verifies that CUDA kernels can be launched successfully and produce
 * correct results. Tests multiple kernel types to verify the complete
 * CUDA compilation and execution pipeline.
 * 
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @version 0.0.1
 */
TEST_F(CudaLinkingTest, KernelExecution)
{
	constexpr int size = TEST_ARRAY_SIZE;
	constexpr size_t bytes = size * sizeof(float);

	std::vector<float> h_data(size);
	for (int i = 0; i < size; ++i) {
		h_data[i] = static_cast<float>(i);
	}

	float *d_data = nullptr;
	cudaError_t err = cudaMalloc(&d_data, bytes);
	ASSERT_TRUE(check_cuda_error(err, "cudaMalloc"));

	err = cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice);
	ASSERT_TRUE(check_cuda_error(err, "cudaMemcpy H2D"));

	ASSERT_NO_THROW(launch_test_kernel(d_data, size));

	err = cudaDeviceSynchronize();
	ASSERT_TRUE(check_cuda_error(
		err, "cudaDeviceSynchronize after test kernel"));

	std::vector<float> h_result(size);
	err = cudaMemcpy(h_result.data(), d_data, bytes,
			 cudaMemcpyDeviceToHost);
	ASSERT_TRUE(check_cuda_error(err, "cudaMemcpy D2H"));

	for (int i = 0; i < size; ++i) {
		float expected = 2.0f * static_cast<float>(i);
		EXPECT_NEAR(h_result[i], expected, FLOATING_POINT_TOLERANCE)
			<< "Kernel result mismatch at index " << i;
	}

	constexpr float test_pattern = 42.0f;
	ASSERT_NO_THROW(
		launch_memory_pattern_kernel(d_data, size, test_pattern));

	err = cudaDeviceSynchronize();
	ASSERT_TRUE(check_cuda_error(
		err, "cudaDeviceSynchronize after pattern kernel"));

	err = cudaMemcpy(h_result.data(), d_data, bytes,
			 cudaMemcpyDeviceToHost);
	ASSERT_TRUE(check_cuda_error(err, "cudaMemcpy D2H after pattern"));

	for (int i = 0; i < size; ++i) {
		float expected = test_pattern + static_cast<float>(i);
		EXPECT_NEAR(h_result[i], expected, FLOATING_POINT_TOLERANCE)
			<< "Pattern mismatch at index " << i;
	}

	cudaFree(d_data);
}

/**
 * @brief Tests advanced kernel features and device function calls
 * 
 * Verifies that complex kernels with device function calls and
 * multiple operations work correctly. This tests the CUDA compiler's
 * ability to handle sophisticated device code.
 * 
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @version 0.0.1
 */
TEST_F(CudaLinkingTest, AdvancedKernelFeatures)
{
	constexpr int size = TEST_ARRAY_SIZE;
	constexpr size_t bytes = size * sizeof(float);

	std::vector<float> h_data(size);
	for (int i = 0; i < size; ++i) {
		h_data[i] = static_cast<float>(i) + 1.0f;
	}

	float *d_data = nullptr;
	cudaError_t err = cudaMalloc(&d_data, bytes);
	ASSERT_TRUE(check_cuda_error(err, "cudaMalloc"));

	err = cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice);
	ASSERT_TRUE(check_cuda_error(err, "cudaMemcpy H2D"));

	ASSERT_NO_THROW(launch_advanced_test_kernel(d_data, size));

	err = cudaDeviceSynchronize();
	ASSERT_TRUE(check_cuda_error(err, "cudaDeviceSynchronize"));

	std::vector<float> h_result(size);
	err = cudaMemcpy(h_result.data(), d_data, bytes,
			 cudaMemcpyDeviceToHost);
	ASSERT_TRUE(check_cuda_error(err, "cudaMemcpy D2H"));

	for (int i = 0; i < size; ++i) {
		float input = static_cast<float>(i) + 1.0f;
		float expected = input * input + 1.0f;

		if (i % 2 == 0) {
			expected *= 0.5f;
		} else {
			expected += 10.0f;
		}

		EXPECT_NEAR(h_result[i], expected, FLOATING_POINT_TOLERANCE)
			<< "Advanced kernel mismatch at index " << i;
	}

	cudaFree(d_data);
}
