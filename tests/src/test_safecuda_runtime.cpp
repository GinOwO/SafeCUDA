/**
 * @file test_safecuda_runtime.cpp
 * @brief Host-side implementation of SafeCUDA runtime tests
 *
 * Contains host-side test logic, metadata validation, table verification,
 * and error detection tests for SafeCUDA's runtime bounds checking system.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-10-23
 * @version 1.0.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-10-23: Initial implementation with Google Test
 */

#include "test_safecuda_runtime.h"
#include "test_safecuda_runtime.cuh"

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>

using namespace safecuda_runtime_tests;

void SafeCudaRuntimeTest::SetUp()
{
	if (const cudaError_t err = cudaDeviceReset();
	    err != cudaSuccess && err != cudaErrorInvalidDevice) {
		std::cerr << "Warning: cudaDeviceReset failed: "
			  << cudaGetErrorString(err) << std::endl;
	}
}

void SafeCudaRuntimeTest::TearDown()
{
	cudaError_t err = cudaDeviceReset();
	if (err != cudaSuccess && err != cudaErrorInvalidDevice) {
		std::cerr << "Warning: cudaDeviceReset in teardown failed: "
			  << cudaGetErrorString(err) << std::endl;
	}
}

bool SafeCudaRuntimeTest::check_cuda_error(cudaError_t error,
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
 * @brief Tests SafeCUDA metadata injection during allocation
 *
 * Verifies that cudaMalloc correctly injects the 16-byte metadata header
 * containing the magic word (0x5AFE) before the user-accessible memory region.
 * This test ensures the fundamental metadata structure is properly initialized.
 */
TEST_F(SafeCudaRuntimeTest, MetadataInjection)
{
	constexpr size_t test_size = 1024 * sizeof(float);
	float *d_ptr = nullptr;

	cudaError_t err =
		cudaMalloc(reinterpret_cast<void **>(&d_ptr), test_size);
	ASSERT_TRUE(check_cuda_error(err, "cudaMalloc"));
	ASSERT_NE(d_ptr, nullptr) << "cudaMalloc returned null pointer";

	uint16_t magic = 0;
	char *metadata_base = reinterpret_cast<char *>(d_ptr) - 16;
	err = cudaMemcpy(&magic, metadata_base, sizeof(uint16_t),
			 cudaMemcpyDeviceToHost);
	ASSERT_TRUE(check_cuda_error(err, "cudaMemcpy metadata"));

	EXPECT_EQ(magic, EXPECTED_MAGIC)
		<< "Magic word mismatch: expected 0x5AFE, got 0x" << std::hex
		<< magic;

	err = cudaFree(d_ptr);
	EXPECT_TRUE(check_cuda_error(err, "cudaFree"));
}

/**
 * @brief Tests allocation table population with multiple allocations
 *
 * Performs multiple sequential memory allocations to verify that the
 * SafeCUDA allocation table correctly tracks all allocated memory regions.
 * This test ensures table management works correctly under typical usage.
 */
TEST_F(SafeCudaRuntimeTest, AllocationTablePopulation)
{
	constexpr int num_allocs = 10;
	std::vector<float *> pointers(num_allocs);

	for (int i = 0; i < num_allocs; ++i) {
		size_t size = (i + 1) * 256 * sizeof(float);
		cudaError_t err = cudaMalloc(
			reinterpret_cast<void **>(&pointers[i]), size);
		ASSERT_TRUE(check_cuda_error(err, "cudaMalloc allocation " +
							  std::to_string(i)));
		ASSERT_NE(pointers[i], nullptr)
			<< "Allocation " << i << " returned null";
	}

	for (int i = 0; i < num_allocs; ++i) {
		cudaError_t err = cudaFree(pointers[i]);
		EXPECT_TRUE(check_cuda_error(err, "cudaFree allocation " +
							  std::to_string(i)));
	}
}

/**
 * @brief Tests bounds checking with valid memory access patterns
 *
 * Launches a kernel that performs valid in-bounds memory accesses and
 * verifies that SafeCUDA's bounds checking does not interfere with
 * legitimate memory operations. This is a baseline test for correct
 * operation without false positives.
 */
// TEST_F(SafeCudaRuntimeTest, ValidMemoryAccess)
// {
// 	constexpr int size = TEST_ARRAY_SIZE;
// 	constexpr size_t bytes = size * sizeof(float);
//
// 	float *d_data = nullptr;
// 	cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_data), bytes);
// 	ASSERT_TRUE(check_cuda_error(err, "cudaMalloc"));
//
// 	ASSERT_NO_THROW(launch_valid_access_kernel(d_data, size));
//
// 	err = cudaDeviceSynchronize();
// 	ASSERT_TRUE(check_cuda_error(err, "cudaDeviceSynchronize"));
//
// 	std::vector<float> h_result(size);
// 	err = cudaMemcpy(h_result.data(), d_data, bytes,
// 			 cudaMemcpyDeviceToHost);
// 	ASSERT_TRUE(check_cuda_error(err, "cudaMemcpy D2H"));
//
// 	for (int i = 0; i < size; ++i) {
// 		float expected = static_cast<float>(i) * 2.0f;
// 		EXPECT_NEAR(h_result[i], expected, FLOATING_POINT_TOLERANCE)
// 			<< "Mismatch at index " << i;
// 	}
//
// 	cudaFree(d_data);
// }

/**
 * @brief Tests interior pointer bounds checking functionality
 *
 * Verifies that SafeCUDA correctly handles interior pointers (pointers
 * offset from the base allocation address) by performing the slow-path
 * linear scan through the allocation table. This tests the fallback
 * mechanism when the fast-path magic word check fails.
 */
// TEST_F(SafeCudaRuntimeTest, InteriorPointerAccess)
// {
// 	constexpr int size = TEST_ARRAY_SIZE;
// 	constexpr size_t bytes = size * sizeof(float);
// 	constexpr int offset = 64;
//
// 	float *d_data = nullptr;
// 	cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_data), bytes);
// 	ASSERT_TRUE(check_cuda_error(err, "cudaMalloc"));
//
// 	ASSERT_NO_THROW(launch_interior_pointer_kernel(d_data, size, offset));
//
// 	err = cudaDeviceSynchronize();
// 	ASSERT_TRUE(check_cuda_error(err, "cudaDeviceSynchronize"));
//
// 	std::vector<float> h_result(size);
// 	err = cudaMemcpy(h_result.data(), d_data, bytes,
// 			 cudaMemcpyDeviceToHost);
// 	ASSERT_TRUE(check_cuda_error(err, "cudaMemcpy D2H"));
//
// 	for (int i = 0; i < size - offset; ++i) {
// 		auto expected = static_cast<float>(i);
// 		EXPECT_NEAR(h_result[i + offset], expected,
// 			    FLOATING_POINT_TOLERANCE)
// 			<< "Interior pointer access failed at index "
// 			<< i + offset;
// 	}
//
// 	cudaFree(d_data);
// }

/**
 * @brief Tests metadata injection across various allocation sizes
 *
 * Performs allocations of different sizes to verify that metadata
 * injection and magic word verification work correctly regardless of
 * allocation size. This ensures the system handles edge cases in
 * memory alignment and size variations.
 */
TEST_F(SafeCudaRuntimeTest, MixedAllocationSizes)
{
	std::vector<size_t> test_sizes = {16 * sizeof(float),
					  256 * sizeof(float),
					  1024 * sizeof(float),
					  4096 * sizeof(float)};

	for (size_t test_size : test_sizes) {
		float *d_ptr = nullptr;
		cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_ptr),
					     test_size);
		ASSERT_TRUE(check_cuda_error(
			err,
			"cudaMalloc for size " + std::to_string(test_size)));
		ASSERT_NE(d_ptr, nullptr);

		uint16_t magic = 0;
		char *metadata_base = reinterpret_cast<char *>(d_ptr) - 16;
		err = cudaMemcpy(&magic, metadata_base, sizeof(uint16_t),
				 cudaMemcpyDeviceToHost);
		ASSERT_TRUE(check_cuda_error(err, "cudaMemcpy metadata check"));
		EXPECT_EQ(magic, EXPECTED_MAGIC);

		cudaFree(d_ptr);
	}
}

/**
 * @brief Tests cudaMallocManaged integration with SafeCUDA
 *
 * Verifies that unified memory allocations (cudaMallocManaged) are
 * correctly instrumented with SafeCUDA metadata and bounds checking.
 * This ensures the system works with both regular and managed memory.
 */
// TEST_F(SafeCudaRuntimeTest, cudaMallocManagedIntegration)
// {
// 	constexpr size_t test_size = 1024 * sizeof(float);
// 	float *d_ptr = nullptr;
//
// 	cudaError_t err =
// 		cudaMallocManaged(reinterpret_cast<void **>(&d_ptr), test_size);
// 	ASSERT_TRUE(check_cuda_error(err, "cudaMallocManaged"));
// 	ASSERT_NE(d_ptr, nullptr);
//
// 	uint16_t magic = 0;
// 	char *metadata_base = reinterpret_cast<char *>(d_ptr) - 16;
// 	std::memcpy(&magic, metadata_base, sizeof(uint16_t));
//
// 	EXPECT_EQ(magic, EXPECTED_MAGIC);
//
// 	err = cudaFree(d_ptr);
// 	EXPECT_TRUE(check_cuda_error(err, "cudaFree"));
// }

/**
 * @brief Tests proper handling of null pointer in cudaFree
 *
 * Verifies that cudaFree(nullptr) is handled gracefully without
 * crashing or producing errors. This is standard CUDA behavior
 * that SafeCUDA must preserve.
 */
TEST_F(SafeCudaRuntimeTest, NullPointerHandling)
{
	cudaError_t err = cudaFree(nullptr);
	EXPECT_TRUE(check_cuda_error(err, "cudaFree(nullptr)"));
}

/**
 * @brief Tests allocation table capacity limits
 *
 * Performs numerous allocations to verify the allocation table's
 * behavior under stress and to identify the maximum number of
 * concurrent allocations supported. This helps validate the
 * fixed-size table design (1024 entries).
 */
// TEST_F(SafeCudaRuntimeTest, AllocationTableCapacity)
// {
// 	constexpr int max_test_allocs = 100;
// 	std::vector<float *> pointers;
//
// 	for (int i = 0; i < max_test_allocs; ++i) {
// 		float *d_ptr = nullptr;
// 		cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_ptr),
// 					     256 * sizeof(float));
// 		if (err != cudaSuccess) {
// 			std::cout << "Allocation limit reached at " << i
// 				  << " allocations" << std::endl;
// 			break;
// 		}
// 		pointers.push_back(d_ptr);
// 	}
//
// 	EXPECT_GT(pointers.size(), 0) << "No allocations succeeded";
//
// 	for (auto ptr : pointers) {
// 		cudaFree(ptr);
// 	}
// }

/**
 * @brief Tests direct invocation of bounds check with valid pointer
 *
 * Directly calls __bounds_check_safecuda from device code with a valid
 * pointer to verify that the bounds checking function correctly validates
 * in-bounds memory access without throwing exceptions when exceptions
 * are not enabled.
 */
// TEST_F(SafeCudaRuntimeTest, DirectBoundsCheckValidNoException)
// {
// 	constexpr int size = TEST_ARRAY_SIZE;
// 	constexpr size_t bytes = size * sizeof(float);
// 	float *d_data = nullptr;
// 	cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_data), bytes);
// 	ASSERT_TRUE(check_cuda_error(err, "cudaMalloc"));
// 	ASSERT_NO_THROW(launch_manual_bounds_check(d_data, size, 0));
// 	err = cudaDeviceSynchronize();
// 	EXPECT_TRUE(check_cuda_error(err, "cudaDeviceSynchronize"));
// 	cudaFree(d_data);
// }

/**
 * @brief Tests bounds check with invalid pointer and exception throwing
 *
 * Enables exception throwing via environment variable and verifies that
 * out-of-bounds access triggers a runtime_error exception with appropriate
 * error message. This tests the integration of bounds checking with the
 * exception-based error reporting mechanism.
 */
// TEST_F(SafeCudaRuntimeTest, DirectBoundsCheckInvalidWithException)
// {
// 	setenv("SAFECUDA_THROW_OOB", "1", 1);
// 	constexpr int size = TEST_ARRAY_SIZE;
// 	constexpr size_t bytes = size * sizeof(float);
// 	float *d_data = nullptr;
// 	cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_data), bytes);
// 	ASSERT_TRUE(check_cuda_error(err, "cudaMalloc"));
// 	ASSERT_NO_THROW(launch_manual_bounds_check(d_data, size, 1));
// 	EXPECT_THROW(cudaDeviceSynchronize(), std::runtime_error);
// 	cudaFree(d_data);
// 	unsetenv("SAFECUDA_THROW_OOB");
// }

/**
 * @brief Tests error reporting without exception throwing
 *
 * Verifies that when exception throwing is disabled, bounds violations
 * are still detected and logged but do not terminate the program. This
 * tests the default error reporting behavior via stderr logging.
 */
// TEST_F(SafeCudaRuntimeTest, ErrorReportingWithoutException)
// {
// 	constexpr int size = TEST_ARRAY_SIZE;
// 	constexpr size_t bytes = size * sizeof(float);
// 	float *d_data = nullptr;
// 	cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_data), bytes);
// 	ASSERT_TRUE(check_cuda_error(err, "cudaMalloc"));
// 	ASSERT_NO_THROW(launch_manual_bounds_check(d_data, size, 1));
// 	EXPECT_NO_THROW(cudaDeviceSynchronize());
// 	ASSERT_NO_THROW(launch_manual_bounds_check(d_data, size, 0));
// 	EXPECT_NO_THROW(cudaDeviceSynchronize());
// 	cudaFree(d_data);
// }

/**
 * @brief Tests error reporting for Use After Free
 *
 * Verifies that use-after-free errors are properly detected and reported. Tests
 * the alternative error checking path with exception throwing enabled.
 */
// TEST_F(SafeCudaRuntimeTest, UseAfterFree)
// {
// 	setenv("SAFECUDA_THROW_FREED_MEMORY", "1", 1);
// 	constexpr int size = TEST_ARRAY_SIZE;
// 	constexpr size_t bytes = size * sizeof(float);
// 	float *d_data = nullptr;
// 	cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_data), bytes);
// 	ASSERT_TRUE(check_cuda_error(err, "cudaMalloc"));
// 	ASSERT_NO_THROW(launch_manual_bounds_check(d_data, size, 0));
// 	EXPECT_NO_THROW(cudaDeviceSynchronize());
// 	err = cudaFree(d_data);
// 	ASSERT_TRUE(check_cuda_error(err, "cudaFree"));
// 	EXPECT_NO_THROW(cudaDeviceSynchronize());
// 	ASSERT_NO_THROW(launch_manual_bounds_check(d_data, size, 0));
// 	EXPECT_THROW(cudaDeviceSynchronize(), std::runtime_error);
// 	unsetenv("SAFECUDA_THROW_FREED_MEMORY");
// }
