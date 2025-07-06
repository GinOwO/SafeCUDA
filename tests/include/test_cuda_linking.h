/**
 * @file test_cuda_linking.h
 * @brief CUDA linking test declarations and common includes
 * 
 * Contains function declarations, common includes, and shared definitions
 * for CUDA linking tests that verify basic GPU functionality.
 * 
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-07-06
 * @version 0.0.1
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 * 
 * Change Log:
 * - 2025-07-06: Initial implementation with Google Test
 */

#ifndef TEST_CUDA_LINKING_H
#define TEST_CUDA_LINKING_H

#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace cuda_linking_tests
{
constexpr int DEFAULT_BLOCK_SIZE = 256;
constexpr int TEST_ARRAY_SIZE = 1024;
constexpr float FLOATING_POINT_TOLERANCE = 1e-6f;

class CudaLinkingTest : public ::testing::Test {
    protected:
	void SetUp() override;
	void TearDown() override;

	void print_device_info(int device_id);
	bool check_cuda_error(cudaError_t error, const std::string &operation);
};

} // namespace cuda_linking_tests

#endif // TEST_CUDA_LINKING_H