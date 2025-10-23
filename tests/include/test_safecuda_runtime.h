/**
* @file test_safecuda_runtime.h
 * @brief SafeCUDA runtime bounds checking test declarations
 *
 * Contains function declarations, common includes, and shared definitions
 * for testing SafeCUDA's runtime bounds checking system including metadata,
 * table management, and error detection.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-10-23
 * @version 1.0.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-10-23: Initial implementation with Google Test
 */

#ifndef TEST_SAFECUDA_RUNTIME_H
#define TEST_SAFECUDA_RUNTIME_H

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <string>

namespace safecuda_runtime_tests
{

constexpr int DEFAULT_BLOCK_SIZE = 256;
constexpr int TEST_ARRAY_SIZE = 1024;
constexpr float FLOATING_POINT_TOLERANCE = 1e-6f;
constexpr uint16_t EXPECTED_MAGIC = 0x5AFE;

class SafeCudaRuntimeTest : public ::testing::Test {
    protected:
	void SetUp() override;
	void TearDown() override;
	static bool check_cuda_error(cudaError_t error,
				     const std::string &operation);
};

} // namespace safecuda_runtime_tests

#endif // TEST_SAFECUDA_RUNTIME_H
