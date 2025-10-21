/**
* @file test_sf_nvcc_ptx_injector.h
 * @brief Test sf-nvcc ptx injection
 *
 * Contains test fixture class and helper methods for testing
 * sf-nvcc ptx injection
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-09-22
 * @version 0.0.1
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-09-22: Initial implementation with Google Test
 */

#ifndef TEST_SF_NVCC_PTX_INJECTOR_H
#define TEST_SF_NVCC_PTX_INJECTOR_H

#include <gtest/gtest.h>

#include <filesystem>

namespace sf_nvcc_ptx_injection_tests
{
class SfNvccPtxInjectionTest : public ::testing::Test {
    public:
	std::filesystem::path temp_file;

    protected:
	void SetUp() override;
	void TearDown() override;
	void write_ptx(std::string_view content) const;
};
}

#endif //TEST_SF_NVCC_PTX_INJECTOR_H
