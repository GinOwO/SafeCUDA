/**
 * @file test_sf_nvcc_pattern_matcher.h
 * @brief Test sf-nvcc pattern matching
 *
 * Contains test fixture class and helper methods for testing
 * sf-nvcc pattern matching
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-09-22
 * @version 0.0.1
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-09-22: Initial implementation with Google Test
 */

#ifndef TEST_SF_NVCC_PATTERN_MATCHER_H
#define TEST_SF_NVCC_PATTERN_MATCHER_H

#include <gtest/gtest.h>

#include <filesystem>

namespace sf_nvcc_pattern_matching_tests
{
class SfNvccPatternMatchingTest : public ::testing::Test {
    public:
	std::filesystem::path temp_file;

    protected:
	void SetUp() override;
	void TearDown() override;
	void write_ptx(std::string_view content) const;
};
}

#endif //TEST_SF_NVCC_PATTERN_MATCHER_H
