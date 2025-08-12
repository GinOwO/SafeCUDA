/**
 * @file sf_nvcc_parsing_tests.h
 * @brief Test sf-nvcc argument parsing
 *
 * Contains test fixture class and helper methods for testing
 * sf-nvcc argument parsing
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-08-13
 * @version 0.0.1
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-08-13: Initial implementation with Google Test
 */

#ifndef SF_NVCC_PARSING_TESTS_H
#define SF_NVCC_PARSING_TESTS_H

#include <gtest/gtest.h>

namespace sf_nvcc_parsing_tests
{
class SfNvccParsingTest : public ::testing::Test {
    protected:
	void SetUp() override {};
	void TearDown() override {};
};

}

#endif //SF_NVCC_PARSING_TESTS_H
