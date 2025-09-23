/**
 * @file test_sf_nvcc_parsing.h
 * @brief Test sf-nvcc argument parsing
 *
 * Contains test fixture class and helper methods for testing
 * sf-nvcc argument parsing
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-08-13
 * @version 0.0.2
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-09-22: Renamed File to be style compliant
 * - 2025-08-13: Initial implementation with Google Test
 */

#ifndef TEST_SF_NVCC_PARSING_H
#define TEST_SF_NVCC_PARSING_H

#include <gtest/gtest.h>

namespace sf_nvcc_parsing_tests
{
class SfNvccParsingTest : public ::testing::Test {
    protected:
	void SetUp() override {};
	void TearDown() override {};
};

}

#endif //TEST_SF_NVCC_PARSING_H
