/**
 * @file bootstrap_test.cpp
 * @brief Test to check if GTest is working properly
 * 
 * If this test fails then GTest was not installed properly
 * 
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in> 
 * @version 0.0.1
 * @date 2025-07-06
 * @copyright Copyright (c) 2025
 * 
 * Change Log:
 * - 2025-07-05: Initial implementation
 */

#include <gtest/gtest.h>

TEST(GTEST_TEST, BootStrap)
{
	EXPECT_EQ(1 + 1, 2);
}
