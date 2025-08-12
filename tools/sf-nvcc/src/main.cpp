/**
 * @file main.cpp
 * @brief Entry point for sf-nvcc SafeCUDA compiler wrapper
 *
 * Main executable that acts as a drop-in replacement for nvcc,
 * providing automatic PTX modification for bounds checking injection.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-08-12
 * @version 0.0.1
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-08-12: Initial File
 */

#include "nvcc_wrapper.h"

#include <iostream>

int main(int argc, char *argv[])
{
	std::cout << "Test" << std::endl;
}
