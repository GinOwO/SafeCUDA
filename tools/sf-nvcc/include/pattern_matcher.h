/**
 * @file pattern_matcher.h
 * @brief PTX instruction pattern recognition system
 *
 * Identifies PTX load/store instructions that require bounds checking
 * instrumentation instruction analysis.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-08-12
 * @version 1.0.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-09-22: Initial Implementation
 * - 2025-08-12: Initial File
 */

#ifndef SAFECUDA_PATTERN_MATCHER_H
#define SAFECUDA_PATTERN_MATCHER_H

#include <string>
#include <vector>

namespace safecuda::tools::sf_nvcc
{
/**
 * @brief Information about a single instruction line in PTX
 *
 * Stores the line number(int64_t) and the list of lexemes in that
 * line (vector<string>)
 */
struct Instruction {
	int64_t line_number;
	std::vector<std::string> lexemes;

	Instruction()
		: line_number(-1) {};
};

/**
 * @brief Function to find global instructions in a file
 *
 * Reads the input file and returns a vector of struct Instructions with
 * all required Instructions for PTX modification.
 *
 * @param path String containing the path to the PTX file
 * @return Vector<Instructions> containing all lexemes of relevant lines.
 *
 * @throws std::runtime_error if it cannot open file at path
 */
std::vector<struct Instruction> find_all_ptx(std::string_view path);

} // namespace safecuda::tools::sf_nvcc

#endif // SAFECUDA_PATTERN_MATCHER_H
