/**
 * @file pattern_matcher.cpp
 * @brief Implementation of PTX pattern recognition system
 *
 * Contains logic for identifying PTX load/store instructions that
 * require SafeCUDA bounds checking instrumentation.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-08-12
 * @version 0.0.2
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-09-22: Initial Implementation
 * - 2025-08-12: Initial File
 */

#include "pattern_matcher.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace sf_nvcc = safecuda::tools::sf_nvcc;

// Static helper functions for PTX analysis

/**
 * @brief Tokenize a single line of PTX code into lexemes
 * @param line The PTX instruction line to tokenize
 * @return Vector of lexemes (tokens) from the line
 */
static std::vector<std::string>
tokenizeLine(const std::string &line) // NOLINT(*-no-recursion)
{
	std::vector<std::string> tokens;
	std::string current_token;

	for (size_t i = 0; i < line.length(); ++i) {
		char c = line[i];

		if (std::isspace(c)) {
			if (!current_token.empty()) {
				tokens.push_back(current_token);
				current_token.clear();
			}
			continue;
		}

		if (c == '[') {
			if (!current_token.empty()) {
				tokens.push_back(current_token);
				current_token.clear();
			}

			tokens.emplace_back("[");

			size_t bracket_start = i + 1;
			size_t bracket_end = bracket_start;
			int bracket_depth = 1;

			for (size_t j = bracket_start;
			     j < line.length() && bracket_depth > 0; ++j) {
				if (line[j] == '[')
					bracket_depth++;
				else if (line[j] == ']')
					bracket_depth--;
				bracket_end = j;
			}

			if (bracket_depth == 0) {
				std::string bracket_content = line.substr(
					bracket_start,
					bracket_end - bracket_start);

				std::vector<std::string> bracket_tokens =
					tokenizeLine(bracket_content);
				for (const auto &token : bracket_tokens) {
					tokens.push_back(token);
				}
				tokens.emplace_back("]");
				i = bracket_end;
			} else {
				current_token += c;
			}
		} else if (c == ',') { // NOLINT(*-branch-clone)
			if (!current_token.empty()) {
				tokens.push_back(current_token);
				current_token.clear();
			}
		} else if (c == ';') {
			if (!current_token.empty()) {
				tokens.push_back(current_token);
				current_token.clear();
			}
		} else if (c == '+' || c == '-' || c == '*' || c == '/') {
			if (!current_token.empty()) {
				tokens.push_back(current_token);
				current_token.clear();
			}
			tokens.emplace_back(1, c);
		} else {
			current_token += c;
		}
	}

	if (!current_token.empty()) {
		tokens.push_back(current_token);
	}

	return tokens;
}

/**
 * @brief Check if an instruction is a global memory access that needs bounds checking
 * @param lexemes Vector of lexemes representing the instruction
 * @return True if instruction accesses global memory and needs instrumentation
 */
static bool isGlobalMemoryInstruction(const std::vector<std::string> &lexemes)
{
	if (lexemes.empty())
		return false;

	const std::string &instruction = lexemes[0];

	return instruction.find("ld.global") ==
		       0 || // ld.global.f32, ld.global.u64, etc.
	       instruction.find("st.global") ==
		       0 || // st.global.f32, st.global.u32, etc.
	       instruction.find("atom.global") == 0 || // atom.global.add, etc.
	       instruction.find("red.global") == 0; // red.global.add, etc.
}

/**
 * @brief Check if a line is a label, comment, or other non-instruction
 * @param line The line to check
 * @return True if line should be preserved as-is without modification
 */
static bool isNonInstructionLine(const std::string &line)
{
	std::string trimmed = line;
	trimmed.erase(0, trimmed.find_first_not_of(" \t"));
	trimmed.erase(trimmed.find_last_not_of(" \t") + 1);

	return trimmed.empty() || trimmed[0] == '/' || trimmed[0] == '{' ||
	       trimmed[0] == '}' ||
	       trimmed[0] == '.' || // Directive (.version, .target, etc.)
	       trimmed.back() == ':'; // Label (e.g., $L__BB0_2:)
}

std::vector<sf_nvcc::Instruction> sf_nvcc::find_all_ptx(std::string_view path)
{
	std::ifstream file{std::string(path)};
	if (!file.is_open()) {
		throw std::runtime_error("Cannot open PTX file: " +
					 std::string(path));
	}

	std::vector<Instruction> global_memory_instructions;
	std::string line;
	int64_t line_number = 0;

	while (std::getline(file, line)) {
		line_number++;

		if (isNonInstructionLine(line)) {
			continue;
		}

		std::vector<std::string> lexemes = tokenizeLine(line);

		if (lexemes.empty()) {
			continue;
		}

		if (isGlobalMemoryInstruction(lexemes)) {
			Instruction instruction;
			instruction.line_number = line_number;
			instruction.lexemes = std::move(lexemes);

			global_memory_instructions.push_back(instruction);
		}
	}

	file.close();

	return global_memory_instructions;
}
