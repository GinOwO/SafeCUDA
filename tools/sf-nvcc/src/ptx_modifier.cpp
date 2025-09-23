/**
 * @file ptx_modifier.cpp
 * @brief Implementation of PTX modification engine
 *
 * Parses PTX assembly files, identifies memory operations using pattern
 * matching, and injects SafeCUDA bounds checking macros.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-08-12
 * @version 1.0.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-09-23: Now inserts bounds_check directly into every ptx file
 * - 2025-09-22: Initial Implementation
 * - 2025-08-12: Initial File
 */

#include "ptx_modifier.h"

#include "pattern_matcher.h"

#include <fstream>
#include <iostream>

namespace sf_nvcc = safecuda::tools::sf_nvcc;
namespace fs = std::filesystem;

/**
 * @brief Extract address register from tokenized PTX instruction lexemes
 * @param lexemes Vector of tokenized instruction components
 * @return Register name (e.g., "%rd4") or empty string if not found
 */
static std::string
extractAddressRegister(const std::vector<std::string> &lexemes)
{
	for (size_t i = 0; i < lexemes.size() - 1; ++i) {
		if (lexemes[i] == "[" && i + 1 < lexemes.size()) {
			const std::string &reg = lexemes[i + 1];
			if (!reg.empty() && reg[0] == '%') {
				return reg;
			}
		}
	}
	return "";
}

/**
 * @brief Generate bounds check call instruction with proper indentation
 * @param address_reg Register containing memory address to check
 * @param indentation Whitespace string to match original formatting
 * @return Formatted PTX bounds check instruction
 */
static std::string generateBoundsCheckCall(const std::string &address_reg,
					   const std::string &indentation)
{
	return indentation + "call bounds_check, (" + address_reg + ");";
}

/**
 * @brief Extract indentation whitespace from original PTX line
 * @param line Original PTX instruction line
 * @return Leading whitespace string
 */
static std::string getIndentation(const std::string &line)
{
	size_t first_non_space = line.find_first_not_of(" \t");
	if (first_non_space == std::string::npos) {
		return line;
	}
	return line.substr(0, first_non_space);
}

/**
 * @brief Log informational message with cyan color if verbose/debug enabled
 * @param opts SafeCUDA options containing logging flags
 * @param msg Message to display
 */
static void logInfo(const sf_nvcc::SafeCudaOptions &opts,
		    const std::string &msg)
{
	if (opts.enable_verbose || opts.enable_debug) {
		std::cerr << ACOL(ACOL_C, ACOL_DF) << msg << ACOL_RESET()
			  << std::endl;
	}
}

/**
 * @brief Log success message with green color if verbose/debug enabled
 * @param opts SafeCUDA options containing logging flags
 * @param msg Message to display
 */
static void logSuccess(const sf_nvcc::SafeCudaOptions &opts,
		       const std::string &msg)
{
	if (opts.enable_verbose || opts.enable_debug) {
		std::cerr << ACOL(ACOL_G, ACOL_DF) << msg << ACOL_RESET()
			  << std::endl;
	}
}

/**
 * @brief Log error message with red color
 * @param msg Error message to display
 */
static void logError(const std::string &msg)
{
	std::cerr << ACOL(ACOL_R, ACOL_DF) << "Error: " << msg << ACOL_RESET()
		  << std::endl;
}

/**
 * @brief Create backup copy of PTX file with .bak suffix
 * @param ptx_path Path to original PTX file
 * @param opts SafeCUDA options for logging
 * @return True if backup created successfully, false otherwise
 */
static bool createBackupFile(const fs::path &ptx_path,
			     const sf_nvcc::SafeCudaOptions &opts)
{
	fs::path backup_path = ptx_path;
	backup_path += ".bak";

	try {
		fs::copy_file(ptx_path, backup_path,
			      fs::copy_options::overwrite_existing);
		logInfo(opts,
			"Created backup: " + backup_path.filename().string());
		return true;
	} catch (const fs::filesystem_error &) {
		return false;
	}
}

/**
 * @brief Perform PTX file instrumentation by inserting bounds check calls
 * @param ptx_path Path to PTX file to modify
 * @param instructions Vector of identified global memory instructions
 * @param opts SafeCUDA options for logging and debugging
 * @return True if instrumentation successful, false otherwise
 */
static bool
instrumentPTXFile(const fs::path &ptx_path,
		  const std::vector<sf_nvcc::Instruction> &instructions,
		  const sf_nvcc::SafeCudaOptions &opts,
		  sf_nvcc::PtxModificationResult &result)
{
	std::ifstream input_file(ptx_path);
	if (!input_file.is_open()) {
		return false;
	}

	std::vector<std::string> file_lines;
	std::string line;
	while (std::getline(input_file, line)) {
		file_lines.push_back(line);
	}
	input_file.close();

	std::ofstream output_file(ptx_path);
	if (!output_file.is_open()) {
		return false;
	}

	size_t instruction_index = 0;
	size_t instrumented_count = 0;

	for (size_t line_num = 1; line_num <= file_lines.size(); ++line_num) {
		const std::string &current_line = file_lines[line_num - 1];

		if (current_line.starts_with(".address_size")) {
			output_file << current_line << "\n\n"
				    << R"ptx(.func bounds_check(
	.param .b64 bounds_check_param_0
)
{
	ret;
}
)ptx" << "\n\n";
			continue;
		}

		bool needs_instrumentation = false;
		std::string bounds_check_call;

		if (!(instruction_index < instructions.size() &&
		      instructions[instruction_index].line_number ==
			      static_cast<int64_t>(line_num))) {
			output_file << current_line << "\n";
			continue;
		}

		const auto &instr = instructions[instruction_index];
		std::string address_reg = extractAddressRegister(instr.lexemes);

		if (!address_reg.empty()) {
			std::string indentation = getIndentation(current_line);
			bounds_check_call = generateBoundsCheckCall(
				address_reg, indentation);
			needs_instrumentation = true;
			instrumented_count++;

			if (opts.enable_debug || opts.enable_verbose) {
				std::string instr_preview;
				if (!instr.lexemes.empty()) {
					instr_preview = instr.lexemes[0];
					if (instr.lexemes.size() > 1) {
						instr_preview +=
							" " + instr.lexemes[1];
						if (instr.lexemes.size() > 2) {
							instr_preview += "...";
						}
					}
				}
				logInfo(opts, "Instrumenting line " +
						      std::to_string(line_num) +
						      ": " + instr_preview);
				logInfo(opts,
					"  → Extracted address register: " +
						address_reg);
			}
		}
		instruction_index++;

		if (needs_instrumentation) {
			output_file << bounds_check_call << "\n";
		}
		output_file << current_line << "\n";
	}

	result.instructions_modified = instrumented_count;
	output_file.close();

	return true;
}

sf_nvcc::PtxModificationResult
sf_nvcc::insert_bounds_check(const fs::path &ptx_path,
			     const sf_nvcc::SafeCudaOptions &sf_opts)
{
	PtxModificationResult result;
	const auto start = std::chrono::steady_clock::now();

	if (!fs::exists(ptx_path)) {
		logError("PTX file not found: " + ptx_path.string());
		return result;
	}

	if (!fs::is_regular_file(ptx_path)) {
		logError("Path is not a regular file: " + ptx_path.string());
		return result;
	}

	if (sf_opts.enable_debug && !createBackupFile(ptx_path, sf_opts)) {
		logError("Failed to create backup file");
		return result;
	}

	logInfo(sf_opts, "Starting PTX modification for: " +
				 ptx_path.filename().string());

	std::vector<Instruction> instructions;
	try {
		instructions = find_all_ptx(ptx_path.string());
	} catch (const std::exception &e) {
		logError("Failed to parse PTX file: " + std::string(e.what()));
		return result;
	}

	logInfo(sf_opts, "Found " + std::to_string(instructions.size()) +
				 " global memory instructions");

	if (instructions.empty()) {
		logInfo(sf_opts,
			"No global memory instructions found - no instrumentation needed");
		result.success = true;
		result.modified_ptx_path = ptx_path.string();
		result.modification_time_ms =
			std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::steady_clock::now() - start)
				.count();
		return result;
	}

	if (!instrumentPTXFile(ptx_path, instructions, sf_opts, result)) {
		logError("Failed to instrument PTX file");
		return result;
	}

	logSuccess(sf_opts, "Successfully instrumented " +
				    std::to_string(instructions.size()) +
				    " instructions");
	result.success = true;
	result.modification_time_ms =
		std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - start)
			.count();
	result.modified_ptx_path = ptx_path.string();
	return result;
}

sf_nvcc::PtxModificationResult
sf_nvcc::modify_ptx(const fs::path &ptx_path, const SafeCudaOptions &sf_opts)
{
	PtxModificationResult result;
	result.success = true;
	result.modified_ptx_path = ptx_path.string();

	if (sf_opts.enable_bounds_check) {
		const PtxModificationResult current_res =
			insert_bounds_check(result.modified_ptx_path, sf_opts);
		result.success &= current_res.success;
		result.instructions_modified +=
			current_res.instructions_modified;
		result.modification_time_ms += current_res.modification_time_ms;
		result.modified_ptx_path = current_res.modified_ptx_path;
	}
	if (sf_opts.enable_verbose) {
		std::cout << "Modification on file: \t\t"
			  << result.modified_ptx_path << "\n\tStatus: "
			  << (result.success ? "Success" : "Failed")
			  << "\n\tInstructions Modified: "
			  << result.instructions_modified
			  << "\n\tModification Time(ms): "
			  << result.modification_time_ms << "\n";
	}
	std::cout << std::endl;

	return result;
}
