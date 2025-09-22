/**
 * @file sf_options.h
 * @brief Struct to pass around sf-nvcc command line options
 *
 * Contains data structures and utilities for parsing and managing command line
 * options for the sf-nvcc wrapper tool. Separates SafeCUDA-specific options
 * (prefixed with -sf-) from standard NVCC options that get passed through.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-08-13
 * @version 0.0.2
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-08-13: Added output file switch
 * - 2025-08-13: Initial File
 */

#ifndef SAFECUDA_SF_OPTIONS_H
#define SAFECUDA_SF_OPTIONS_H

#include <string>
#include <vector>

// ANSI COLOR BUILDER
#define ACOL(C, TB) "\033[" TB C "m"
#define ACOL_RESET() "\033[0m"

#define ACOL_DF "3"
#define ACOL_BF "9"
#define ACOL_DB "4"
#define ACOL_BB "10"

#define ACOL_K "0"
#define ACOL_R "1"
#define ACOL_G "2"
#define ACOL_Y "3"
#define ACOL_BL "4"
#define ACOL_M "5"
#define ACOL_C "6"
#define ACOL_W "7"

namespace safecuda::tools::sf_nvcc
{
/**
 * @brief Configuration options for SafeCUDA PTX modification
 *
 * Holds all SafeCUDA-specific options parsed from -sf-* command line arguments.
 */
struct SafeCudaOptions {
	bool enable_bounds_check = true; ///< Enable runtime bounds checking.
	bool enable_debug = false; ///< Enable debug instrumentation.
	bool enable_verbose = false; ///< Enable verbose logging.
	bool fail_fast = false; ///< Abort on first violation.
	bool log_violations = false; ///< Log memory violations.
	std::string log_file{"stderr"}; ///< Path for violation logs.
	std::string keep_dir; ///< Directory to store intermediate files.
	std::string output_path; ///< Directory to store final output
};

/**
 * @brief Complete option set for sf-nvcc compilation
 *
 * Contains both SafeCUDA-specific options and standard NVCC arguments.
 */
struct SfNvccOptions {
	SafeCudaOptions safecuda_opts; ///< SafeCUDA-specific options.
	std::vector<std::string> nvcc_args; ///< Remaining NVCC arguments.
};

/**
 * @brief Parse command line arguments into structured options
 *
 * Separates SafeCUDA-specific options (starting with -sf-) from standard
 * NVCC arguments. Validates option syntax and provides error reporting.
 *
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return SfNvccOptions Parsed options structure
 * @throws std::invalid_argument for malformed SafeCUDA options
 *
 * @note --keep, --keep-dir, -c, --ptx to nvcc are ignored
 */
SfNvccOptions parse_command_line(int argc, char *argv[]);

/**
 * @brief Print sf-nvcc help text.
 */
void print_help();

/**
 * @brief Print sf-nvcc version info.
 */
void print_version();

/**
 * @brief Print verbose arguments
 */
void print_args(const SafeCudaOptions &safecuda_opts,
		const std::vector<std::string> &nvcc_args);

} // namespace safecuda::tools

#endif // SAFECUDA_SF_OPTIONS_H
