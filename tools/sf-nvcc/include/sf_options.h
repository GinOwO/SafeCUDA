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
 * @version 1.1.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-10-28: Refactored to use nvcc -dryrun command orchestration
 * - 2025-10-23: Changed fail fast functionality and so default is true now
 * - 2025-09-23: Rewrote stuff around NvccOptions
 * - 2025-09-22: Removed some switches, moved stuff around for more modularity
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
	bool fail_fast = true; ///< Abort on first violation.
	bool log_violations = false; ///< Log memory violations.
	std::string log_file{"stderr"}; ///< Path for violation logs.
	std::string keep_dir; ///< Directory to store intermediate files.
};

/**
 * @brief Configuration options for NVCC
 *
 * Holds all NVCC-specific options parsed from command line arguments.
 */
struct NvccOptions {
	std::vector<std::string> input_files; ///< NVCC input files
	std::vector<std::string> nvcc_args; ///< Remaining NVCC arguments.
	std::string output_path; ///< NVCC output file path
};

/**
 * @brief Complete option set for sf-nvcc compilation
 *
 * Contains both SafeCUDA-specific options and standard NVCC arguments.
 */
struct SfNvccOptions {
	SafeCudaOptions safecuda_opts; ///< SafeCUDA-specific options.
	NvccOptions nvcc_opts; ///< NVCC arguments
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
 * @note All NVCC arguments are preserved and passed through except -dryrun
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
		const NvccOptions &nvcc_opts);

} // namespace safecuda::tools::sf_nvcc

#endif // SAFECUDA_SF_OPTIONS_H
