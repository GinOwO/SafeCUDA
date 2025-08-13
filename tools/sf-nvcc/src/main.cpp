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

#include "sf_options.h"

#include <iostream>
#include <string>
#include <cstdlib>

using namespace safecuda;

int main(int argc, char *argv[])
{
	try {
		auto [safecuda_opts, nvcc_args] =
			tools::parse_command_line(argc, argv);
		if (safecuda_opts.enable_verbose) {
			const auto &[enable_bounds_check, enable_debug,
				     enable_verbose, cache_size, fail_fast,
				     log_violations, log_file] = safecuda_opts;
			std::cout << ACOL(ACOL_Y, ACOL_BB)
				  << ACOL(ACOL_K, ACOL_DF)
				  << "sf-nvcc options:" ACOL_RESET() "\n\t"
				  << "enable_bounds_check: " << std::boolalpha
				  << enable_bounds_check << "\n\t"
				  << "enable_debug: " << enable_debug << "\n\t"
				  << "enable_verbose: " << enable_verbose
				  << "\n\t"
				  << "cache_size: " << cache_size << "\n\t"
				  << "fail_fast: " << fail_fast << "\n\t"
				  << "log_violations: " << log_violations
				  << "\n\t"
				  << "log_file: "
				  << (log_file.empty() ? "<none>" : log_file)
				  << "\n\t"
				  << "nvcc_args: \n\t\t";
			for (const auto &arg : nvcc_args)
				std::cout << arg << "\n\t\t";
			std::cout << "\n";
		}
	} catch (std::invalid_argument &e) {
		std::cerr << e.what();
	}

	return EXIT_SUCCESS;
}
