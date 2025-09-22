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
#include "sf_options.h"

#include <iostream>
#include <string>
#include <cstdlib>

using namespace safecuda;

int main(const int argc, char *argv[])
{
	try {
		auto [safecuda_opts, nvcc_args] =
			tools::sf_nvcc::parse_command_line(argc, argv);
		tools::sf_nvcc::TemporaryFileManager temp_mgr(safecuda_opts);
		if (safecuda_opts.enable_verbose)
			tools::sf_nvcc::print_args(safecuda_opts, nvcc_args);

		tools::sf_nvcc::generate_intermediate(nvcc_args, safecuda_opts,
						      temp_mgr);

		for (const auto &path : temp_mgr.filter_ptx_paths()) {
			if (safecuda_opts.enable_verbose)
				std::cout << path << '\n';
			// TODO: call modify here
			// tools::sf_nvcc::modify_ptx(path, safecuda_opts);
		}
		// tools::sf_nvcc::resume_nvcc(temp_mgr.get_intermediate_files(),
		// 			    nvcc_args);
	} catch (std::invalid_argument &e) {
		std::cerr << e.what() << '\n';
		return EXIT_FAILURE;
	} catch (std::runtime_error &e) {
		std::cerr << e.what() << '\n';
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
