/**
 * @file main.cpp
 * @brief Entry point for sf-nvcc SafeCUDA compiler wrapper
 *
 * Main executable that acts as a drop-in replacement for nvcc,
 * providing automatic PTX modification for bounds checking injection.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-08-12
 * @version 1.0.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
* Change Log:
 * - 2025-10-28: Refactored to use nvcc -dryrun based pipeline
 * - 2025-09-23: Finalized core components
 * - 2025-08-12: Initial File
 */

#include "nvcc_wrapper.h"
#include "sf_options.h"
#include <iostream>
#include <stdexcept>
#include <unordered_map>

using namespace safecuda;

int main(const int argc, char *argv[])
{
	try {
		tools::sf_nvcc::SfNvccOptions sf_nvcc_options =
			tools::sf_nvcc::parse_command_line(argc, argv);

		auto [safecuda_opts, nvcc_args] = sf_nvcc_options;

		tools::sf_nvcc::TemporaryFileManager temp_mgr(safecuda_opts);

		if (safecuda_opts.enable_verbose)
			tools::sf_nvcc::print_args(safecuda_opts, nvcc_args);

		// Execute nvcc -dryrun to get compilation pipeline
		tools::sf_nvcc::DryRunParser parser =
			tools::sf_nvcc::execute_dryrun(nvcc_args,
						       safecuda_opts);

		// Execute commands up to and including PTX generation
		std::vector<std::filesystem::path> ptx_paths =
			tools::sf_nvcc::execute_pre_ptx_stage(
				parser, safecuda_opts, temp_mgr);

		// Modify all generated PTX files
		std::unordered_map<std::string, std::string> modified_ptx_map;
		for (const auto &ptx_path : ptx_paths) {
			auto result = tools::sf_nvcc::modify_ptx(ptx_path,
								 safecuda_opts);
			modified_ptx_map[ptx_path.string()] =
				result.modified_ptx_path;
		}

		std::fflush(stdout);

		// Execute remaining compilation commands with modified PTX
		tools::sf_nvcc::execute_post_ptx_stage(
			modified_ptx_map, parser, safecuda_opts, nvcc_args);

	} catch (std::invalid_argument &e) {
		std::cerr << e.what() << '\n';
		return EXIT_FAILURE;
	} catch (std::runtime_error &e) {
		std::cerr << e.what() << '\n';
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
