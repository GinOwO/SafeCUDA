/**
 * @file sf_options.cpp
 * @brief Implementation of struct to pass around sf-nvcc command line options
 *
 * Contains data structures and utilities for parsing and managing command line
 * options for the sf-nvcc wrapper tool. Separates SafeCUDA-specific options
 * (prefixed with -sf-) from standard NVCC options that get passed through.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-08-13
 * @version 1.0.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-09-22: Removed some switches, moved stuff around for more modularity
 * - 2025-08-13: Enabling debug also now enables verbose, added output switch
 * - 2025-08-13: Initial File
 */

#include "sf_options.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <filesystem>

#include <zlib.h>

namespace sf_nvcc = safecuda::tools::sf_nvcc;

extern unsigned char help_txt_gz[];
extern unsigned int help_txt_gz_len;

static std::string missing_input(const std::string &arg)
{
	return ACOL(ACOL_K, ACOL_DF) ACOL(
		       ACOL_R,
		       ACOL_BB) "sf-nvcc parse error:" ACOL_RESET() "Missing argument for sf-nvcc argument: \"" +
	       arg + "\"\n";
}

static std::string bad_value(const std::string &arg, const std::string &val)
{
	return ACOL(ACOL_K, ACOL_DF) ACOL(
		       ACOL_R,
		       ACOL_BB) "sf-nvcc parse error:" ACOL_RESET() " Malformed value for sf-nvcc argument: \"" +
	       arg + "\" does not accept \"" + val + "\"\n";
}

/* Sample nvcc command
nvcc \
	-O3 -DNDEBUG -Xcompiler -fPIC \
	-Wno-deprecated-gpu-targets \
	--extended-lambda \
	--expt-relaxed-constexpr \
	--generate-code arch=compute_52,code=sm_52 \
	--generate-code arch=compute_60,code=sm_60 \
	--generate-code arch=compute_61,code=sm_61 \
	--generate-code arch=compute_75,code=sm_75 \
	--generate-code arch=compute_86,code=sm_86 \
	-rdc=true \
	examples/example.cpp \
	examples/kernel1.cu \
	examples/kernel2.cu \
	-o example
 */

sf_nvcc::SfNvccOptions sf_nvcc::parse_command_line(const int argc, char *argv[])
{
	SfNvccOptions options;
	int arg_pos = 1;

	while (arg_pos < argc) {
		if (std::string arg{argv[arg_pos]}; arg.starts_with("-sf-")) {
			if (arg == "-sf-help") {
				print_help();
				std::exit(EXIT_SUCCESS);
			}

			if (arg == "-sf-version") {
				print_version();
			} else if (++arg_pos >= argc) {
				throw std::invalid_argument(missing_input(arg));
			} else if (std::string val{argv[arg_pos]};
				   arg == "-sf-bounds-check") {
				if (val == "true")
					options.safecuda_opts
						.enable_bounds_check = true;
				else if (val == "false")
					options.safecuda_opts
						.enable_bounds_check = false;
				else
					throw std::invalid_argument(
						bad_value(arg, val));
			} else if (arg == "-sf-debug") {
				if (val == "true")
					options.safecuda_opts.enable_debug =
						true;
				else if (val == "false")
					options.safecuda_opts.enable_debug =
						false;
				else
					throw std::invalid_argument(
						bad_value(arg, val));
			} else if (arg == "-sf-verbose") {
				if (val == "true")
					options.safecuda_opts.enable_verbose =
						true;
				else if (val == "false")
					options.safecuda_opts.enable_verbose =
						false;
				else
					throw std::invalid_argument(
						bad_value(arg, val));
			} else if (arg == "-sf-fail-fast") {
				if (val == "true")
					options.safecuda_opts.fail_fast = true;
				else if (val == "false")
					options.safecuda_opts.fail_fast = false;
				else
					throw std::invalid_argument(
						bad_value(arg, val));
			} else if (arg == "-sf-log-violations") {
				if (val == "true")
					options.safecuda_opts.log_violations =
						true;
				else if (val == "false")
					options.safecuda_opts.log_violations =
						false;
				else
					throw std::invalid_argument(
						bad_value(arg, val));
			} else if (arg == "-sf-log-path") {
				options.safecuda_opts.log_file = val;
			} else if (arg == "-sf-keep-dir") {
				options.safecuda_opts.keep_dir = val;
			} else {
				throw std::invalid_argument(
					std::string{"Invalid argument: "} +
					arg);
			}

		} else {
			if (arg == "-o") {
				if (++arg_pos >= argc) {
					throw std::runtime_error(
						missing_input(arg));
				}
				options.nvcc_opts.output_path =
					std::string(argv[arg_pos]);
			} else if (arg.ends_with(".cu")) {
				options.nvcc_opts.input_files.emplace_back(
					argv[arg_pos]);
			} else if (!(arg.starts_with("--keep") ||
				     arg.starts_with("-rdc") || arg == "-c" ||
				     arg == "-dc" || arg == "-dlink" ||
				     arg == "--ptx")) {
				options.nvcc_opts.nvcc_args.emplace_back(arg);
			}
		}
		arg_pos++;
	}

	if (options.safecuda_opts.keep_dir.empty()) {
#if defined(__unix__) || defined(__APPLE__) || defined(__linux__)
		char tmpl[] = "/tmp/sf-nvccXXXXXX";
		const int fd = mkstemp(tmpl);
		if (fd == -1)
			throw std::runtime_error("mkstemp failed");
		close(fd);
		std::string path(tmpl);
		std::filesystem::remove(path);
#else
		char buf[L_tmpnam];
		if (tmpnam(buf) == nullptr)
			throw std::runtime_error("tmpnam failed");
		std::string path(buf);
		path = std::filesystem::temp_directory_path() / path;
#endif
		options.safecuda_opts.keep_dir = path;
	}

	if (options.safecuda_opts.enable_debug)
		options.safecuda_opts.enable_verbose = true;

	return options;
}

void sf_nvcc::print_help()
{
	std::string out(HELP_TEXT_LEN, '\0');
	z_stream zs{};
	zs.next_in = help_txt_gz;
	zs.avail_in = help_txt_gz_len;

	if (inflateInit2(&zs, 16 + MAX_WBITS) != Z_OK)
		throw std::runtime_error("inflateInit failed");

	zs.next_out = reinterpret_cast<Bytef *>(&out[0]);
	zs.avail_out = out.size();

	if (int ret = inflate(&zs, Z_FINISH); ret != Z_STREAM_END)
		throw std::runtime_error("inflate failed");

	inflateEnd(&zs);
	out.resize(zs.total_out);
	std::cout << out << '\n';
}

inline void sf_nvcc::print_version()
{
	std::cout << ACOL(ACOL_G, ACOL_BB)
			     ACOL(ACOL_W, ACOL_DF) "SafeCUDA Version: "
		  << PROJECT_VERSION << ACOL_RESET() "\n";
}

void sf_nvcc::print_args(const SafeCudaOptions &safecuda_opts,
			 const NvccOptions &nvcc_opts)
{
	const auto &[enable_bounds_check, enable_debug, enable_verbose,
		     fail_fast, log_violations, log_file,
		     keep_dir] = safecuda_opts;
	std::cout << ACOL(ACOL_Y, ACOL_BB) << ACOL(ACOL_K, ACOL_DF)
		  << "sf-nvcc options:" ACOL_RESET() "\n\t"
		  << "enable_bounds_check: " << std::boolalpha
		  << enable_bounds_check << "\n\t"
		  << "enable_debug: " << enable_debug << "\n\t"
		  << "enable_verbose: " << enable_verbose << "\n\t"
		  << "fail_fast: " << fail_fast << "\n\t"
		  << "log_violations: " << log_violations << "\n\t"
		  << "log_file: " << (log_file.empty() ? "<none>" : log_file)
		  << "\n\t"
		  << "keep_dir: " << keep_dir << "\n\t"
		  << "output_path: " << nvcc_opts.output_path << "\n\t"
		  << "nvcc_args: \n\t\t";

	std::cout << "Input args:\n\t\t";
	for (const auto &arg : nvcc_opts.input_files)
		std::cout << arg << "\n\t\t";

	std::cout << "\n\t\tOther args:\n\t\t";
	for (const auto &arg : nvcc_opts.nvcc_args)
		std::cout << arg << "\n\t\t";

	std::cout << "\n";
}
