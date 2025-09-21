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
 * @version 0.0.2
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
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

static inline std::string missing_input(const std::string &arg)
{
	return ACOL(ACOL_K, ACOL_DF) ACOL(
		       ACOL_R,
		       ACOL_BB) "sf-nvcc parse error:" ACOL_RESET() "Missing argument for sf-nvcc argument: \"" +
	       arg + "\"\n";
}

static inline std::string bad_value(const std::string &arg,
				    const std::string &val)
{
	return ACOL(ACOL_K, ACOL_DF) ACOL(
		       ACOL_R,
		       ACOL_BB) "sf-nvcc parse error:" ACOL_RESET() " Malformed value for sf-nvcc argument: \"" +
	       arg + "\" does not accept \"" + val + "\"\n";
}

/* Sample nvcc command
nvcc \
	-g -G -O0 -Xcompiler -fPIC \
	-Xptxas -O0 \
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
			} else if (arg == "-sf-version") {
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
			} else if (arg == "-sf-cache-size") {
				try {
					options.safecuda_opts.cache_size =
						std::stoi(val);
				} catch (std::invalid_argument &_) {
					throw std::invalid_argument(
						bad_value(arg, val));
				}
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
					throw missing_input("-o");
				}
				options.safecuda_opts.output_path =
					std::string(argv[arg_pos]);
			} else if (!(arg.starts_with("--keep") || arg == "-c" ||
				     arg == "--ptx"))
				options.nvcc_args.emplace_back(arg);
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
