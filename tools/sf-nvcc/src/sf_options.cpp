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
 * @version 0.0.1
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-08-13: Initial File
 */

#include "sf_options.h"

#include <zlib.h>
#include <stdexcept>
#include <iostream>

extern unsigned char help_txt_gz[];
extern unsigned int help_txt_gz_len;
static constexpr int MAX_HELP_TEXT_LEN = 4096;

static inline std::string missing_input(const std::string &arg)
{
	return ACOL(ACOL_R,
		    ACOL_DF) "sf-nvcc parse error:" ACOL_RESET() "Missing argument for sf-nvcc argument: \"" +
	       arg + "\"\n";
}

static inline std::string bad_value(const std::string &arg,
				    const std::string &val)
{
	return ACOL(ACOL_R,
		    ACOL_DF) "sf-nvcc parse error:" ACOL_RESET() " Malformed value for sf-nvcc argument: \"" +
	       arg + "\" does not accept \"" + val + "\"\n";
}

safecuda::tools::SfNvccOptions
safecuda::tools::parse_command_line(const int argc, char *argv[])
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
			} else {
				throw std::invalid_argument(
					std::string{"Invalid argument: "} +
					arg);
			}

		} else {
			options.nvcc_args.emplace_back(arg);
		}
		arg_pos++;
	}

	return options;
}

void safecuda::tools::print_help()
{
	std::string out(MAX_HELP_TEXT_LEN, '\0');
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

inline void safecuda::tools::print_version()
{
	std::cout << ACOL(ACOL_G, ACOL_BB)
			     ACOL(ACOL_W, ACOL_DF) "SafeCUDA Version: "
		  << PROJECT_VERSION << ACOL_RESET() "\n";
}
