/**
 * @file nvcc_wrapper.cpp
 * @brief Implementation of NVCC compiler wrapper
 *
 * Coordinates the PTX modification pipeline by intercepting NVCC calls,
 * generating PTX, modifying it for bounds checking, and resuming compilation.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-08-12
 * @version 1.0.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * Change Log:
 * - 2025-10-28: Refactored to use nvcc -dryrun command orchestration
 * - 2025-10-23: Integrated with libsafecuda_device.a for compilation,
 *               modified functionality of fail fast
 * - 2025-09-23: Added support for resuming compilation of ptx with nvcc
 * - 2025-09-22: Added support for resuming compilation of modified ptx files
 * - 2025-08-13: Initial Implementation
 * - 2025-08-12: Initial File
 */

#include "nvcc_wrapper.h"
#include <array>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>

namespace sf_nvcc = safecuda::tools::sf_nvcc;
namespace fs = std::filesystem;

sf_nvcc::TemporaryFileManager::TemporaryFileManager(
	const SafeCudaOptions &sf_opts)
	: preserve_on_exit(sf_opts.enable_debug)
{
	if (!fs::exists(sf_opts.keep_dir))
		fs::create_directories(sf_opts.keep_dir);
	dir_path = sf_opts.keep_dir;
};

sf_nvcc::TemporaryFileManager::~TemporaryFileManager()
{
	if (!preserve_on_exit)
		fs::remove_all(this->dir_path);
}

inline fs::path sf_nvcc::TemporaryFileManager::get_working_dir() const noexcept
{
	return this->dir_path;
}

std::vector<fs::path>
sf_nvcc::TemporaryFileManager::filter_ptx_paths() const noexcept
{
	std::vector<fs::path> result;
	for (const auto &p : temp_files) {
		if (p.extension() == ".ptx")
			result.push_back(p);
	}
	return result;
}

std::vector<fs::path>
sf_nvcc::TemporaryFileManager::get_intermediate_files() const noexcept
{
	return this->temp_files;
}

void sf_nvcc::TemporaryFileManager::add_file(const fs::path &path) noexcept
{
	temp_files.emplace_back(path);
}

/**
 * @brief Check if a string is an environment variable assignment
 *
 * Variable assignments have format: VARIABLE_NAME=value
 * Commands typically start with quotes, slashes, or executable names
 *
 * @param str String to check
 * @return true if string is a variable assignment
 */
static bool is_variable_assignment(const std::string &str)
{
	if (str.empty())
		return false;

	size_t equals_pos = str.find('=');
	if (equals_pos == std::string::npos || equals_pos == 0)
		return false;

	for (size_t i = 0; i < equals_pos; ++i) {
		char c = str[i];
		if (!std::isupper(c) && c != '_' && !std::isdigit(c)) {
			return false;
		}
	}

	return true;
}

namespace
{
struct PCloseDeleter {
	void operator()(FILE *f) const
	{
		if (f)
			pclose(f);
	}
};

};

/**
 * @brief Execute a shell command and capture its output
 *
 * @param cmd Command to execute
 * @return Command output as string
 * @throws std::runtime_error if command execution fails
 */
static std::string exec_command(const std::string &cmd)
{
	std::array<char, 128> buffer{};
	std::string result;

	const std::unique_ptr<FILE, PCloseDeleter> pipe(
		popen(cmd.c_str(), "r"));
	if (!pipe) {
		throw std::runtime_error("popen() failed for command: " + cmd);
	}
	while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
		result += buffer.data();
	}
	return result;
}

void sf_nvcc::DryRunParser::parse(const std::string &dryrun_output)
{
	std::istringstream iss(dryrun_output);
	std::string line;
	size_t cmd_index = 0;

	std::regex ptx_output_regex(R"ptx(-o\s+"?([^"\s]+\.ptx)"?)ptx");

	while (std::getline(iss, line)) {
		if (line.empty() || !line.starts_with("#$"))
			continue;

		std::string command = line.substr(3);

		if (command.starts_with("rm "))
			continue;

		if (is_variable_assignment(command)) {
			if (command.find("LIBRARIES") != std::string::npos) {
				command =
					R"(LIBRARIES="-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs":"-L/usr/local/cuda/bin/../targets/x86_64-linux/lib")";
			}
			auto pos = command.find('=');
			if (pos == std::string::npos)
				return;

			std::string key = command.substr(0, pos);
			std::string val = command.substr(pos + 1);

			while (!val.empty() && (val.front() == ' '))
				val.erase(val.begin());
			while (!val.empty() && (val.back() == ' '))
				val.pop_back();

			if (val.size() >= 2 && val.front() == '"' &&
			    val.back() == '"') {
				val = val.substr(1, val.size() - 2);
			}

			setenv(key.c_str(), val.c_str(), 1);
			continue;
		}

		commands.push_back(command);

		if (command.find("cicc") != std::string::npos) {
			std::smatch match;
			if (std::regex_search(command, match,
					      ptx_output_regex)) {
				std::string ptx_path = match[1].str();
				ptx_files[ptx_path] = ptx_path;
				ptx_stage_end_index = cmd_index + 1;
			}
		}

		cmd_index++;
	}

	if (ptx_files.empty()) {
		throw std::runtime_error(
			"No PTX generation commands found in nvcc -dryrun output");
	}
}

std::vector<std::string> sf_nvcc::DryRunParser::get_pre_ptx_commands() const
{
	return std::vector<std::string>(commands.begin(),
					commands.begin() + ptx_stage_end_index);
}

std::vector<std::string> sf_nvcc::DryRunParser::get_post_ptx_commands(
	const std::unordered_map<std::string, std::string> &modified_ptx_map)
	const
{
	std::vector<std::string> post_commands(
		commands.begin() + ptx_stage_end_index, commands.end());

	for (auto &cmd : post_commands) {
		for (const auto &[orig_ptx, modified_ptx] : modified_ptx_map) {
			size_t pos = 0;
			while ((pos = cmd.find(orig_ptx, pos)) !=
			       std::string::npos) {
				cmd.replace(pos, orig_ptx.length(),
					    modified_ptx);
				pos += modified_ptx.length();
			}
		}
	}

	return post_commands;
}

sf_nvcc::DryRunParser sf_nvcc::execute_dryrun(const NvccOptions &nvcc_opts,
					      const SafeCudaOptions &sf_opts)
{
	std::string dryrun_cmd = "nvcc -dryrun -lcudart";

	for (const std::string &arg : nvcc_opts.nvcc_args) {
		dryrun_cmd += " " + arg;
	}

	if (sf_opts.enable_verbose) {
		std::cout << ACOL(ACOL_Y, ACOL_BF)
			  << "Executing dryrun: " << ACOL_RESET() << dryrun_cmd
			  << "\n";
	}

	std::string dryrun_output;
	try {
		dryrun_output = exec_command(dryrun_cmd + " 2>&1");
	} catch (const std::exception &e) {
		throw std::runtime_error(
			std::string("nvcc -dryrun execution failed: ") +
			e.what());
	}

	if (sf_opts.enable_verbose) {
		std::cout << ACOL(ACOL_C, ACOL_DF) << "Dryrun output:\n"
			  << ACOL_RESET() << dryrun_output << "\n";
	}

	DryRunParser parser;
	parser.parse(dryrun_output);

	if (sf_opts.enable_verbose) {
		std::cout << ACOL(ACOL_G, ACOL_BF) << "Found "
			  << parser.ptx_files.size()
			  << " PTX file(s) to generate" << ACOL_RESET() << "\n";
		std::cout
			<< ACOL(ACOL_C, ACOL_DF)
			<< "Total commands: " << parser.commands.size()
			<< ", Pre-PTX commands: " << parser.ptx_stage_end_index
			<< ACOL_RESET() << "\n";
	}

	return parser;
}

std::vector<fs::path>
sf_nvcc::execute_pre_ptx_stage(const DryRunParser &parser,
			       const SafeCudaOptions &sf_opts,
			       TemporaryFileManager &temp_mgr)
{
	auto pre_ptx_cmds = parser.get_pre_ptx_commands();

	if (sf_opts.enable_verbose) {
		std::cout << ACOL(ACOL_Y, ACOL_BB) << ACOL(ACOL_K, ACOL_DF)
			  << "Executing pre-PTX compilation stage ("
			  << pre_ptx_cmds.size() << " commands)" << ACOL_RESET()
			  << "\n";
	}

	for (const auto &cmd : pre_ptx_cmds) {
		if (sf_opts.enable_verbose) {
			std::cout << ACOL(ACOL_C, ACOL_DF)
				  << "Executing: " << ACOL_RESET() << cmd
				  << "\n";
		}

		int ret = std::system(cmd.c_str());
		if (ret != 0) {
			throw std::runtime_error(
				"Pre-PTX command failed with exit code " +
				std::to_string(ret) + ": " + cmd);
		}
	}

	std::vector<fs::path> ptx_paths;
	for (const auto &[ptx_file, _] : parser.ptx_files) {
		fs::path ptx_path(ptx_file);
		if (fs::exists(ptx_path)) {
			ptx_paths.push_back(ptx_path);
			temp_mgr.add_file(ptx_path);

			if (sf_opts.enable_verbose) {
				std::cout << ACOL(ACOL_G, ACOL_BF)
					  << "Generated PTX: " << ACOL_RESET()
					  << ptx_path << "\n";
			}
		} else {
			throw std::runtime_error(
				"Expected PTX file not generated: " + ptx_file);
		}
	}

	return ptx_paths;
}

bool sf_nvcc::execute_post_ptx_stage(
	const std::unordered_map<std::string, std::string> &modified_ptx_map,
	const DryRunParser &parser, const SafeCudaOptions &sf_opts,
	const NvccOptions &nvcc_opts)
{
	auto post_ptx_cmds = parser.get_post_ptx_commands(modified_ptx_map);

	if (sf_opts.enable_verbose) {
		std::cout << ACOL(ACOL_Y, ACOL_BB) << ACOL(ACOL_K, ACOL_DF)
			  << "Executing post-PTX compilation stage ("
			  << post_ptx_cmds.size() << " commands)"
			  << ACOL_RESET() << "\n";
	}

	const std::string op_path{"-o \"" + nvcc_opts.output_path + "\""};

	for (auto &cmd : post_ptx_cmds) {
		if (sf_opts.enable_verbose) {
			std::cout << ACOL(ACOL_C, ACOL_DF)
				  << "Executing: " << ACOL_RESET() << cmd
				  << "\n";
		}

		int ret = std::system(cmd.c_str());
		if (ret != 0) {
			throw std::runtime_error(
				"Post-PTX command failed with exit code " +
				std::to_string(ret) + ": " + cmd);
		}
	}

	return true;
}
