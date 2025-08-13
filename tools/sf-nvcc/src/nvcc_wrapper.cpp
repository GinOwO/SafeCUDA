/**
 * @file nvcc_wrapper.cpp
 * @brief Implementation of NVCC compiler wrapper
 *
 * Coordinates the PTX modification pipeline by intercepting NVCC calls,
 * generating PTX, modifying it for bounds checking, and resuming compilation.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-08-12
 * @version 0.0.1
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-08-12: Initial File
 * - 2025-08-13: Initial Implementation
 */

#include "nvcc_wrapper.h"

#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <ranges>
#include <iostream>
#include <unordered_set>

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
	return temp_files | std::views::filter([](const fs::path &p) {
		       return p.extension() == ".ptx";
	       }) |
	       std::ranges::to<std::vector<fs::path>>();
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

void sf_nvcc::generate_intermediate(const std::vector<std::string> &nvcc_args,
				    const SafeCudaOptions &sf_opts,
				    TemporaryFileManager &temp_mgr)
{
	static const std::unordered_set<std::string> extensions{
		".ii", ".c", ".ptx", ".gpu", ".cubin"};

	std::string command = "nvcc --keep -c --keep-dir=" +
			      temp_mgr.get_working_dir().string();

	for (const std::string &arg : nvcc_args)
		command += " " + arg;

	if (sf_opts.enable_verbose) {
		std::cout << ACOL(ACOL_Y, ACOL_BF)
			  << "Executing: " << ACOL_RESET() << command << "\n";
	}

	if (std::system(command.c_str())) {
		const std::string s{ACOL(
			ACOL_R, ACOL_DF) "NVCC command failed:\n" ACOL_RESET()};
		throw std::runtime_error(s + command);
	}

	for (const auto &p :
	     fs::directory_iterator(temp_mgr.get_working_dir())) {
		if (!fs::is_directory(p) &&
		    extensions.contains(p.path().extension().string())) {
			temp_mgr.add_file(p);
		}
	}
}

// sf_nvcc::PtxModificationResult
// sf_nvcc::modify_ptx(const fs::path &ptx_path, const SafeCudaOptions &sf_opts)
// {
// 	return {};
// }
//
// bool sf_nvcc::resume_nvcc(const std::vector<std::string> &intermediate_paths,
// 			  const std::vector<std::string> &nvcc_args)
// {
// 	return true;
// }
