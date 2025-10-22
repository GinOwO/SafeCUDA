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
 * - 2025-10-23: Integrated with libsafecuda_device.a for compilation
 * - 2025-09-23: Added support for resuming compilation of ptx with nvcc
 * - 2025-09-22: Added support for resuming compilation of modified ptx files
 * - 2025-08-13: Initial Implementation
 * - 2025-08-12: Initial File
 */

#include "nvcc_wrapper.h"

#include <cstdlib>
#include <ranges>
#include <iostream>
#include <unordered_set>
#include <fstream>

namespace sf_nvcc = safecuda::tools::sf_nvcc;
namespace fs = std::filesystem;

static constexpr char safecuda_lib[] =
#ifdef NDEBUG
	"./cmake-build-Release/libsafecuda_device.a"
#elif DEBUG
	return "./cmake-build-Debug/libsafecuda_device.a"
#else
	return "/usr/local/lib/libsafecuda_device.a"
#endif
	;

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

void sf_nvcc::generate_intermediate(const NvccOptions &nvcc_opts,
				    const SafeCudaOptions &sf_opts,
				    TemporaryFileManager &temp_mgr)
{
	static const std::unordered_set<std::string> extensions{
		".ii", ".c", ".ptx", ".gpu", ".cubin"};

	std::string command =
		"nvcc -Wno-deprecated-gpu-targets --keep -dc -rdc=true --keep-dir=" +
		temp_mgr.get_working_dir().string();

	for (const std::string &arg : nvcc_opts.nvcc_args)
		command += " " + arg;

	for (const std::string &arg : nvcc_opts.input_files)
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

bool sf_nvcc::resume_nvcc(const std::vector<fs::path> &ptx_paths,
			  const SfNvccOptions &sf_nvcc_opts,
			  TemporaryFileManager &temp_mgr)
{
	const SafeCudaOptions &sf_opts = sf_nvcc_opts.safecuda_opts;
	const NvccOptions &nvcc_opts = sf_nvcc_opts.nvcc_opts;

	if (sf_opts.enable_verbose) {
		std::cout << ACOL(ACOL_C, ACOL_BF)
			  << "Using SafeCUDA library: " << ACOL_RESET()
			  << safecuda_lib << "\n";
	}

	if (ptx_paths.empty()) {
		std::cerr << ACOL(ACOL_R, ACOL_DF)
			  << "Error: No PTX files provided for compilation"
			  << ACOL_RESET() << std::endl;
		return false;
	}

	std::string c_cpp_args, nvcc_args;
	std::vector<std::string> cu_files = nvcc_opts.input_files;

	for (size_t i = 0; i < nvcc_opts.nvcc_args.size(); i++) {
		const std::string &arg = nvcc_opts.nvcc_args[i];
		if (arg.ends_with(".cpp")) {
			c_cpp_args += arg + " ";
		} else if (arg == "--generate-code") {
			i++; // Skip arch specification
		} else {
			nvcc_args += arg + " ";
		}
	}

	// compile .cu files for HOST functions only (no device code)
	std::vector<std::string> cu_obj_paths;
	for (const auto &cu_file : cu_files) {
		fs::path cu_obj_path =
			temp_mgr.get_working_dir() /
			(fs::path(cu_file).stem().string() + "_host.o");

		std::string cu_compile_cmd =
			"nvcc -Wno-deprecated-gpu-targets --compile " +
			cu_file + " -o " + cu_obj_path.string();

		if (sf_opts.enable_verbose) {
			std::cout << ACOL(ACOL_Y, ACOL_BF)
				  << "Compiling .cu host code\nExecuting: "
				  << ACOL_RESET() << cu_compile_cmd << "\n";
		}

		if (std::system(cu_compile_cmd.c_str())) {
			throw std::runtime_error(
				"CU host compilation failed: " +
				cu_compile_cmd);
		}

		cu_obj_paths.push_back(cu_obj_path.string());
		temp_mgr.add_file(cu_obj_path);
	}

	std::vector<std::string> ptx_obj_paths;
	for (const auto &ptx_path : ptx_paths) {
		if (!fs::exists(ptx_path)) {
			std::cerr << ACOL(ACOL_R, ACOL_DF)
				  << "Error: PTX file not found: "
				  << ptx_path.string() << ACOL_RESET()
				  << std::endl;
			return false;
		}

		std::string filename = ptx_path.stem().string();
		std::string arch_flag;

		size_t compute_pos = filename.find("compute_");
		if (compute_pos != std::string::npos) {
			std::string compute_ver =
				filename.substr(compute_pos + 8);
			arch_flag = " -arch=sm_" + compute_ver;
		}

		fs::path ptx_obj_path = temp_mgr.get_working_dir() /
					(ptx_path.stem().string() + ".o");

		std::string ptx_compile_cmd =
			"nvcc -Wno-deprecated-gpu-targets --device-c" +
			arch_flag + " " + ptx_path.string() + " -o " +
			ptx_obj_path.string();

		if (sf_opts.enable_verbose) {
			std::cout << ACOL(ACOL_Y, ACOL_BF)
				  << "Compiling PTX to object\nExecuting: "
				  << ACOL_RESET() << ptx_compile_cmd << "\n";
		}

		if (std::system(ptx_compile_cmd.c_str())) {
			throw std::runtime_error("PTX compilation failed: " +
						 ptx_compile_cmd);
		}

		ptx_obj_paths.push_back(ptx_obj_path.string());
		temp_mgr.add_file(ptx_obj_path);
	}

	// Device link PTX objects + libsafecuda.so
	fs::path dlink_path = temp_mgr.get_working_dir() / "kernels_dlink.o";
	std::string dlink_command = "nvcc -dlink ";
	for (const auto &obj : ptx_obj_paths) {
		dlink_command += obj + " ";
	}
	dlink_command += safecuda_lib;
	dlink_command += " -o " + dlink_path.string();

	if (sf_opts.enable_verbose) {
		std::cout << ACOL(ACOL_Y, ACOL_BF)
			  << "Device linking PTX objects\nExecuting: "
			  << ACOL_RESET() << dlink_command << "\n";
	}

	if (std::system(dlink_command.c_str())) {
		throw std::runtime_error("Device linking failed: " +
					 dlink_command);
	}
	temp_mgr.add_file(dlink_path);

	// combine everything
	std::string final_command = "nvcc " + nvcc_args + " " + c_cpp_args;

	for (const auto &obj : cu_obj_paths) {
		final_command += obj + " ";
	}
	for (const auto &obj : ptx_obj_paths) {
		final_command += obj + " ";
	}

	final_command += dlink_path.string() + " " + safecuda_lib;
	final_command += " -o " + nvcc_opts.output_path;

	if (sf_opts.enable_verbose) {
		std::cout << ACOL(ACOL_Y, ACOL_BF)
			  << "Final linking\nExecuting: " << ACOL_RESET()
			  << final_command << "\n";
	}

	if (std::system(final_command.c_str())) {
		throw std::runtime_error("Final linking failed: " +
					 final_command);
	}

	return true;
}
