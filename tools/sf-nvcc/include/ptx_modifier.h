/**
 * @file ptx_modifier.h
 * @brief PTX modification engine for injecting SafeCUDA bounds checking
 *
 * Core logic for injecting bounds checking instructions into
 * load/store instructions.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-08-12
 * @version 0.0.2
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-09-22: Initial Implementation
 * - 2025-08-12: Initial File
 */

#ifndef SAFECUDA_PTX_MODIFIER_H
#define SAFECUDA_PTX_MODIFIER_H

#include "sf_options.h"

#include <filesystem>

namespace safecuda::tools::sf_nvcc
{
/**
 * @brief Result of PTX modification operation with detailed statistics
 *
 * Contains success status, modification counts, and diagnostic messages
 * for logging and error reporting purposes.
 */
struct PtxModificationResult {
	bool success;
	size_t instructions_modified{};
	std::string modified_ptx_path;
	uint64_t modification_time_ms = 0.0;

	PtxModificationResult()
		: success(false) {};
};

/**
 * @brief Inserts bounds checking ptx instructions into the ptx file
 *
 * Inserts ptx guards into the file depending on values in sf_opts
 * If enable_debug is turned on, then
 *
 * @param ptx_path path of the ptx file to modify
 * @param sf_opts SafeCudaOptions with flags for certain functionality
 * @return PtxModificationResult with result of PTX modifications
 */
PtxModificationResult insert_bounds_check(const std::filesystem::path &ptx_path,
					  const SafeCudaOptions &sf_opts);

/**
 * @brief Insert guards to a generated PTX file
 *
 * Perform parsing PTX assembly, identifying memory operations,
 * and injecting bounds checking guards etc into load/store instructions.
 *
 * @param ptx_path Path of ptx file to be modified
 * @param sf_opts SafeCUDA options for controlling PTX modification
 * @return PtxModificationResult done by the function for logging purposes
 *
 * @throws std::runtime_error if PTX parsing fails or modification generates invalid code
 * @throws std::filesystem::filesystem_error if modified files cannot be written
 * @note Original files are preserved with .bak suffix for debugging if sf_opts.enable_debug is true
 */
PtxModificationResult modify_ptx(const std::filesystem::path &ptx_path,
				 const SafeCudaOptions &sf_opts);
} // namespace safecuda::tools::nvcc

#endif // SAFECUDA_PTX_MODIFIER_H
