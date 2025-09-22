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
 * @brief Inserts bounds checking ptx instructions into the ptx file
 *
 * Inserts ptx guards into the file depending on values in sf_opts
 * If enable_debug is turned on, then
 *
 * @param ptx_path path of the ptx file to modify
 * @param sf_opts SafeCudaOptions with flags for certain functionality
 * @return boolean indicating success or failure
 */
bool insert_bounds_check(const std::filesystem::path &ptx_path,
			 const sf_nvcc::SafeCudaOptions &sf_opts);
} // namespace safecuda::tools::nvcc

#endif // SAFECUDA_PTX_MODIFIER_H
