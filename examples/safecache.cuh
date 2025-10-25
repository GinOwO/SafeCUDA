/**
 * @file safecache.cuh
 * @brief SafeCUDA fallback cache header file
 *
 * This file contains the header file for the dynamic fallback cache
 *
 * @author Anirudh <anirudh.sridhar2022@vitstudent.ac.in>
 * @date 2025-09-22
 * @version 0.1.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-10-23: Reworked Errors to work with bitwise OR as well
 * - 2025-10-22: Removed redundancies and fixed a [[noreturn]] bug on check_cuda
 * - 2025-10-15: Revised code to make class vars non-static and redeclare
 *		 constructor, destructor etc. Also, some other code changes.
 * - 2025-10-11: Revised code to be device and host side functions.
 * - 2025-09-22: Initial implementation
 */

#ifndef SAFECACHE_H
#define SAFECACHE_H

#include <cuda_runtime.h>
#include <cstdint>

namespace safecuda::memory
{

struct Entry {
	std::uintptr_t start_addr;
	std::uint32_t block_size;
	std::uint32_t flags;
	std::uint32_t epochs;
};

struct Metadata {
	std::uint16_t magic;
	std::uint8_t padding[6];
	Entry *entry;
};

struct AllocationTable {
	Entry *entries;
	std::uint32_t count;
	std::uint32_t capacity;
};

enum ErrorCode {
	NO_ERROR = 0,
	ERROR_OUT_OF_BOUNDS = 1 << 0,
	ERROR_FREED_MEMORY = 1 << 1,
	ERROR_INVALID_POINTER = 1 << 2
};
}

extern "C" cudaError_t
set_device_table_pointer(safecuda::memory::AllocationTable *ptr);

extern "C" __device__ bool FREED_MEM_DEV;
extern "C" __device__ void __bounds_check_safecuda(void *ptr);
extern "C" __device__ void __bounds_check_safecuda_no_trap(void *ptr);

#endif
