/**
 * @file safecache.cuh
 * @brief SafeCUDA fallback cache header file
 *
 * This file will contains the header file for the dynamic fallback cache
 *
 * @author Anirudh <anirudh.sridhar2022@vitstudent.ac.in>
 * @date 2025-09-22
 * @version 0.0.1
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-09-22: Initial implementation
 * - 2025-10-11: Revised code to be device and host side functions.
 * - 2025-10-15: Revised code to make class vars non-static and redeclare
 *		constructor, destructor etc. Also some other code changes.
 */

#ifndef SAFECACHE_H
#define SAFECACHE_H

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

extern "C" __managed__ void *dynamic_cache;

namespace safecuda::cache
{

struct CacheEntry {
	std::uintptr_t start_addr;
	std::uint32_t block_size;
	std::uint8_t flags;
	std::uint32_t epochs;

	__host__ __device__ explicit CacheEntry(const std::uintptr_t start_addr = 0,
					   const std::uint32_t block_size = 0,
					   const std::uint8_t flags = 0,
					   const std::uint32_t epochs = 0)
	    : start_addr(start_addr)
	    , block_size(block_size)
	    , flags(flags)
	    , epochs(epochs)
	{
	}
};

class DynamicCache {
    private:
	CacheEntry *d_buf;
	size_t d_size;
	size_t d_capacity;

    public:
	__host__ explicit DynamicCache(size_t initial_capacity);

	__host__ ~DynamicCache();

	__host__ __device__ DynamicCache(const DynamicCache &) = delete;

	__host__ __device__ DynamicCache &
	operator=(const DynamicCache &) = delete;

	__host__ __device__ DynamicCache(DynamicCache &&) = delete;

	__host__ __device__ DynamicCache &operator=(DynamicCache &&) = delete;

	__host__ [[noreturn]] static void _check_cuda(cudaError_t err);
	__host__ [[nodiscard]] static CacheEntry
	_init_cache_entry(const std::uintptr_t address, const std::uint32_t  size, const std::uint8_t flags, const std::uint32_t epochs);
	__host__ void _extend_cache();

	__host__ CacheEntry* push(const std::uintptr_t address, const std::uint32_t  size, const std::uint8_t flags, const std::uint32_t epochs);

	__host__ __device__ [[nodiscard]] bool search(uintptr_t address) const;
};

}

#endif
