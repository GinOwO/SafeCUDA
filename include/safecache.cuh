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
	uintptr_t start_address;
	size_t size;

	__host__ __device__ explicit CacheEntry(const uintptr_t address = 0,
						const size_t size = 0)
		: start_address(address)
		, size(size)
	{
	}
};

class DynamicCache {
    private:
	CacheEntry *d_buf;
	size_t d_size;
	size_t d_capacity;

    public:
	__host__ __device__ explicit DynamicCache(size_t initial_capacity);

	__host__ __device__ ~DynamicCache();

	__host__ __device__ DynamicCache(const DynamicCache &) = delete;

	__host__ __device__ DynamicCache &
	operator=(const DynamicCache &) = delete;

	__host__ __device__ DynamicCache(DynamicCache &&) = delete;

	__host__ __device__ DynamicCache &operator=(DynamicCache &&) = delete;

	__host__ [[noreturn]] static void _check_cuda(cudaError_t err);
	__host__ [[nodiscard]] static CacheEntry
	_init_cache_entry(uintptr_t address, size_t size);
	__host__ void _extend_cache();

	__host__ void push(uintptr_t address, size_t size);

	__host__ __device__ [[nodiscard]] bool search(uintptr_t address) const;
};

}

#endif
