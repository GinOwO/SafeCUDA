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
	static CacheEntry *d_buf;
	static size_t d_size;
	static size_t d_capacity;

    public:
	__host__ static void init(size_t initial_capacity = 16);
	__host__ static void destroy();

	__host__ static void _check_cuda(cudaError_t err);
	__host__ static CacheEntry _init_cache_entry(uintptr_t address,
						     size_t size);
	__host__ static void _extend_cache();

	__host__ static void push(uintptr_t address, size_t size);

	__host__ __device__ static bool search(uintptr_t address);
};

}

#endif
