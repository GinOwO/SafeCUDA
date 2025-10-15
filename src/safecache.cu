/**
 * @file safecache.cu
 * @brief SafeCUDA fallback cache implementation
 *
 * This file contains the implementation for the dynamic fallback cache
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

#include "safecache.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>

extern "C" __managed__ void *dynamic_cache = nullptr;

namespace safecuda::cache
{
DynamicCache::DynamicCache(size_t initial_capacity)
	: d_buf(nullptr), d_size(0), d_capacity(initial_capacity)
{

	if (dynamic_cache) {
		d_buf = static_cast<CacheEntry *>(dynamic_cache);
		return;
	}

	_check_cuda(cudaMallocManaged(&dynamic_cache, initial_capacity * sizeof(CacheEntry)));
	d_buf = static_cast<CacheEntry *>(dynamic_cache);
	d_size = 0;
}

DynamicCache::~DynamicCache()
{
	if (d_buf) cudaFree(d_buf);
	d_buf = nullptr;
	d_size = 0;
	d_capacity = 0;
}

inline void DynamicCache::_check_cuda(const cudaError_t err)
{
	if (err != cudaSuccess) {
		std::fprintf(stderr, "CUDA Error %d: %s\n", err,
			     cudaGetErrorString(static_cast<cudaError_t>(err)));
		std::exit(EXIT_FAILURE);
	}
}

inline CacheEntry DynamicCache::_init_cache_entry(const uintptr_t address,
					   const size_t size)
{
	return CacheEntry(address, size);
}

void DynamicCache::_extend_cache()
{
	if (d_size < d_capacity)
		return;

	const size_t new_capacity = d_capacity * 2;
	CacheEntry *new_buf = nullptr;
	_check_cuda(
		cudaMallocManaged(&new_buf, new_capacity * sizeof(CacheEntry)));

	for (size_t i = 0; i < d_size; ++i) {
		new_buf[i] = d_buf[i];
	}

	cudaFree(d_buf);
	d_buf = new_buf;
	d_capacity = new_capacity;
}

void DynamicCache::push(const uintptr_t address, const size_t size)
{
	if (!d_buf) {
		std::fprintf(stderr, "DynamicCache not initialized\n");
		std::exit(EXIT_FAILURE);
	}

	if (d_size >= d_capacity) {
		_extend_cache();
	}

	d_buf[d_size++] = _init_cache_entry(address, size);
}

bool DynamicCache::search(const uintptr_t address) const
{
	for (size_t i = 0; i < d_size; ++i) {
		if (d_buf[i].start_address == address) {
			return true;
		}
	}

	return false;
}

}