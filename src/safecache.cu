/**
 * @file safecache.cu
 * @brief SafeCUDA fallback cache implementation
 *
 * This file contains the implementation for the dynamic fallback cache
 *
 * @author Anirudh <anirudh.sridhar2022@vitstudent.ac.in>
 * @date 2025-09-22
 * @version 0.0.2
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-09-22: Initial implementation
 * - 2025-10-11: Revised code to be device and host side functions.
 * - 2025-10-15: Revised code to make class vars non-static and redeclare
 *		constructor, destructor etc. Also, some other code changes.
 * - 2025-10-22: Removed redundancies and fixed a [[noreturn]] bug on check_cuda
 */

#include "safecache.cuh"

#include "safecuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>

extern "C" __managed__ void *dynamic_cache = nullptr;

namespace safecuda::cache
{
DynamicCache::DynamicCache(const size_t initial_capacity)
	: d_buf(nullptr)
	, d_size(0)
	, d_capacity(initial_capacity)
{
	cudaError_t result = real_cudaMallocManaged(
		reinterpret_cast<void **>(&d_buf),
		initial_capacity * sizeof(CacheEntry), cudaMemAttachGlobal);
	if (result == cudaSuccess)
		std::cerr << "[SafeCUDA] d_buf Allocation succeeded.\n";
	else
		std::cerr << "[SafeCUDA] d_buf Allocation failed: "
			  << cudaGetErrorString(result) << "\n";
	d_size = 0;
}

DynamicCache::~DynamicCache()
{
	if (d_buf) {
		if (const cudaError_t result = real_cudaFree(d_buf);
		    result == cudaSuccess)
			std::cerr
				<< "[SafeCUDA] d_buf DeAllocation succeeded.\n";
		else
			std::cerr << "[SafeCUDA] d_buf DeAllocation failed: "
				  << cudaGetErrorString(result) << "\n";
	}

	d_buf = nullptr;
	d_size = 0;
	d_capacity = 0;
}

inline void DynamicCache::_check_cuda(const cudaError_t err)
{
	if (err != cudaSuccess) {
		std::fprintf(stderr, "CUDA Error %d: %s\n", err,
			     cudaGetErrorString(err));
		std::exit(EXIT_FAILURE);
	}
}

inline CacheEntry DynamicCache::_init_cache_entry(const std::uintptr_t address,
						  const std::uint32_t size,
						  const std::uint8_t flags,
						  const std::uint32_t epochs)
{
	return CacheEntry(address, size, flags, epochs);
}

void DynamicCache::_extend_cache()
{
	if (d_size < d_capacity)
		return;

	const size_t new_capacity = d_capacity * 2;
	CacheEntry *new_buf = nullptr;
	_check_cuda(real_cudaMallocManaged(reinterpret_cast<void **>(&new_buf),
					   new_capacity * sizeof(CacheEntry),
					   cudaMemAttachGlobal));

	for (size_t i = 0; i < d_size; ++i) {
		new_buf[i] = d_buf[i];
	}

	real_cudaFree(d_buf);
	d_buf = new_buf;
	d_capacity = new_capacity;
}

CacheEntry *DynamicCache::push(const std::uintptr_t address,
			       const std::uint32_t size,
			       const std::uint8_t flags,
			       const std::uint32_t epochs)
{
	if (!d_buf) {
		std::fprintf(stderr, "DynamicCache not initialized\n");
		std::exit(EXIT_FAILURE);
	}

	if (d_size >= d_capacity) {
		_extend_cache();
	}

	d_buf[d_size] = _init_cache_entry(address, size, flags, epochs);
	return &d_buf[d_size++];
}

bool DynamicCache::search(const uintptr_t address) const
{
	for (size_t i = 0; i < d_size; ++i) {
		if (d_buf[i].start_addr == address) {
			return true;
		}
	}

	return false;
}

}
