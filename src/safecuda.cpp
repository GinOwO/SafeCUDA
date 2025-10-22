/**
 * @file safecuda.cpp
 * @brief SafeCUDA expose for real CUDA functions
 * 
 * This file contains the expose for real CUDA functions
 * 
 * @author Navin <navinkumar.ao2022@vitstudent.ac.in>
 * @date 2025-07-05
 * @version 0.0.2
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 * 
 * Change Log:
 * - 2025-07-05: Initial implementation
 * - 2025-10-18: Intercepted cudaMallocManaged and cudaFree.
 * - 2025-10-22: Removed redundancies and fixed styling
 */
#include "safecuda.h"

#include "memtable.h"
#include "safecache.cuh"

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <iostream>

extern "C" cudaError_t cudaMallocManaged(void **dev_ptr, const size_t size,
					 const unsigned int flags)
{
	if (!safecuda::real_cudaMallocManaged) {
		std::cerr
			<< "[SafeCUDA] Failed to find original cudaMallocManaged: "
			<< dlerror() << std::endl;
	}

	std::cerr << "[SafeCUDA] Intercepted cudaMallocManaged(" << size
		  << " bytes)" << std::endl;

	const size_t new_size = size + sizeof(safecuda::memtable::Header);
	void *base_ptr = nullptr;

	const cudaError_t result =
		safecuda::real_cudaMallocManaged(&base_ptr, new_size, flags);

	if (result == cudaSuccess)
		std::cerr << "[SafeCUDA] Allocation succeeded.\n";
	else
		std::cerr << "[SafeCUDA] Allocation failed: "
			  << cudaGetErrorString(result) << "\n";

	*dev_ptr = safecuda::memtable::init_header(base_ptr);

	safecuda::cache::CacheEntry *entry_ptr = safecuda::dynamic_cache->push(
		reinterpret_cast<std::uintptr_t>(*dev_ptr),
		static_cast<std::uint32_t>(size), 0, 0);

	const bool res = safecuda::dynamic_cache->search(
		reinterpret_cast<uintptr_t>(*dev_ptr));
	if (res == true)
		std::cerr << "[SafeCUDA] Found "
			  << reinterpret_cast<uintptr_t>(*dev_ptr)
			  << " entry in memory.\n";
	else
		std::cerr << "[SafeCUDA] entry not found in memory.\n";

	auto *header = static_cast<safecuda::memtable::Header *>(base_ptr);
	header->entry = entry_ptr;

	std::cerr << "entry-start_addr - " << header->entry->start_addr
		  << std::endl;
	std::cerr << "entry-size - " << header->entry->block_size << std::endl;
	std::cerr << "magic_word - " << header->magic_word << std::endl;

	return result;
}

extern "C" cudaError_t cudaFree(void *dev_ptr)
{
	if (!safecuda::real_cudaFree) {
		std::cerr << "[SafeCUDA] Failed to find original cudaFree: "
			  << dlerror() << std::endl;
	}

	std::cerr << "[SafeCUDA] Intercepted cudaFree" << "\n";
	void *to_delete = static_cast<std::uint8_t *>(dev_ptr) -
			  sizeof(safecuda::memtable::Header);
	safecuda::memtable::delete_entry(to_delete);

	const cudaError_t result = safecuda::real_cudaFree(to_delete);
	if (result == cudaSuccess)
		std::cerr << "[SafeCUDA] DeAllocation succeeded.\n";
	else
		std::cerr << "[SafeCUDA] DeAllocation failed: "
			  << cudaGetErrorString(result) << "\n";

	return result;
}

namespace safecuda
{
cudaMallocManaged_t real_cudaMallocManaged = nullptr;
cudaFree_t real_cudaFree = nullptr;
__managed__ cache::DynamicCache *dynamic_cache = nullptr;

void init_symbols()
{
	real_cudaMallocManaged = reinterpret_cast<cudaMallocManaged_t>(
		dlsym(RTLD_NEXT, "cudaMallocManaged"));
	real_cudaFree =
		reinterpret_cast<cudaFree_t>(dlsym(RTLD_NEXT, "cudaFree"));

	if (!real_cudaMallocManaged || !real_cudaFree) {
		std::cerr << "[SafeCUDA] Failed to resolve CUDA symbols.\n";
		return;
	}

	const cudaError_t result = real_cudaMallocManaged(
		reinterpret_cast<void **>(&dynamic_cache),
		sizeof(cache::DynamicCache), cudaMemAttachGlobal);
	if (result == cudaSuccess)
		std::cerr << "[SafeCUDA] d_cache Allocation succeeded.\n";
	else
		std::cerr << "[SafeCUDA] d_cache DeAllocation failed: "
			  << cudaGetErrorString(result) << "\n";
	new (dynamic_cache) cache::DynamicCache(1024);
}

void shutdown()
{
	dynamic_cache->~DynamicCache();
	if (const cudaError_t result = real_cudaFree(dynamic_cache);
	    result == cudaSuccess)
		std::cerr << "[SafeCUDA] d_cache DeAllocation succeeded.\n";
	else
		std::cerr << "[SafeCUDA] d_cache DeAllocation failed: "
			  << cudaGetErrorString(result) << "\n";

	dynamic_cache = nullptr;
}
}
