/**
 * @file safecuda.cpp
 * @brief SafeCUDA host-side implementation (placeholder)
 * 
 * This file will contain the main SafeCUDA host-side implementation
 * including memory allocation interception and metadata management.
 * 
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-07-05
 * @version 0.0.1
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 * 
 * Change Log:
 * - 2025-07-05: Initial implementation
 * - 2025-10-18: Intercepted cudaMallocManaged and cudaFree.
 */
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <iostream>
#include <mutex>

#include "safecuda.h"
#include "memtable.h"
#include "safecache.cuh"

extern "C" cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {

    if (!safecuda::real_cudaMallocManaged) {
		std::cerr << "[SafeCUDA] Failed to find original cudaMallocManaged: " << dlerror() << std::endl;
    }

	std::cerr << "[SafeCUDA] Intercepted cudaMallocManaged(" << size << " bytes)" << std::endl;

	size_t newSize = size + sizeof(safecuda::memtable::Header);
	void *basePtr = nullptr;

	cudaError_t result = safecuda::real_cudaMallocManaged(&basePtr, newSize, flags);

	if (result == cudaSuccess)
		std::cerr << "[SafeCUDA] Allocation succeeded.\n";
	else
		std::cerr << "[SafeCUDA] Allocation failed: " << cudaGetErrorString(result) << "\n";

	*devPtr = safecuda::memtable::init_header(basePtr);

	safecuda::cache::CacheEntry* entry_ptr = safecuda::dynamic_cache->push(reinterpret_cast<std::uintptr_t>(*devPtr), static_cast<std::uint32_t>(size), 0, 0);

	bool res = safecuda::dynamic_cache->search(reinterpret_cast<uintptr_t>(*devPtr));
	if(res == true)
		std::cerr << "[SafeCUDA] Found " << reinterpret_cast<uintptr_t>(*devPtr) << " entry in memory.\n";
	else
		std::cerr << "[SafeCUDA] entry not found in memory.\n";

	safecuda::memtable::Header *header = reinterpret_cast<safecuda::memtable::Header*>(basePtr);
	header->entry = entry_ptr;

	std::cerr << "entry-start_addr - " << reinterpret_cast<uintptr_t>(header->entry->start_addr) << std::endl;
	std::cerr << "entry-size - " << header->entry->block_size << std::endl;
	std::cerr << "magic_word - " << header->magic_word << std::endl;

	return result;
}

extern "C" cudaError_t cudaFree(void* devPtr) {

    if (!safecuda::real_cudaFree) {
		std::cerr << "[SafeCUDA] Failed to find original cudaFree: " << dlerror() << std::endl;
    }

	std::cerr << "[SafeCUDA] Intercepted cudaFree" << "\n";
	void* todelete = reinterpret_cast<void*>(reinterpret_cast<std::uint8_t*>(devPtr) - sizeof(safecuda::memtable::Header));
	safecuda::memtable::delete_entry(todelete);

	cudaError_t result = safecuda::real_cudaFree(todelete);
	if (result == cudaSuccess)
		std::cerr << "[SafeCUDA] DeAllocation succeeded.\n";
	else
		std::cerr << "[SafeCUDA] DeAllocation failed: " << cudaGetErrorString(result) << "\n";

	return result;
}

namespace safecuda{
	cudaMallocManaged_t real_cudaMallocManaged = nullptr;
	cudaFree_t real_cudaFree = nullptr;
	__managed__ safecuda::cache::DynamicCache* dynamic_cache = nullptr;

	void init_symbols() {
		real_cudaMallocManaged = (cudaMallocManaged_t)dlsym(RTLD_NEXT, "cudaMallocManaged");
    	real_cudaFree = (cudaFree_t)dlsym(RTLD_NEXT, "cudaFree");

    	if (!real_cudaMallocManaged || !real_cudaFree){
			std::cerr << "[SafeCUDA] Failed to resolve CUDA symbols.\n";
			return;
		}

		cudaError_t result = real_cudaMallocManaged(reinterpret_cast<void**>(&dynamic_cache), sizeof(safecuda::cache::DynamicCache), cudaMemAttachGlobal);
		if (result == cudaSuccess)
			std::cerr << "[SafeCUDA] d_cache Allocation succeeded.\n";
		else
			std::cerr << "[SafeCUDA] d_cache DeAllocation failed: " << cudaGetErrorString(result) << "\n";
		new (dynamic_cache) safecuda::cache::DynamicCache(1024);

		return;
	}

	void shutdown() {
		dynamic_cache->~DynamicCache();
		cudaError_t result = real_cudaFree(dynamic_cache);
		if (result == cudaSuccess)
			std::cerr << "[SafeCUDA] d_cache DeAllocation succeeded.\n";
		else
			std::cerr << "[SafeCUDA] d_cache DeAllocation failed: " << cudaGetErrorString(result) << "\n";

		dynamic_cache = nullptr;

		return;
	}
}
