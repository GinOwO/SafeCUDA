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
	safecuda::memtable::Entry* entry = nullptr;

	cudaError_t entry_result = safecuda::real_cudaMallocManaged(reinterpret_cast<void**>(&entry), sizeof(safecuda::memtable::Entry), flags);
	if (entry_result == cudaSuccess)
		std::cerr << "[SafeCUDA] Entry Allocation succeeded.\n";
	else
		std::cerr << "[SafeCUDA] Entry Allocation failed: " << cudaGetErrorString(entry_result) << "\n";

	cudaError_t result = safecuda::real_cudaMallocManaged(&basePtr, newSize, flags);

	*devPtr = safecuda::memtable::init_header(basePtr, size, 2, 2, entry);
	safecuda::memtable::validate_header(*devPtr);

	safecuda::dynamic_cache->push(reinterpret_cast<uintptr_t>(*devPtr), static_cast<size_t>(newSize));
	bool res = safecuda::dynamic_cache->search(reinterpret_cast<uintptr_t>(*devPtr));
	if(res == true)
		std::cerr << "[SafeCUDA] Found " << reinterpret_cast<uintptr_t>(*devPtr) << " entry in memory.\n";
	else
		std::cerr << "[SafeCUDA] entry not found in memory.\n";

	if (result == cudaSuccess)
		std::cerr << "[SafeCUDA] Allocation succeeded.\n";
	else
		std::cerr << "[SafeCUDA] Allocation failed: " << cudaGetErrorString(result) << "\n";

	return result;
}

extern "C" cudaError_t cudaFree(void* devPtr) {

    if (!safecuda::real_cudaFree) {
		std::cerr << "[SafeCUDA] Failed to find original cudaFree: " << dlerror() << std::endl;
    }

	std::cerr << "[SafeCUDA] Intercepted cudaFree" << "\n";
	void* todelete = reinterpret_cast<void*>(reinterpret_cast<std::uint8_t*>(devPtr) - sizeof(safecuda::memtable::Header));

	cudaError_t entry_result = safecuda::real_cudaFree(safecuda::memtable::delete_entry(todelete));
	if (entry_result == cudaSuccess)
		std::cerr << "[SafeCUDA] Entry DeAllocation succeeded.\n";
	else
		std::cerr << "[SafeCUDA] Entry DeAllocation failed: " << cudaGetErrorString(entry_result) << "\n";

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
		new (dynamic_cache) safecuda::cache::DynamicCache(5);

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
