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

#include "safecache.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>

safecuda::memory::AllocationTable *h_table = nullptr;
extern __constant__ safecuda::memory::AllocationTable *d_table;

namespace safecuda
{
cudaMalloc_t real_cudaMalloc = nullptr;
cudaMallocManaged_t real_cudaMallocManaged = nullptr;
cudaFree_t real_cudaFree = nullptr;
cudaDeviceSynchronize_t real_cudaDeviceSynchronize = nullptr;

void init_safecuda()
{
	real_cudaMalloc =
		reinterpret_cast<cudaMalloc_t>(dlsym(RTLD_NEXT, "cudaMalloc"));
	real_cudaMallocManaged = reinterpret_cast<cudaMallocManaged_t>(
		dlsym(RTLD_NEXT, "cudaMallocManaged"));
	real_cudaFree =
		reinterpret_cast<cudaFree_t>(dlsym(RTLD_NEXT, "cudaFree"));
	real_cudaDeviceSynchronize = reinterpret_cast<cudaDeviceSynchronize_t>(
		dlsym(RTLD_NEXT, "cudaDeviceSynchronize"));
	cudaHostAlloc(&h_table, sizeof(memory::AllocationTable),
		      cudaHostAllocMapped);
	h_table->count = 1;
	h_table->capacity = 1024;
	std::memset(&h_table->entries[0], 0, sizeof(memory::Entry));
	memory::AllocationTable *d_table_ptr = nullptr;
	cudaHostGetDevicePointer(&d_table_ptr, h_table, 0);
	cudaError_t err = set_device_table_pointer(d_table_ptr);
	if (err != cudaSuccess) {
		fprintf(stderr,
			"[SafeCUDA] Failed to set device table pointer: %s\n",
			cudaGetErrorString(err));
	}
}

void shutdown_safecuda()
{
	if (h_table)
		cudaFreeHost(h_table);
	h_table = nullptr;
}
}

extern "C" cudaError_t cudaMalloc(void **dev_ptr, std::size_t size)
{
	if (!safecuda::real_cudaMalloc)
		safecuda::init_safecuda();
	void *base = nullptr;
	cudaError_t err = safecuda::real_cudaMalloc(&base, size + 16);
	if (err != cudaSuccess) {
		std::fprintf(stderr, "[SafeCUDA] cudaMalloc failed: %s\n",
			     cudaGetErrorString(err));
		return err;
	}
	if (h_table->count >= 1024) {
		safecuda::real_cudaFree(base);
		std::fprintf(stderr, "[SafeCUDA] Allocation table full\n");
		return cudaErrorMemoryAllocation;
	}
	void *user_ptr = static_cast<char *>(base) + 16;
	safecuda::memory::Entry *entry = &h_table->entries[h_table->count++];
	entry->start_addr = reinterpret_cast<std::uintptr_t>(user_ptr);
	entry->block_size = size;
	entry->flags = 1;
	entry->epochs = 0;
	safecuda::memory::Metadata meta = {0x5AFE, {0}, entry};
	cudaMemcpy(base, &meta, 16, cudaMemcpyHostToDevice);
	*dev_ptr = user_ptr;
	return cudaSuccess;
}

extern "C" cudaError_t cudaMallocManaged(void **dev_ptr, std::size_t size,
					 unsigned int flags)
{
	if (!safecuda::real_cudaMallocManaged)
		safecuda::init_safecuda();
	void *base = nullptr;
	cudaError_t err =
		safecuda::real_cudaMallocManaged(&base, size + 16, flags);
	if (err != cudaSuccess) {
		std::fprintf(stderr,
			     "[SafeCUDA] cudaMallocManaged failed: %s\n",
			     cudaGetErrorString(err));
		return err;
	}
	if (h_table->count >= 1024) {
		safecuda::real_cudaFree(base);
		std::fprintf(stderr, "[SafeCUDA] Allocation table full\n");
		return cudaErrorMemoryAllocation;
	}
	void *user_ptr = static_cast<std::int8_t *>(base) + 16;
	safecuda::memory::Entry *entry = &h_table->entries[h_table->count++];
	entry->start_addr = reinterpret_cast<std::uintptr_t>(user_ptr);
	entry->block_size = size;
	entry->flags = 1;
	entry->epochs = 0;
	safecuda::memory::Metadata meta = {0x5AFE, {0}, entry};
	cudaMemcpy(base, &meta, 16, cudaMemcpyHostToDevice);
	*dev_ptr = user_ptr;
	return cudaSuccess;
}

extern "C" cudaError_t cudaFree(void *ptr)
{
	if (!ptr)
		return cudaSuccess;
	void *base = static_cast<std::int8_t *>(ptr) - 16;
	safecuda::memory::Metadata meta{};
	cudaMemcpy(&meta, base, 16, cudaMemcpyDeviceToHost);
	if (meta.magic == 0x5AFE && meta.entry) {
		meta.entry->flags = 0;
		meta.entry->start_addr = static_cast<std::uintptr_t>(-1);
		meta.entry->block_size = static_cast<std::uint32_t>(-1);
	}
	return safecuda::real_cudaFree(base);
}

extern "C" cudaError_t cudaDeviceSynchronize()
{
	cudaError_t err = safecuda::real_cudaDeviceSynchronize();
	if (h_table && h_table->entries[0].flags != 0) {
		std::uintptr_t addr = h_table->entries[0].start_addr;
		std::uint32_t code = h_table->entries[0].flags;
		std::fprintf(stderr,
			     "[SafeCUDA] Bounds violation at 0x%lx, code=%u\n",
			     addr, code);
		h_table->entries[0].flags = 0;
		h_table->entries[0].start_addr = 0;
	}
	return err;
}
