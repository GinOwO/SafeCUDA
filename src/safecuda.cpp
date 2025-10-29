/**
 * @file safecuda.cpp
 * @brief SafeCUDA expose for real CUDA functions
 * 
 * This file contains the expose for real CUDA functions
 * 
 * @author Navin <navinkumar.ao2022@vitstudent.ac.in>
 * @date 2025-07-05
 * @version 1.0.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 * 
 * Change Log:
 * - 2025-10-30: Fixed a potential infinite recursion, removed epoch, decided to
 *		keep magic word in as it's a small cost in exchange for not doing
 *		a scan on the CPU side during cudaFree
 * - 2025-10-29: Moved a lot of stuff, manually passing d_table as arg to 2
 *		param kernels now, ptx side is generic for n param kernels but
 *		not sure of how to make it for n length in host side
 * - 2025-10-23: Reworked table and merged files, added cudaMalloc,
 *		cudaDeviceSynchronize, cudaGetLastError. Added exceptions for
 *		cudaDeviceSynchronize and cudaGetLastError
 * - 2025-10-22: Removed redundancies and fixed styling
 * - 2025-10-18: Intercepted cudaMallocManaged and cudaFree.
 * - 2025-07-05: Initial implementation
 */

#include "safecuda.h"

#include "safecache.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>

static safecuda::memory::AllocationTable *h_table = nullptr;
static safecuda::memory::AllocationTable *d_table_ptr = nullptr;

namespace safecuda
{
constexpr uint32_t TABLE_ENTRIES = 1024;

cudaMalloc_t real_cudaMalloc = nullptr;
cudaMallocManaged_t real_cudaMallocManaged = nullptr;
cudaFree_t real_cudaFree = nullptr;
cudaDeviceSynchronize_t real_cudaDeviceSynchronize = nullptr;
cudaGetLastError_t real_cudaGetLastError = nullptr;
cudaLaunchKernel_t real_cudaLaunchKernel = nullptr;

static void sync_table_to_device()
{
	constexpr size_t entry_bytes = TABLE_ENTRIES * sizeof(memory::Entry);
	constexpr size_t table_bytes =
		sizeof(memory::AllocationTable) + entry_bytes;
	cudaMemcpy(d_table_ptr, h_table, table_bytes, cudaMemcpyHostToDevice);
}

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
	real_cudaGetLastError = reinterpret_cast<cudaGetLastError_t>(
		dlsym(RTLD_NEXT, "cudaGetLastError"));
	real_cudaLaunchKernel = reinterpret_cast<cudaLaunchKernel_t>(
		dlsym(RTLD_NEXT, "cudaLaunchKernel"));

	constexpr size_t entry_bytes = TABLE_ENTRIES * sizeof(memory::Entry);
	constexpr size_t table_bytes =
		sizeof(memory::AllocationTable) + entry_bytes;
	cudaHostAlloc(&h_table, table_bytes, cudaHostAllocMapped);

	if (!h_table) {
		printf("Failed to allocate to h_table\n");
		exit(1);
	}

	h_table->entries = reinterpret_cast<memory::Entry *>(h_table + 1);
	h_table->count = 1;
	h_table->capacity = TABLE_ENTRIES;
	std::memset(h_table->entries, 0, entry_bytes);
	void *ptr = nullptr;
	real_cudaMalloc(&ptr, table_bytes);
	d_table_ptr = static_cast<memory::AllocationTable *>(ptr);
	cudaMemcpy(d_table_ptr, h_table, table_bytes, cudaMemcpyHostToDevice);

	real_cudaDeviceSynchronize();
}

void shutdown_safecuda()
{
	if (d_table_ptr)
		cudaFree(d_table_ptr);
	if (h_table)
		cudaFreeHost(h_table);
	d_table_ptr = nullptr;
	h_table = nullptr;
}

void check_and_report_errors()
{
	if (h_table->entries[0].flags == memory::NO_ERROR)
		return;
	const std::uintptr_t addr = h_table->entries[0].start_addr;
	std::uint32_t code = h_table->entries[0].flags;

	if (code & memory::ERROR_OUT_OF_BOUNDS) {
		char addr_buf[32];
		std::snprintf(addr_buf, sizeof(addr_buf), "0x%lx", addr);
		const std::string error_msg =
			std::string("[SafeCUDA] Out-of-bounds access at ") +
			addr_buf + " (code=0x" + std::to_string(code) + ")";

		std::fprintf(stderr, "%s\n", error_msg.c_str());
		throw std::runtime_error(error_msg);
	}

	if (code & memory::ERROR_FREED_MEMORY) {
		char addr_buf[32];
		std::snprintf(addr_buf, sizeof(addr_buf), "0x%lx", addr);
		const std::string error_msg =
			std::string("[SafeCUDA] Use-after-free at ") +
			addr_buf + " (code=0x" + std::to_string(code) + ")";

		std::fprintf(stderr, "%s\n", error_msg.c_str());
		throw std::runtime_error(error_msg);
	}

	if (code & memory::ERROR_INVALID_POINTER) {
		char addr_buf[32];
		std::snprintf(addr_buf, sizeof(addr_buf), "0x%lx", addr);
		const std::string error_msg =
			std::string("[SafeCUDA] Invalid pointer access at ") +
			addr_buf + " (code=0x" + std::to_string(code) + ")";

		std::fprintf(stderr, "%s\n", error_msg.c_str());
		throw std::runtime_error(error_msg);
	}
	h_table->entries[0].flags = 0;
	h_table->entries[0].start_addr = 0;
}

}

extern "C" cudaError_t cudaMalloc(void **devPtr, const std::size_t size)
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
	if (h_table->count >= safecuda::TABLE_ENTRIES) {
		safecuda::real_cudaFree(base);
		std::fprintf(stderr, "[SafeCUDA] Allocation table full\n");
		return cudaErrorMemoryAllocation;
	}
	void *user_ptr = static_cast<char *>(base) + 16;
	safecuda::memory::Entry *entry = &h_table->entries[h_table->count++];
	entry->start_addr = reinterpret_cast<std::uintptr_t>(user_ptr);
	entry->block_size = size;
	entry->flags = safecuda::memory::NO_ERROR;
	const safecuda::memory::Metadata meta = {0x5AFE, {0}, entry};
	cudaMemcpy(base, &meta, 16, cudaMemcpyHostToDevice);
	safecuda::sync_table_to_device();
	*devPtr = user_ptr;
	return cudaSuccess;
}

extern "C" cudaError_t cudaMallocManaged(void **devPtr, const std::size_t size,
					 const unsigned int flags)
{
	if (!safecuda::real_cudaMallocManaged)
		safecuda::init_safecuda();
	void *base = nullptr;
	const cudaError_t err =
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
	entry->flags = safecuda::memory::NO_ERROR;
	const safecuda::memory::Metadata meta = {0x5AFE, {0}, entry};
	cudaMemcpy(base, &meta, 16, cudaMemcpyHostToDevice);
	safecuda::sync_table_to_device();
	*devPtr = user_ptr;
	return cudaSuccess;
}

extern "C" cudaError_t cudaFree(void *devPtr)
{
	if (!devPtr)
		return cudaSuccess;

	void *base = static_cast<std::int8_t *>(devPtr) - 16;
	safecuda::memory::Metadata meta{};
	cudaMemcpy(&meta, base, 16, cudaMemcpyDeviceToHost);
	if (meta.magic == 0x5AFE && meta.entry)
		meta.entry->flags |= safecuda::memory::ERROR_FREED_MEMORY;
	safecuda::sync_table_to_device();
	return safecuda::real_cudaFree(base);
}

extern "C" cudaError_t cudaDeviceSynchronize()
{
	if (!safecuda::real_cudaDeviceSynchronize)
		safecuda::init_safecuda();
	const cudaError_t err = safecuda::real_cudaDeviceSynchronize();
	safecuda::check_and_report_errors();
	return err;
}

extern "C" cudaError_t cudaGetLastError()
{
	if (!safecuda::real_cudaGetLastError)
		safecuda::init_safecuda();
	safecuda::check_and_report_errors();
	return safecuda::real_cudaGetLastError();
}

extern "C" cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim,
					dim3 blockDim, void **args,
					size_t sharedMem, cudaStream_t stream)
{
	if (!safecuda::real_cudaLaunchKernel)
		safecuda::init_safecuda();

	constexpr int numParams = 2;
	constexpr size_t size = (numParams + 2) * sizeof(void *);

	void **newParams = static_cast<void **>(alloca(size));
	newParams[0] = &d_table_ptr;
	newParams[1] = args[0];
	newParams[2] = args[1];
	newParams[numParams] = nullptr;

	return safecuda::real_cudaLaunchKernel(func, gridDim, blockDim,
					       newParams, sharedMem, stream);
}
