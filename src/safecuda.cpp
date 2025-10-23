/**
 * @file safecuda.cpp
 * @brief SafeCUDA expose for real CUDA functions
 * 
 * This file contains the expose for real CUDA functions
 * 
 * @author Navin <navinkumar.ao2022@vitstudent.ac.in>
 * @date 2025-07-05
 * @version 0.1.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 * 
 * Change Log:
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

safecuda::memory::AllocationTable *h_table = nullptr;

extern __constant__ safecuda::memory::AllocationTable *d_table;
extern __device__ bool FREED_MEM_DEV;

namespace safecuda
{
constexpr std::uint32_t MIN_TABLE_SIZE = 16;
constexpr std::uint32_t DEFAULT_TABLE_SIZE = 1024;
constexpr std::uint32_t MAX_TABLE_SIZE = 65536;

cudaMalloc_t real_cudaMalloc = nullptr;
cudaMallocManaged_t real_cudaMallocManaged = nullptr;
cudaFree_t real_cudaFree = nullptr;
cudaDeviceSynchronize_t real_cudaDeviceSynchronize = nullptr;
cudaGetLastError_t real_cudaGetLastError = nullptr;

static std::uint32_t get_table_size_from_env()
{
	const char *env_size = std::getenv("SAFECUDA_TABLE_SIZE");
	if (!env_size)
		return DEFAULT_TABLE_SIZE;

	char *end_ptr = nullptr;
	const long size = std::strtol(env_size, &end_ptr, 10);

	if (end_ptr == env_size || *end_ptr != '\0')
		return DEFAULT_TABLE_SIZE;
	if (size < MIN_TABLE_SIZE)
		return MIN_TABLE_SIZE;

	if (size > MAX_TABLE_SIZE)
		return MAX_TABLE_SIZE;

	return static_cast<std::uint32_t>(size);
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

	const std::uint32_t table_size = get_table_size_from_env();

	constexpr size_t table_bytes = sizeof(memory::AllocationTable);
	const size_t entries_bytes = table_size * sizeof(memory::Entry);

	cudaHostAlloc(&h_table, table_bytes, cudaHostAllocMapped);
	cudaHostAlloc(&h_table->entries, entries_bytes, cudaHostAllocMapped);

	h_table->count = 1;
	h_table->capacity = table_size;
	std::memset(&h_table->entries[0], 0, sizeof(memory::Entry));

	memory::AllocationTable *d_table_ptr = nullptr;
	cudaHostGetDevicePointer(&d_table_ptr, h_table, 0);

	memory::Entry *d_entries_ptr = nullptr;
	cudaHostGetDevicePointer(&d_entries_ptr, h_table->entries, 0);

	cudaMemcpy(&d_table_ptr->entries, &d_entries_ptr,
		   sizeof(memory::Entry *), cudaMemcpyHostToDevice);

	const bool freed = std::getenv("SAFECUDA_THROW_FREED_MEMORY");
	cudaMemcpyToSymbol(&FREED_MEM_DEV, &freed, sizeof(bool));

	cudaError_t err = set_device_table_pointer(d_table_ptr);
	if (err != cudaSuccess) {
		fprintf(stderr,
			"[SafeCUDA] Failed to set device table pointer: %s\n",
			cudaGetErrorString(err));
	}
}

void shutdown_safecuda()
{
	if (h_table) {
		if (h_table->entries)
			cudaFreeHost(h_table->entries);
		cudaFreeHost(h_table);
	}
	h_table = nullptr;
}

static bool throw_exception_on(const memory::ErrorCode err)
{
	if (!err)
		return false;

	return ((err & memory::ERROR_OUT_OF_BOUNDS) &&
		std::getenv("SAFECUDA_THROW_OOB")) ||
	       ((err & memory::ERROR_INVALID_POINTER) &&
		std::getenv("SAFECUDA_THROW_INVALID_POINTER")) ||
	       ((err & memory::ERROR_FREED_MEMORY) &&
		std::getenv("SAFECUDA_THROW_FREED_MEMORY"));
}

void check_and_report_errors()
{
	if (!h_table || h_table->entries[0].flags == memory::NO_ERROR)
		return;
	const std::uintptr_t addr = h_table->entries[0].start_addr;
	std::uint32_t code = h_table->entries[0].flags;
	const auto err = static_cast<memory::ErrorCode>(code);
	if (code & memory::ERROR_OUT_OF_BOUNDS) {
		char addr_buf[32];
		std::snprintf(addr_buf, sizeof(addr_buf), "0x%lx", addr);
		const std::string error_msg =
			std::string("[SafeCUDA] Out-of-bounds access at ") +
			addr_buf + " (code=0x" + std::to_string(code) + ")";

		std::fprintf(stderr, "%s\n", error_msg.c_str());
		if (throw_exception_on(err))
			throw std::runtime_error(error_msg);
	}

	if (code & memory::ERROR_FREED_MEMORY) {
		char addr_buf[32];
		std::snprintf(addr_buf, sizeof(addr_buf), "0x%lx", addr);
		const std::string error_msg =
			std::string("[SafeCUDA] Use-after-free at ") +
			addr_buf + " (code=0x" + std::to_string(code) + ")";

		std::fprintf(stderr, "%s\n", error_msg.c_str());
		if (throw_exception_on(err))
			throw std::runtime_error(error_msg);
	}

	if (code & memory::ERROR_INVALID_POINTER) {
		char addr_buf[32];
		std::snprintf(addr_buf, sizeof(addr_buf), "0x%lx", addr);
		const std::string error_msg =
			std::string("[SafeCUDA] Invalid pointer access at ") +
			addr_buf + " (code=0x" + std::to_string(code) + ")";

		std::fprintf(stderr, "%s\n", error_msg.c_str());
		if (throw_exception_on(err))
			throw std::runtime_error(error_msg);
	}
	h_table->entries[0].flags = 0;
	h_table->entries[0].start_addr = 0;
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
	if (h_table->count >= safecuda::get_table_size_from_env()) {
		safecuda::real_cudaFree(base);
		std::fprintf(stderr, "[SafeCUDA] Allocation table full\n");
		return cudaErrorMemoryAllocation;
	}
	void *user_ptr = static_cast<char *>(base) + 16;
	safecuda::memory::Entry *entry = &h_table->entries[h_table->count++];
	entry->start_addr = reinterpret_cast<std::uintptr_t>(user_ptr);
	entry->block_size = size;
	entry->flags = safecuda::memory::NO_ERROR;
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
	entry->flags = safecuda::memory::NO_ERROR;
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
	if (meta.magic == 0x5AFE && meta.entry)
		meta.entry->flags |= safecuda::memory::ERROR_FREED_MEMORY;

	return safecuda::real_cudaFree(base);
}

extern "C" cudaError_t cudaDeviceSynchronize()
{
	const cudaError_t err = safecuda::real_cudaDeviceSynchronize();
	safecuda::check_and_report_errors();
	return err;
}

extern "C" cudaError_t cudaGetLastError()
{
	safecuda::check_and_report_errors();
	return safecuda::real_cudaGetLastError();
}
