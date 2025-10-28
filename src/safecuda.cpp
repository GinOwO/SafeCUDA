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
#include <iostream>
#include <stdexcept>

#include "safecache.cuh"

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
cudaConfigureCall_t real_cudaConfigureCall = nullptr;
cudaSetupArgument_t real_cudaSetupArgument = nullptr;
cudaLaunch_t real_cudaLaunch = nullptr;
cuLaunchKernel_t real_cuLaunchKernel = nullptr;
cudaLaunchKernel_t real_cudaLaunchKernel = nullptr;

static void sync_table_to_device()
{
	size_t entry_bytes = TABLE_ENTRIES * sizeof(memory::Entry);
	size_t table_bytes = sizeof(memory::AllocationTable) + entry_bytes;
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
	real_cudaConfigureCall = reinterpret_cast<cudaConfigureCall_t>(
		dlsym(RTLD_NEXT, "cudaConfigureCall"));
	real_cudaSetupArgument = reinterpret_cast<cudaSetupArgument_t>(
		dlsym(RTLD_NEXT, "cudaSetupArgument"));
	real_cudaLaunch =
		reinterpret_cast<cudaLaunch_t>(dlsym(RTLD_NEXT, "cudaLaunch"));
	real_cuLaunchKernel = reinterpret_cast<cuLaunchKernel_t>(
		dlsym(RTLD_NEXT, "cuLaunchKernel"));
	real_cudaLaunchKernel = reinterpret_cast<cudaLaunchKernel_t>(
		dlsym(RTLD_NEXT, "cudaLaunchKernel"));

	size_t entry_bytes = TABLE_ENTRIES * sizeof(memory::Entry);
	size_t table_bytes = sizeof(memory::AllocationTable) + entry_bytes;
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

	cudaDeviceSynchronize();
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
	entry->flags = safecuda::memory::NO_ERROR;
	entry->epochs = 0;
	safecuda::memory::Metadata meta = {0x5AFE, {0}, entry};
	cudaMemcpy(base, &meta, 16, cudaMemcpyHostToDevice);
	safecuda::sync_table_to_device();
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
	safecuda::sync_table_to_device();
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

thread_local size_t tls_arg_end = 0;

extern "C" cudaError_t cudaConfigureCall(dim3 grid, dim3 block,
					 size_t sharedMem, cudaStream_t stream)
{
	if (!safecuda::real_cudaConfigureCall)
		safecuda::init_safecuda();
	puts("in conf call");
	tls_arg_end = 0;
	return safecuda::real_cudaConfigureCall(grid, block, sharedMem, stream);
}

extern "C" cudaError_t cudaSetupArgument(const void *arg, size_t size,
					 size_t offset)
{
	if (!safecuda::real_cudaConfigureCall)
		safecuda::init_safecuda();
	puts("in setup arg");
	cudaError_t r = safecuda::real_cudaSetupArgument(arg, size, offset);
	// update tracked end
	size_t end = offset + size;
	if (end > tls_arg_end)
		tls_arg_end = end;
	return r;
}

extern "C" cudaError_t cudaLaunch(const void *func)
{
	if (!safecuda::real_cudaConfigureCall)
		safecuda::init_safecuda();

	if (!safecuda::real_cudaLaunch)
		return cudaErrorUnknown;

	puts("in launch");
	// append our pointer at tls_arg_end
	if (d_table_ptr) {
		// call the real setup arg to place our pointer at offset tls_arg_end
		// we pass the pointer value (device pointer)
		void *val =
			&d_table_ptr; // Note: cudaSetupArgument expects host pointer to data
		safecuda::real_cudaSetupArgument(&val, sizeof(void *),
						 tls_arg_end);
		// (no need to update tls_arg_end further)
	}
	// Now call the real launch
	return safecuda::real_cudaLaunch(func);
}

extern "C" CUresult cuLaunchKernel(CUfunction f, unsigned int gx,
				   unsigned int gy, unsigned int gz,
				   unsigned int bx, unsigned int by,
				   unsigned int bz, unsigned int shmem,
				   CUstream stream, void **kernelParams,
				   void **extra)
{
	if (!safecuda::real_cuLaunchKernel)
		safecuda::init_safecuda();
	void *orig0 = kernelParams[0];
	void *orig1 = kernelParams[1];

	// allocate new param array
	void **newParams = (void **)malloc(3 * sizeof(void *));
	newParams[0] = orig0;
	newParams[1] = orig1;
	newParams[2] = &d_table_ptr;

	// launch using new array
	auto res = safecuda::real_cuLaunchKernel(
		f, gx, gy, gz, bx, by, bz, shmem, stream, newParams, extra);

	free(newParams);

	return res;
}

extern "C" cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim,
					dim3 blockDim, void **args,
					size_t sharedMem, cudaStream_t stream)
{
	if (!safecuda::real_cudaLaunchKernel)
		safecuda::init_safecuda();
	printf(""); // I do not know why it happens but if you remove it it breaks and doesnt work
	int count = 0;
	while (args[count])
		count++;
	void **newArgs = (void **)alloca((count + 1) * sizeof(void *));
	newArgs[2] = &d_table_ptr;
	for (int i = 3; i < count; ++i)
		newArgs[i + 1] = args[i];

	return safecuda::real_cudaLaunchKernel(func, gridDim, blockDim, newArgs,
					       sharedMem, stream);
}
