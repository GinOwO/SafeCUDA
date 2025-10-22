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
 * - 2025-10-22: Removed redundancies and fixed a [[noreturn]] bug on
 */

#include "safecache.cuh"

__constant__ safecuda::memory::AllocationTable *d_table = nullptr;

extern "C" cudaError_t
set_device_table_pointer(safecuda::memory::AllocationTable *ptr)
{
	return cudaMemcpyToSymbol(d_table, &ptr,
				  sizeof(safecuda::memory::AllocationTable *));
}

__device__ void __bounds_check_safecuda(void *ptr)
{
	const auto *meta = reinterpret_cast<safecuda::memory::Metadata *>(
		static_cast<std::int8_t *>(ptr) - 16);
	if (meta->magic == 0x5AFE) {
		safecuda::memory::Entry *entry = meta->entry;
		const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
		if (entry == nullptr)
			goto slow;
		if (entry->flags == 0) {
			if (atomicCAS(&d_table->entries[0].flags, 0,
				      safecuda::memory::ERROR_FREED_MEMORY) ==
			    0) {
				d_table->entries[0].start_addr = addr;
			}
			__trap();
		}
		if (addr >= entry->start_addr &&
		    addr < entry->start_addr + entry->block_size)
			return;
	}
slow:
	for (std::uint32_t i = 1; i < d_table->count; ++i) {
		safecuda::memory::Entry *entry = &d_table->entries[i];
		if (const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
		    addr >= entry->start_addr &&
		    addr < entry->start_addr + entry->block_size) {
			if (entry->flags == 0) {
				if (atomicCAS(&d_table->entries[0].flags, 0,
					      safecuda::memory::
						      ERROR_FREED_MEMORY) ==
				    0) {
					d_table->entries[0].start_addr = addr;
				}
				__trap();
			}
			return;
		}
	}
	if (atomicCAS(&d_table->entries[0].flags, 0,
		      safecuda::memory::ERROR_INVALID_POINTER) == 0) {
		d_table->entries[0].start_addr =
			reinterpret_cast<std::uintptr_t>(ptr);
	}
	__trap();
}
