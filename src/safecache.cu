/**
 * @file safecache.cu
 * @brief SafeCUDA fallback cache implementation
 *
 * This file contains the implementation for the dynamic fallback cache
 *
 * @author Anirudh <anirudh.sridhar2022@vitstudent.ac.in>
 * @date 2025-09-22
 * @version 0.1.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-10-23: Reworked and merged with memtable tb fix incompatibility
 * - 2025-10-22: Removed redundancies and fixed a [[noreturn]] bug on
 * - 2025-10-15: Revised code to make class vars non-static and redeclare
 *		constructor, destructor etc. Also, some other code changes.
 * - 2025-10-11: Revised code to be device and host side functions.
 * - 2025-09-22: Initial implementation
 */

#include "safecache.cuh"

#include <cstdio>

__constant__ safecuda::memory::AllocationTable *d_table = nullptr;
__device__ bool FREED_MEM_DEV = false;

extern "C" cudaError_t
set_device_table_pointer(safecuda::memory::AllocationTable *ptr)
{
	return cudaMemcpyToSymbol(d_table, &ptr,
				  sizeof(safecuda::memory::AllocationTable *));
}

__device__ void __bounds_check_safecuda(void *ptr)
{
	printf("Here\n");
	const auto *meta = reinterpret_cast<safecuda::memory::Metadata *>(
		static_cast<std::int8_t *>(ptr) - 16);

	if (!FREED_MEM_DEV && meta->magic == 0x5AFE) {
		safecuda::memory::Entry *entry = meta->entry;
		if (entry == nullptr)
			goto slow;

		const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
		if (addr >= entry->start_addr &&
		    addr < entry->start_addr + entry->block_size) {
			// If inside range but freed
			if (entry->flags &
			    safecuda::memory::ERROR_FREED_MEMORY) {
				const auto old = atomicOr(
					&d_table->entries[0].flags,
					safecuda::memory::ERROR_FREED_MEMORY);
				if ((old &
				     safecuda::memory::ERROR_FREED_MEMORY) ==
				    0) {
					d_table->entries[0].start_addr = addr;
				}
				__trap();
			}
			// clearly valid
			return;
		}
		// outside range
		const auto old =
			atomicOr(&d_table->entries[0].flags,
				 safecuda::memory::ERROR_OUT_OF_BOUNDS);
		if ((old & safecuda::memory::ERROR_OUT_OF_BOUNDS) == 0) {
			d_table->entries[0].start_addr = addr;
		}
		__trap();

		return;
	}
slow:
	const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
	bool freed = false;
	std::int32_t idx = -1;

	for (std::uint32_t i = 1; i < d_table->count; ++i) {
		safecuda::memory::Entry *entry = &d_table->entries[i];

		if (entry->start_addr <= addr &&
		    addr < entry->start_addr + entry->block_size) {
			// if valid just return
			if (entry->flags == safecuda::memory::NO_ERROR)
				return;
			// we need a deferred mechanism for freed mem here
			if (entry->flags &
			    safecuda::memory::ERROR_FREED_MEMORY) {
				freed = true;
				idx = i;
			}
		}
	}

	if (freed) {
		const auto old = atomicOr(&d_table->entries[0].flags,
					  safecuda::memory::ERROR_FREED_MEMORY);
		if ((old & safecuda::memory::ERROR_FREED_MEMORY) == 0) {
			d_table->entries[0].start_addr = addr;
			d_table->entries[0].block_size = idx;
		}
		__trap();
		return;
	}

	const auto old = atomicOr(&d_table->entries[0].flags,
				  safecuda::memory::ERROR_OUT_OF_BOUNDS);

	if ((old & safecuda::memory::ERROR_OUT_OF_BOUNDS) == 0) {
		d_table->entries[0].start_addr =
			reinterpret_cast<std::uintptr_t>(ptr);
	}

	__trap();
}

__device__ void __bounds_check_safecuda_no_trap(void *ptr)
{
	const auto *meta = reinterpret_cast<safecuda::memory::Metadata *>(
		static_cast<std::int8_t *>(ptr) - 16);

	if (meta->magic == 0x5AFE) {
		safecuda::memory::Entry *entry = meta->entry;

		if (entry == nullptr)
			goto slow;

		const auto addr = reinterpret_cast<std::uintptr_t>(ptr);

		if (addr >= entry->start_addr &&
		    addr < entry->start_addr + entry->block_size) {
			// If inside range but freed
			if (entry->flags &
			    safecuda::memory::ERROR_FREED_MEMORY) {
				const auto old = atomicOr(
					&d_table->entries[0].flags,
					safecuda::memory::ERROR_FREED_MEMORY);
				if ((old &
				     safecuda::memory::ERROR_FREED_MEMORY) ==
				    0) {
					d_table->entries[0].start_addr = addr;
				}
			}
			// clearly valid
			return;
		}
		// outside range
		const auto old =
			atomicOr(&d_table->entries[0].flags,
				 safecuda::memory::ERROR_OUT_OF_BOUNDS);
		if ((old & safecuda::memory::ERROR_OUT_OF_BOUNDS) == 0) {
			d_table->entries[0].start_addr = addr;
		}

		return;
	}
slow:
	const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
	bool freed = false;
	std::int32_t idx = -1;
	for (std::uint32_t i = 1; i < d_table->count; ++i) {
		safecuda::memory::Entry *entry = &d_table->entries[i];

		if (entry->start_addr <= addr &&
		    addr < entry->start_addr + entry->block_size) {
			// if valid just return
			if (entry->flags == safecuda::memory::NO_ERROR)
				return;

			// we need a deferred mechanism for freed mem here
			if (entry->flags &
			    safecuda::memory::ERROR_FREED_MEMORY) {
				freed = true;
				idx = i;
			}
		}
	}

	if (freed) {
		const auto old = atomicOr(&d_table->entries[0].flags,
					  safecuda::memory::ERROR_FREED_MEMORY);
		if ((old & safecuda::memory::ERROR_FREED_MEMORY) == 0) {
			d_table->entries[0].start_addr = addr;
			d_table->entries[0].block_size = idx;
		}

		return;
	}

	const auto old = atomicOr(&d_table->entries[0].flags,
				  safecuda::memory::ERROR_OUT_OF_BOUNDS);

	if ((old & safecuda::memory::ERROR_OUT_OF_BOUNDS) == 0) {
		d_table->entries[0].start_addr =
			reinterpret_cast<std::uintptr_t>(ptr);
	}
}
