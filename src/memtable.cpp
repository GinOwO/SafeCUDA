/**
 * @file memtable.cpp
 * @brief SafeCUDA host-side implementation (placeholder)
 *
 * This File contains the implementation of memtable.cpp
 *
 * @author Navin Kumar <navinkumar.ao2022@vitstudent.ac.in>
 * @date 2025-10-18
 * @version 0.0.1
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-07-05: Initial implementation
 */
#include "memtable.h"
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace safecuda::memtable{

	std::uint32_t* init_header(void *basePtr, size_t block_size, std::uint8_t flags, std::uint32_t epochs, safecuda::memtable::Entry *entry){

		safecuda::memtable::Header *header = reinterpret_cast<safecuda::memtable::Header*>(basePtr);
		header->magic_word = safecuda::memtable::MAGIC_WORD;
		header->memory = reinterpret_cast<uint32_t*>(reinterpret_cast<std::uint8_t*>(header) + sizeof(safecuda::memtable::Header));

		entry->start_addr = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(header->memory));
		entry->block_size = static_cast<std::uint32_t>(block_size);
		entry->flags = flags;
		entry->epochs = epochs;

		header->entry = entry;

		return header->memory;
	}

	bool validate_header(void* basePtr){
		safecuda::memtable::Header *header = reinterpret_cast<safecuda::memtable::Header*>(reinterpret_cast<std::uint8_t*>(basePtr) - sizeof(safecuda::memtable::Header));

		if(header->magic_word == safecuda::memtable::MAGIC_WORD){
			return true;
		}
		return false;
	}

	void* delete_entry(void *basePtr){
		safecuda::memtable::Header *header = reinterpret_cast<safecuda::memtable::Header*>(basePtr);
		void* res = reinterpret_cast<void*>(header->entry);
		header->entry = nullptr;

		return res;
	}
}
