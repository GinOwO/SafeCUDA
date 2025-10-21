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

	std::uintptr_t* init_header(void *basePtr){

		safecuda::memtable::Header *header = reinterpret_cast<safecuda::memtable::Header*>(basePtr);
		header->magic_word = safecuda::memtable::MAGIC_WORD;
		header->memory = reinterpret_cast<uintptr_t*>(reinterpret_cast<std::uint8_t*>(header) + sizeof(safecuda::memtable::Header));
		header->entry = nullptr;

		return header->memory;
	}

	bool validate_header(void* basePtr){
		safecuda::memtable::Header *header = reinterpret_cast<safecuda::memtable::Header*>(reinterpret_cast<std::uint8_t*>(basePtr) - sizeof(safecuda::memtable::Header));

		if(header->magic_word == safecuda::memtable::MAGIC_WORD){
			return true;
		}
		return false;
	}

	void delete_entry(void *basePtr){
		safecuda::memtable::Header *header = reinterpret_cast<safecuda::memtable::Header*>(basePtr);
		header->entry = nullptr;

		return;
	}
}
