/**
 * @file memtable.cpp
 * @brief SafeCUDA memory functions
 *
 * Implementations for different parts of memtables
 *
 * @author Navin Kumar <navinkumar.ao2022@vitstudent.ac.in>
 * @date 2025-07-05
 * @version 0.0.2
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-07-05: Initial implementation
 * - 2025-10-22: Removed redundancies and fixed styling
 */

#include "memtable.h"

#include <cstdint>
#include <iostream>

namespace safecuda::memtable
{

std::uintptr_t *init_header(void *base_ptr)
{
	auto *header = static_cast<Header *>(base_ptr);
	header->magic_word = MAGIC_WORD;
	header->memory = reinterpret_cast<uintptr_t *>(
		reinterpret_cast<std::uint8_t *>(header) + sizeof(Header));
	header->entry = nullptr;

	return header->memory;
}

bool validate_header(void *base_ptr)
{
	const auto *header = reinterpret_cast<Header *>(
		static_cast<std::uint8_t *>(base_ptr) - sizeof(Header));

	if (header->magic_word == MAGIC_WORD) {
		return true;
	}
	return false;
}

void delete_entry(void *base_ptr)
{
	auto *header = static_cast<Header *>(base_ptr);
	header->entry = nullptr;
}
}
