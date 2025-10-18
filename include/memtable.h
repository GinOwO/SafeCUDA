/**
* @file memtable.h
 * @brief SafeCUDA metadata header file
 *
 * This file will contains the header file for the metadata table
 *
 * @author Anirudh <anirudh.sridhar2022@vitstudent.ac.in>
 * @date 2025-09-22
 * @version 0.0.1
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-09-22: Initial implementation
 */
#ifndef MEMTABLE_H
#define MEMTABLE_H

#include <cstdint>
#include <cstddef>

namespace safecuda::memtable{

	constexpr std::int16_t MAGIC_WORD = 0x5AFE;

	struct Entry {
		std::uint32_t start_addr;
		std::uint32_t block_size;
		std::uint8_t flags;
		std::uint32_t epochs;
	};

	struct Header {
		std::int16_t magic_word = MAGIC_WORD;
		std::int8_t _pad1[6];
		Entry *entry;
		std::uint32_t *memory;
	};

	std::uint32_t* init_header(void *basePtr, size_t block_size, std::uint8_t flags, std::uint32_t epochs, Entry *entry);
	bool validate_header(void *basePtr);
	void* delete_entry(void *basePtr);

}

#endif // !MEMTABLE_H
